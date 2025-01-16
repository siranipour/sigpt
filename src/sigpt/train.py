import functools
import math
import os
import pathlib
import time

import tiktoken
import torch
import torch.nn as nn
import torch.optim as optim
from torch import distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb
from sigpt import architecture, data, log
from sigpt.config import DDPConfig, ModelConfig, OptimizerConfig, SchedulerConfig
from sigpt.env import Device

log = log.setup_logger()
EVAL_FREQUENCY: int = 100


def get_model_weights_path(root: str | None = None) -> pathlib.Path:
    root = root or os.getcwd()
    return pathlib.Path(root) / "model_state.pt"


def train(
    model_config: ModelConfig,
    optimizer_config: OptimizerConfig,
    scheduler_config: SchedulerConfig,
    encoder: tiktoken.Encoding,
    max_iters: int,
    micro_batch_size: int,
    batch_size: int,
    device: Device,
    model_checkpoint_path: pathlib.Path,
    eval_iters: int,
    ddp: DDPConfig | None = None,
    is_main_process: bool = True,
) -> None:
    torch.set_float32_matmul_precision("high")

    if ddp is not None:
        dist.init_process_group(backend="nccl")

    grad_accum_steps = compute_gradient_accumulation_steps(micro_batch_size, batch_size, ddp)
    log.info(f"Using gradient accumulation steps of {grad_accum_steps}")
    tokens_per_iter = (
        grad_accum_steps
        * batch_size
        * model_config.block_size
        * (1 if ddp is None else ddp.world_size)
    )
    log.info(f"Processing {tokens_per_iter} tokens per iteration")
    model = prepare_model(model_config, device, ddp)

    if is_main_process:
        wandb.watch(model, log_freq=EVAL_FREQUENCY)

    optimizer = prepare_optimizer(model, optimizer_config, scheduler_config)
    scheduler = prepare_scheduler(optimizer, scheduler_config)

    # Add +1 to the block size in order to slice out the next token as the target
    train_dl = data.fetch_dataset_loader(
        "train", encoder, batch_size, model_config.block_size + 1, num_workers=4, ddp=ddp
    )
    train_iterator = iter(train_dl)

    validation_dl = iter(
        data.fetch_dataset_loader(
            "validation",
            encoder,
            batch_size,
            model_config.block_size + 1,
            num_workers=4,
            ddp=ddp,
            shuffle=False,
        )
    )
    best_val_loss = float("inf")

    for idx in range(max_iters):
        timer_start = time.time()
        _ = optimizer.zero_grad()

        for micro_step in range(grad_accum_steps):
            example = next(train_iterator)
            x, y = example[..., :-1], example[..., 1:]
            x, y = map(functools.partial(tensor_to_device, device=device), (x, y))

            with torch.autocast(device_type=device.get_target(), dtype=torch.bfloat16):
                logits = model(x)
                loss = compute_loss(logits, y, grad_accum_steps)

            if ddp is not None and micro_step != grad_accum_steps - 1:
                with model.no_sync():
                    _ = loss.backward()
            else:
                _ = loss.backward()

        unclipped_grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), optimizer_config.max_grad_norm
        )
        _ = optimizer.step()
        _ = scheduler.step()
        timer_stop = time.time()

        if is_main_process and (idx % EVAL_FREQUENCY == 0):
            train_loss = compute_eval_loss(train_dl, model, eval_iters, device, ddp)
            validation_loss = compute_eval_loss(validation_dl, model, eval_iters, device, ddp)
            if validation_loss < best_val_loss:
                checkpoint_model(model, model_checkpoint_path)
            (lr,) = set(scheduler.get_last_lr())
            dt = timer_stop - timer_start
            wandb.log(
                {
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "lr": lr,
                    "unclipped_grad_norm": unclipped_grad_norm,
                    "dt/s": round(dt, 2),
                    "tokens/s": round(tokens_per_iter / dt, 2),
                },
                step=idx,
            )

    if ddp is not None:
        dist.destroy_process_group()

    checkpoint_model(model, model_checkpoint_path)


def checkpoint_model(model: DDP | nn.Module, path: pathlib.Path) -> None:
    state_dict = get_model_state_dict(model)
    torch.save(state_dict, path)
    wandb.save(path)


def get_model_state_dict(model: DDP | nn.Module) -> dict:
    if isinstance(model, DDP):
        return model.module.state_dict()
    return model.state_dict()


def compute_loss(logits: torch.Tensor, targets: torch.Tensor, norm: float) -> torch.Tensor:
    B, T, C = logits.shape
    return F.cross_entropy(logits.reshape(B * T, C), targets.reshape(B * T)) / norm


@torch.no_grad()
def compute_eval_loss(
    dataloader, model: nn.Module, validation_iters: int, device: Device, ddp: DDPConfig | None
) -> float:
    model.eval()
    # Call iter on validation data loader here to always use
    # the same set of validation examples
    it = iter(dataloader)
    total_loss = torch.zeros(1, device=device.get_target())
    for _ in range(validation_iters):
        example = next(it)
        x, y = example[..., :-1], example[..., 1:]
        x, y = map(functools.partial(tensor_to_device, device=device), (x, y))
        with torch.autocast(device_type=device.get_target(), dtype=torch.bfloat16):
            logits = model(x)
            total_loss += compute_loss(logits, y, 1)

    total_loss /= validation_iters

    if ddp is not None:
        dist.all_reduce(total_loss, op=dist.ReduceOp.AVG)
    model.train()
    return total_loss.item()


def tensor_to_device(t: torch.Tensor, device: Device) -> torch.Tensor:
    if device == Device.GPU:
        return t.pin_memory().to(device.get_target(), non_blocking=True)
    return t.to(device.get_target())


def prepare_model(
    model_config: ModelConfig, device: Device, ddp: DDPConfig | None = None
) -> nn.Module:
    model = architecture.Transformer(model_config)
    model.train()
    if device != Device.MPS:
        model = torch.compile(model)
    model.to(device.get_target())
    if ddp is not None:
        model = DDP(model, device_ids=[ddp.local_rank])
    return model


def prepare_optimizer(
    model: nn.Module, optimizer_config: OptimizerConfig, scheduler_config: SchedulerConfig
) -> optim.Optimizer:
    parameters = get_weight_decay_params(model, optimizer_config.weight_decay)
    return optim.AdamW(
        parameters,
        lr=scheduler_config.max_lr,
        betas=(optimizer_config.beta1, optimizer_config.beta2),
        fused=True,
    )


def get_weight_decay_params(model: nn.Module, weight_decay: float) -> list[dict]:
    to_decay, not_to_decay = [], []

    def filter_rule(name: str) -> bool:
        return "weight" in name and "ln" not in name

    for name, tensor in model.named_parameters():
        if not tensor.requires_grad:
            continue
        if filter_rule(name):
            assert tensor.ndim >= 2
            to_decay.append(tensor)
        else:
            not_to_decay.append(tensor)
    return [
        {"params": to_decay, "weight_decay": weight_decay},
        {"params": not_to_decay, "weight_decay": 0.0},
    ]


def prepare_scheduler(
    optimizer: torch.optim.Optimizer, scheduler_config: SchedulerConfig
) -> optim.lr_scheduler.LRScheduler:
    c = scheduler_config

    def lr_lambda(current_step: int) -> float:
        if current_step < c.warmup_steps:
            return c.max_lr * (current_step + 1) / (c.warmup_steps + 1)
        if current_step > c.total_steps:
            return c.min_lr
        T = (current_step - c.warmup_steps) / (c.total_steps - c.warmup_steps)
        return c.min_lr + 0.5 * (c.max_lr - c.min_lr) * (1 + math.cos(math.pi * T))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_gradient_accumulation_steps(
    micro_batch_size: int, batch_size: int, ddp: DDPConfig | None = None
) -> int:
    world_size = ddp.world_size if ddp is not None else 1
    return max(batch_size // (micro_batch_size * world_size), 1)
