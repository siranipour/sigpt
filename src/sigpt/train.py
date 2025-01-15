import enum
import math
import os
import time

import tiktoken
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributed import destroy_process_group, init_process_group
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb
from sigpt import architecture, data
from sigpt.config import DDPConfig, ModelConfig, OptimizerConfig, SchedulerConfig

EVAL_FREQUENCY: int = 2000


class Device(enum.Enum):
    CPU = "cpu"
    MPS = "mps"
    GPU = "cuda"

    def get_target(self) -> str:
        if self.value != "cuda":
            return self.value
        if is_ddp():
            return f"cuda:{get_ddp_config().rank}"
        return "cuda"


def train(
    model_config: ModelConfig,
    optimizer_config: OptimizerConfig,
    scheduler_config: SchedulerConfig,
    encoder: tiktoken.Encoding,
    micro_batch_size: int,
    batch_size: int,
    device: Device,
    ddp: DDPConfig | None = None,
    is_main_process: bool = True,
) -> None:
    torch.set_float32_matmul_precision("high")

    if ddp is not None:
        init_process_group(backend="nccl")

    grad_accum_steps = compute_gradient_accumulation_steps(micro_batch_size, batch_size, ddp)
    tokens_per_iter = (
        grad_accum_steps
        * batch_size
        * model_config.block_size
        * (1 if ddp is None else ddp.world_size)
    )
    model = prepare_model(model_config, device, ddp)

    if is_main_process:
        wandb.watch(model, log_freq=EVAL_FREQUENCY)

    optimizer = prepare_optimizer(model, optimizer_config, scheduler_config)
    scheduler = prepare_scheduler(optimizer, scheduler_config)

    # Add +1 to the block size in order to slice out the next token as the target
    train_dl = data.fetch_dataset_loader(
        "train", encoder, batch_size, model_config.block_size + 1, num_workers=4, ddp=ddp
    )
    data_gen = iter(train_dl)

    idx = 0
    # TODO: think about how to break from here
    while True:
        timer_start = time.time()
        _ = optimizer.zero_grad()

        for micro_step in range(grad_accum_steps):
            example = next(data_gen)
            x, y = example[..., :-1], example[..., 1:]
            if device == Device.GPU:
                x, y = map(
                    lambda t: t.pin_memory().to(device.get_target(), non_blocking=True), (x, y)
                )
            else:
                x, y = map(lambda t: t.to(device.get_target()), (x, y))

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
            (lr,) = set(scheduler.get_last_lr())
            dt = timer_stop - timer_start
            wandb.log(
                {
                    "train_loss": loss.item(),
                    "lr": lr,
                    "unclipped_grad_norm": unclipped_grad_norm,
                    "dt/s": round(dt, 2),
                    "tokens/s": round(tokens_per_iter / dt, 2),
                },
                step=idx,
            )

        idx += 1

    if ddp is not None:
        destroy_process_group()


def compute_loss(logits: torch.Tensor, targets: torch.Tensor, norm: float) -> torch.Tensor:
    B, T, C = logits.shape
    return F.cross_entropy(logits.reshape(B * T, C), targets.reshape(B * T)) / norm


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


def get_device() -> Device:
    if torch.cuda.is_available():
        return Device.GPU
    if torch.mps.is_available():
        return Device.MPS
    return Device.CPU


def is_ddp() -> bool:
    return ("LOCAL_RANK" in os.environ) and ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ)


def get_ddp_config() -> DDPConfig | None:
    if not is_ddp():
        return None
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    return DDPConfig(local_rank, rank, world_size)


def compute_gradient_accumulation_steps(
    micro_batch_size: int, batch_size: int, ddp: DDPConfig | None = None
) -> int:
    world_size = ddp.world_size if ddp is not None else 1
    assert batch_size % (micro_batch_size * world_size) == 0
    return batch_size // (micro_batch_size * world_size)
