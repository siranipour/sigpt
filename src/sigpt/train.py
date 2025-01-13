import enum
import os

import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from sigpt import architecture, data
from sigpt.config import DDPConfig, ModelConfig


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
    encoder: tiktoken.Encoding,
    opt: torch.optim.Optimizer,
    batch_size: int,
    device: Device,
    ddp: DDPConfig | None = None,
) -> None:
    model = prepare_model(model_config, device, ddp)

    # Add +1 to the block size in order to slice out the next token as the target
    train_dl = data.fetch_dataset_loader(
        "train", encoder, batch_size, model_config.block_size + 1, num_workers=4, ddp=ddp
    )

    for example in train_dl:
        x, y = example[..., :-1], example[..., 1:]
        if device == Device.GPU:
            x, y = map(lambda t: t.pin_memory().to(device.get_target(), non_blocking=True), (x, y))
        else:
            x, y = map(lambda t: t.to(device.get_target()), (x, y))

        logits = model(x)
        loss = compute_loss(logits, y)


def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    B, T, C = logits.shape
    return F.cross_entropy(logits.reshape(B * T, C), targets.reshape(B * T))


def prepare_model(
    model_config: ModelConfig, device: Device, ddp: DDPConfig | None = None
) -> nn.Module:
    model = architecture.Transformer(model_config)
    if device != Device.MPS:
        model = torch.compile(model)
    model.to(device.get_target())
    if ddp is not None:
        model = DDP(model, device_ids=[ddp.rank])
    return model


def prepare_optimizer() -> torch.optim.Optimizer:
    pass


def get_device() -> Device:
    if torch.cuda.is_available():
        return Device.GPU
    if torch.mps.is_available():
        return Device.MPS
    return Device.CPU


def is_ddp() -> bool:
    return ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ)


def get_ddp_config() -> DDPConfig | None:
    if not is_ddp():
        return None
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    return DDPConfig(rank, world_size)
