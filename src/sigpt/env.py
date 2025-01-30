import enum
import os

import torch

from sigpt.config import DDPConfig


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


def get_quantized_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    # If using torch.float16, gradient scaling should be implemented in the training loop
    return torch.float32


def is_main_process() -> bool:
    ddp = get_ddp_config()
    return ddp is None or ddp.rank == 0
