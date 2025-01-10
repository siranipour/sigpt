import dataclasses
import functools

import datasets
import tiktoken
import torch
from datasets.distributed import split_dataset_by_node
from torch.utils.data import DataLoader, IterableDataset

DATASET_PATH: str = "allenai/c4"
DATASET_NAME: str = "en"
FEATURE_NAME: str = "text"


@dataclasses.dataclass
class DDPConfig:
    rank: int
    world_size: int

    def __post_init__(self):
        if self.rank not in range(1, self.world_size + 1):
            raise ValueError(
                f"Rank {self.rank} must be in the range [1, {self.world_size})"
            )


def fetch_dataset_loader(
    split: str,
    encoder: tiktoken.Encoding,
    batch_size: int,
    block_size: int,
    num_workers: int = 1,
    prefetch_factor: int = 32,
    ddp: DDPConfig | None = None,
):
    ds = BlockSizedDataset(block_size, split, encoder, ddp)
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(ddp is not None),
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
    )


class BlockSizedDataset(IterableDataset):
    def __init__(
        self,
        block_size: int,
        split: str,
        encoder: tiktoken.Encoding,
        ddp: DDPConfig | None = None,
    ):
        super().__init__()
        self.block_size = block_size
        self.encoder = encoder
        self.ddp = ddp

        self._ds = datasets.load_dataset(
            DATASET_PATH, DATASET_NAME, streaming=True, split=split
        ).select_columns(FEATURE_NAME)

        if ddp is not None:
            self._ds = split_dataset_by_node(self._ds, ddp.rank, ddp.world_size)

    def __iter__(self):
        buffer = []
        for datum in self._ds:
            while len(buffer) < self.block_size:
                encoded = self.encoder.encode(datum[FEATURE_NAME]) + [
                    self.encoder.eot_token
                ]
                buffer.extend(encoded)
            yield torch.tensor(buffer[: self.block_size], dtype=torch.long)
            buffer = buffer[self.block_size :]
