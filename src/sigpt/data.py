import datasets
import tiktoken
import torch
from datasets.distributed import split_dataset_by_node
from torch.utils.data import DataLoader, IterableDataset

from sigpt.config import DDPConfig

DATASET_PATH: str = "allenai/c4"
DATASET_NAME: str = "en"
FEATURE_NAME: str = "text"


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
    # Buffer size to facilitate shuffling the dataset.
    # This buffer is populated in a deterministic manner, but samples are drawn randomly.
    BUFFER_SIZE: int = 1000
    SEED: int = 42

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

        self._ds = (
            datasets.load_dataset(DATASET_PATH, DATASET_NAME, streaming=True, split=split)
            .select_columns(FEATURE_NAME)
            .shuffle(seed=self.SEED, buffer_size=self.BUFFER_SIZE)
        )

        if ddp is not None:
            self._ds = split_dataset_by_node(self._ds, ddp.rank, ddp.world_size)

    def __iter__(self):
        buffer = []
        for datum in self._ds:
            while len(buffer) < self.block_size:
                encoded = self.encoder.encode(datum[FEATURE_NAME]) + [self.encoder.eot_token]
                buffer.extend(encoded)
            yield torch.tensor(buffer[: self.block_size], dtype=torch.long)
            buffer = buffer[self.block_size :]
