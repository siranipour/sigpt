import pytest
import tiktoken

from sigpt import data

@pytest.mark.parametrize("block_size", [1, 32, 1024])
@pytest.mark.parametrize("batch_size", [1, 32])
def test_block_size_dataset(block_size: int, batch_size: int):
    encoder = tiktoken.get_encoding("gpt2")
    dl = data.fetch_dataset_loader("train", encoder, batch_size, block_size)
    assert all(next(iter(dl)).shape == (batch_size, block_size))

    ddp1 = data.DDPConfig(1, 2)
    ddp2 = data.DDPConfig(2, 2)
    dl1 = data.fetch_dataset_loader("train", encoder, batch_size, block_size, ddp=ddp1)
    dl2 = data.fetch_dataset_loader("train", encoder, batch_size, block_size, ddp=ddp2)
    assert not (next(iter(dl1)) == (next(iter(dl2)))).any()


@pytest.mark.parametrize("block_size", [1, 32, 1024])
def test_block_size_dataset(block_size: int):
    encoder = tiktoken.get_encoding("gpt2")
    ds = data.BlockSizedDataset(block_size, "train", encoder)
    assert len(next(iter(ds))) == block_size


def test_ddpconfig():
    valid_config = data.DDPConfig(10, 128)

    bad_configs = [
        (10, 5),
        (0, 10),
        (-5, -10),
        (-10, -5),
    ]
    for (rank, world_size) in bad_configs:
        with pytest.raises(ValueError):
            data.DDPConfig(rank, world_size)

