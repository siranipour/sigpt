import pytest

from sigpt import config


@pytest.mark.parametrize(
    "bad_config",
    [
        # Rank is too large
        (1, 10, 5),
        (10, 0, 10),
        (1, -5, -10),
        (1, -10, -5),
        # Local rank is too large
        (10, 1, 5),
        (0, 10, 10),
        (-5, 1, -10),
        (-10, 1, -5),
    ],
)
def test_ddpconfig(bad_config: tuple[int, int, int]):
    valid_config = config.DDPConfig(1, 10, 128)
    assert valid_config.local_rank == 1
    assert valid_config.rank == 10
    assert valid_config.world_size == 128

    local_rank, rank, world_size = bad_config
    with pytest.raises(ValueError):
        config.DDPConfig(local_rank, rank, world_size)


def test_optimizer_config():
    with pytest.raises(ValueError):
        config.OptimizerConfig(0.1, 0.9, 0.95, -1.0)
