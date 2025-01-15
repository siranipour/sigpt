import pytest

from sigpt import data


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
    valid_config = data.DDPConfig(1, 10, 128)

    local_rank, rank, world_size = bad_config
    with pytest.raises(ValueError):
        data.DDPConfig(local_rank, rank, world_size)
