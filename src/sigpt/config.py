import dataclasses


@dataclasses.dataclass
class DDPConfig:
    rank: int
    world_size: int

    def __post_init__(self):
        if self.rank not in range(1, self.world_size + 1):
            raise ValueError(
                f"Rank {self.rank} must be in the range [1, {self.world_size})"
            )


@dataclasses.dataclass
class ModelConfig:
    block_size: int
    vocab_size: int
    n_embed: int
    n_heads: int
    n_layers: int


def get_gpt_config() -> ModelConfig:
    return ModelConfig(
        block_size=1024, vocab_size=50257, n_layers=12, n_heads=12, n_embed=768
    )
