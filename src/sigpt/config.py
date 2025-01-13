import dataclasses


@dataclasses.dataclass
class DDPConfig:
    local_rank: int
    rank: int
    world_size: int

    def __post_init__(self):
        if self.rank not in range(1, self.world_size + 1):
            raise ValueError(f"Rank {self.rank} must be in the range [1, {self.world_size})")
        if self.local_rank not in range(1, self.world_size + 1):
            raise ValueError(f"Rank {self.rank} must be in the range [1, {self.world_size})")


@dataclasses.dataclass
class ModelConfig:
    block_size: int
    vocab_size: int
    n_embed: int
    n_heads: int
    n_layers: int


@dataclasses.dataclass
class OptimizerConfig:
    weight_decay: float
    beta1: float
    beta2: float


@dataclasses.dataclass
class SchedulerConfig:
    warmup_steps: int
    total_steps: int
    min_lr: float
    max_lr: float


def get_optimizer_config() -> OptimizerConfig:
    return OptimizerConfig(weight_decay=0.1, beta1=0.9, beta2=0.95)


def get_scheduler_config() -> SchedulerConfig:
    return SchedulerConfig(warmup_steps=2000, total_steps=600000, min_lr=6e-5, max_lr=6e-4)


def get_gpt_config() -> ModelConfig:
    return ModelConfig(block_size=1024, vocab_size=50257, n_layers=12, n_heads=12, n_embed=768)
