import tiktoken

import wandb
from sigpt import config, env, train
from sigpt.logging import log

MICRO_BATCH_SIZE: int = 24
BATCH_SIZE: int = 480
MAX_ITERS: int = 20_000
EVAL_ITERS: int = 100
PROJECT_NAME: str = "sigpt"
RUN_NAME: str = "training-run"

if __name__ == "__main__":
    ddp = env.get_ddp_config()
    is_main_process = env.is_main_process()

    if is_main_process:
        wandb.login()
        wandb.init(project=PROJECT_NAME, name=RUN_NAME, entity="s-iranipour-siranipour-io")

    if ddp is not None:
        log.info(f"Running with world size of {ddp.world_size}")

    model_config = config.get_gpt_config()
    optimizer_config = config.get_optimizer_config()
    scheduler_config = config.get_scheduler_config()
    device = env.get_device()
    encoder = tiktoken.get_encoding("gpt2")

    checkpoint_path = train.get_model_weights_path()

    train.train(
        model_config,
        optimizer_config,
        scheduler_config,
        encoder,
        MAX_ITERS,
        MICRO_BATCH_SIZE,
        BATCH_SIZE,
        device,
        checkpoint_path,
        eval_iters=EVAL_ITERS,
        ddp=ddp,
        is_main_process=is_main_process,
    )

    if is_main_process:
        wandb.finish()
