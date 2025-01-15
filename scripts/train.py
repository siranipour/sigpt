import tiktoken

import wandb
from sigpt import config, train

MICRO_BATCH_SIZE = 4
BATCH_SIZE = 4
PROJECT_NAME = "sigpt"
RUN_NAME = "training-run"

if __name__ == "__main__":
    ddp = train.get_ddp_config()
    is_main_process = ddp is None or ddp.rank == 0

    if is_main_process:
        wandb.login()
        wandb.init(project=PROJECT_NAME, name=RUN_NAME, entity="s-iranipour-siranipour-io")

    model_config = config.get_gpt_config()
    optimizer_config = config.get_optimizer_config()
    scheduler_config = config.get_scheduler_config()
    device = train.get_device()
    device = train.Device.CPU
    encoder = tiktoken.get_encoding("gpt2")

    train.train(
        model_config,
        optimizer_config,
        scheduler_config,
        encoder,
        MICRO_BATCH_SIZE,
        BATCH_SIZE,
        device,
        ddp,
        is_main_process,
    )
