import tiktoken

from sigpt import config, train

MICRO_BATCH_SIZE = 4
BATCH_SIZE = 4

if __name__ == "__main__":
    ddp = train.get_ddp_config()
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
    )
