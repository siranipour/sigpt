import tiktoken
from sigpt import config, train

BATCH_SIZE = 1

if __name__ == "__main__":
    ddp = train.get_ddp_config()
    model_config = config.get_gpt_config()
    device = train.get_device()
    encoder = tiktoken.get_encoding("gpt2")
    train.train(model_config, encoder, None, BATCH_SIZE, device, ddp)
