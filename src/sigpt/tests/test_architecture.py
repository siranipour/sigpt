import torch

from sigpt import architecture, config, sample

INPUT_PROMPT = "Hello, I am language model and"

EXPECTED_OUTPUTS_NO_FLASH_ATTENTION = [
    "Hello, I am language model and encounteringieselë franc graduation",
    "Hello, I am language model and Pratt Mat Accord NEOwhy",
]

EXPECTED_OUTPUTS_WITH_FLASH_ATTENTION = [
    "Hello, I am language model and encounteringboolë Scholarship outlandish",
    "Hello, I am language model andBalance Gazette sabotageKings mature",
]


def test_model_output(seed: int = 1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    mdl = architecture.Transformer(config.get_gpt_config())

    mdl_output = sample.generate(mdl, INPUT_PROMPT, batches=2, max_samples=5)

    assert len(mdl_output) == 2

    if not architecture.USE_FLASH_ATTENTION:
        assert all(gen == exp for gen, exp in zip(mdl_output, EXPECTED_OUTPUTS_NO_FLASH_ATTENTION))
    else:
        assert all(
            gen == exp for gen, exp in zip(mdl_output, EXPECTED_OUTPUTS_WITH_FLASH_ATTENTION)
        )
