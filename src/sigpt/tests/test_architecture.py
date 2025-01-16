import torch

from sigpt import architecture, config, sample

INPUT_PROMPT = "Hello, I am language model and"

EXPECTED_OUTPUTS = [
    "Hello, I am language model and encounteringboolë Scholarship outlandish",
    "Hello, I am language model andBalance Gazette sabotageKings mature",
]


def test_model_output(seed: int = 1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    mdl = architecture.Transformer(config.get_gpt_config())

    mdl_output = sample.generate(mdl, INPUT_PROMPT, batches=2, max_samples=5)

    assert len(mdl_output) == 2

    assert all(gen == exp for gen, exp in zip(mdl_output, EXPECTED_OUTPUTS))
