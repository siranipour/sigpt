import tempfile

import torch

from sigpt import architecture, config, inference, train

INPUT_PROMPT: str = "Hello, I am language model and"

EXPECTED_OUTPUTS: list[str] = [
    "Hello, I am language model and encounteringboolë Scholarship outlandish",
    "Hello, I am language model andBalance Gazette sabotageKings mature",
]

EXPECTED_GPT2_PARAMETER_COUNT: int = 124_439_808


def test_model_output(seed: int = 1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    mdl = architecture.Transformer(config.get_gpt_config())

    mdl_output = inference.generate(mdl, INPUT_PROMPT, batches=2, max_samples=5)

    assert len(mdl_output) == 2

    assert all(gen == exp for gen, exp in zip(mdl_output, EXPECTED_OUTPUTS))


def test_model_persistence():
    mdl = architecture.Transformer(config.get_gpt_config())

    with tempfile.TemporaryDirectory() as tempdir:
        path = train.get_model_weights_path(root=tempdir)
        torch.save(mdl.state_dict(), path)
        assert path.exists()


def test_model_parameter_count():
    model = architecture.Transformer(config.get_gpt_config())
    assert architecture.count_trainable_parameters(model) == EXPECTED_GPT2_PARAMETER_COUNT
