import pytest
import torch

from sigpt.model import architecture, sample

INPUT_PROMPT = "Hello, I am language model and"

EXPECTED_OUTPUTS = [
    "Hello, I am language model and encounteringieselë franc graduation",
    "Hello, I am language model and Pratt Mat Accord NEOwhy",
]

def test_model_output(seed: int=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    mdl = architecture.Transformer(architecture.GPTConfig)

    mdl_output = sample.generate(mdl, INPUT_PROMPT, batches=2, max_samples=5)

    assert len(mdl_output) == 2
    assert all(gen == exp for gen, exp in list(zip(mdl_output, EXPECTED_OUTPUTS)))

