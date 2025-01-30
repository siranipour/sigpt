import tempfile

import torch

from sigpt import architecture, config, train

EXPECTED_GPT2_PARAMETER_COUNT: int = 124_439_808


def test_model_persistence():
    mdl = architecture.Transformer(config.get_gpt_config())

    with tempfile.TemporaryDirectory() as tempdir:
        path = train.get_model_weights_path(root=tempdir)
        torch.save(mdl.state_dict(), path)
        assert path.exists()


def test_model_parameter_count():
    model = architecture.Transformer(config.get_gpt_config())
    assert architecture.count_trainable_parameters(model) == EXPECTED_GPT2_PARAMETER_COUNT
