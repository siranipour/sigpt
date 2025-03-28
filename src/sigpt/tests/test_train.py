from sigpt import config, train


def test_grad_accum():
    assert train.compute_gradient_accumulation_steps(4, 4) == 1
    assert train.compute_gradient_accumulation_steps(4, 8) == 2
    assert train.compute_gradient_accumulation_steps(4, 8, config.DDPConfig(1, 1, 2)) == 1
