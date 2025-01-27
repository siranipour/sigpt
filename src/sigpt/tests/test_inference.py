from sigpt import inference


def test_truncate_to_eot():
    assert inference.truncate_to_eot([1, 2, 3, 4], 3) == [1, 2]
    assert inference.truncate_to_eot([1, 2, 3, 4], 5) == [1, 2, 3, 4]
