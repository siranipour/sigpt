from sigpt import sample

def test_truncate_to_eot():
    assert sample.truncate_to_eot([1, 2, 3, 4], 3) == [1, 2]
    assert sample.truncate_to_eot([1, 2, 3, 4], 5) == [1, 2, 3, 4]
