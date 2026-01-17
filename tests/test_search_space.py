import torch

from library import SearchSpace, FloatParameter, IntParameter


def test_search_space():
    param1 = FloatParameter("param1", 0.0, 10.0)
    param2 = IntParameter("param2", 20, 30)
    search_space = SearchSpace(parameters=[param1, param2])

    lows, highs = search_space.bounds()
    print(lows, highs)
    assert ((lows == torch.tensor([0.0, 0.0])).all())
    assert ((highs == torch.tensor([1.0, 1.0])).all())
    
    config = {"param1": 5, "param2": 25}
    encoded = search_space.encode(config)
    assert ((encoded == torch.tensor([0.5, 0.5])).all())

    encoded = torch.tensor([0.5, 0.5])
    decoded = search_space.decode(encoded)
    assert decoded == {"param1": 5.0, "param2": 25}
