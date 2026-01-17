import torch
from .parameter import Parameter
from typing import List

class SearchSpace:
    def __init__(self, parameters: List[Parameter]):
        self.parameters = parameters
    
    def bounds(self) -> torch.Tensor:
        margin = 0.02
        bounds = []
        for param in self.parameters:
            low, high = param.bounds()
            range = (high - low) * margin
            low -= range
            high += range
            if low < 0.0:
                low = 0.0
            bounds.append((low, high))
        return torch.tensor(bounds, dtype=torch.float64)
    
    def encode(self, config: dict) -> torch.Tensor:
        encoded = [param.encode(config[param.name]) for param in self.parameters]
        return torch.tensor(encoded, dtype=torch.float32)
    
    def decode(self, encoded: torch.Tensor) -> dict:
        return {
            param.name: param.decode(encoded[i].item())
            for i, param in enumerate(self.parameters)
        }

    def get_keys(self) -> List[str]:
        return [param.name for param in self.parameters]
