from dataclasses import dataclass
from .parameter import Parameter
from typing import Any, List

@dataclass
class CategoricalParameter(Parameter):
    categories: List[Any]

    def bounds(self) -> tuple[float, float]:
        return (0.0, len(self.categories) - 1.0)
    
    def encode(self, value: Any) -> float:
        if value not in self.categories:
            raise ValueError(f"Value {value} not in categories {self.categories}.")
        return float(self.categories.index(value))
    
    def decode(self, value: float) -> Any:
        index = int(round(value))
        if index < 0 or index >= len(self.categories):
            raise ValueError(f"Encoded value {value} is out of bounds for categories.")
        return self.categories[index]
