from dataclasses import dataclass
from .parameter import Parameter

@dataclass
class FloatParameter(Parameter):
    lower: float
    upper: float

    def bounds(self) -> tuple[float, float]:
        return (self.lower, self.upper)

    def encode(self, value: float) -> float:
        return (value - self.lower) / (self.upper - self.lower)

    def decode(self, value: float) -> float:
        return self.lower + value * (self.upper - self.lower)
    
    def round(self, value) -> float:
        return value