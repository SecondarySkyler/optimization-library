from dataclasses import dataclass
from .parameter import Parameter

@dataclass
class IntParameter(Parameter):
    lower: int
    upper: int

    def bounds(self) -> tuple[float, float]:
        return (self.lower, self.upper)

    def encode(self, value: int) -> float:
        return (value - self.lower) / (self.upper - self.lower)

    def decode(self, value: float) -> int:
        return int(round(self.lower + value * (self.upper - self.lower)))