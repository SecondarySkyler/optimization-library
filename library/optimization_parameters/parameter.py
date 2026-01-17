from dataclasses import dataclass
from typing import Any

@dataclass
class Parameter:
    name: str

    def bounds(self) -> tuple[float, float]:
        raise NotImplementedError("Subclasses must implement this method.")
    
    def encode(self, value: Any) -> float:
        raise NotImplementedError("Subclasses must implement this method.")
    
    def decode(self, value: float) -> Any:
        raise NotImplementedError("Subclasses must implement this method.")