from typing import List, Tuple
from .directions import Directions
from .search_space import SearchSpace

class OptimizationParameters:
    def __init__(self, input: SearchSpace, output: List[str], directions: List[str | Directions]):
        self.input = input
        self.output = output
        self.directions = [Directions(direction) if isinstance(direction, str) else direction for direction in directions]
    

    def _get_keys(self) -> Tuple[List[str], List[str]]:
        input_keys = self.input.get_keys()
        output_keys = self.output
        return input_keys, output_keys
    
    # TODO: think if bounds() should be here or in SearchSpace
    # Maybe here is more clean becuase the lib interacts directly with OptimizationParameters rather than SearchSpace