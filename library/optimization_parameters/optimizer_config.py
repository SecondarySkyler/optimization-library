from enum import Enum

class AcquisitionFunction(str, Enum):
    QLOGEI = "qlogei" 
    QLOGNEI = "qlognei"
    UCB = "ucb"
    QLOGEHVI = "qlogehvi"

class OptimizerConfig:
    def __init__(
            self,
            num_samples: int,
            num_restarts: int,
            acquisition_function: AcquisitionFunction,
            beta: float, # Exploration-exploitation trade-off parameter
    ):
        self.num_samples = num_samples
        self.num_restarts = num_restarts
        self.acquisition_function = acquisition_function
        self.beta = beta