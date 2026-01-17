from typing import Callable, Dict, Any

from ..optimization_parameters.optimization_parameters import OptimizationParameters
from ..optimization_parameters.optimizer_config import OptimizerConfig
from ..etl.extractors.provenance_extractor import ProvenanceExtractor
from ..utils.clustering import perform_clustering

import pandas as pd
import numpy as np
from bayesopt.bayesian_handler import BayesianOptimizer, OptimizationConfig


ObjectiveFunctionType = Callable[[Dict[str, Any]], Any]

class Experiment:
    def __init__(
            self,
            optimization_parameters: OptimizationParameters,
            optimizer_config: OptimizerConfig,
            path_to_prov: str,
            n_iter: int, # Number of optimization iterations
    ):
        self.optimization_parameters = optimization_parameters
        self.optimizer_config = optimizer_config
        self.path_to_prov = path_to_prov
        self.n_iter = n_iter
    
    def _extract_provenance(self):
        input, output = self.optimization_parameters._get_keys()
        extractor = ProvenanceExtractor(self.path_to_prov, {"input": input, "output": output})
        return extractor.extract_all()
    
    def run_clustering(self, X, method: str) -> pd.DataFrame:
        """Run clustering on the input data X using the specified method.
            Returns a DataFrame with the original data and cluster labels.    
        """
        model, labels = perform_clustering(X, method)
        df = pd.DataFrame(data=X, columns=self.optimization_parameters.input.get_keys())
        df['cluster'] = labels
        return df
    
    def optimize(self, objective_function: ObjectiveFunctionType):
        
        # Generate the bounds from the Search Space
        bounds = self.optimization_parameters.input.bounds().T

        for _ in range(self.n_iter):
        # Extract provenance data
            inp, out = self._extract_provenance()
            # inp = [
            #     [np.float64(0.001), np.float64(32.0), np.float64(5.0)], 
            #     [np.float64(0.0001), np.float64(16.0), np.float64(10.0)]
            # ] 
            # out = [
            #     [np.float64(0.6785), np.float64(0.0006668533314950764)], 
            #     [np.float64(0.596), np.float64(0.0015004605520516634)]
            # ]


            # Initialize Bayesian Optimizer
            self.optimizer = BayesianOptimizer(
                OptimizationConfig(
                    self.optimization_parameters.output,
                    self.optimization_parameters.input.get_keys(),
                    self.optimization_parameters.directions,
                    ground_truth_dim=len(inp),
                    n_restarts=self.optimizer_config.num_restarts,
                    raw_samples=self.optimizer_config.num_samples,
                    acqf=self.optimizer_config.acquisition_function,
                    beta=self.optimizer_config.beta,
                ),          
                bounds=bounds,
            )

            # Prepare data for the optimizer
            data = {
                "parameters": inp,
                "metrics": out,
            }

            print(data)

            # Launch the bayesian process to get new candidates
            result = self.optimizer.run(data)

            print("Optimization Results:", result)

            # self.optimizer.prepare_data(data)

            self.optimizer.print_estimations(
                result.posterior.mean,
                result.posterior.variance.sqrt()
            )
            # print(self.optimizer.X_norm)

            # self.optimizer.model_training()
            # print("Surrogate Model trained.")

            # candidates, denorm_candidates, acq_value = self.optimizer.optimize()
            # print("Denormalized Candidates:", denorm_candidates)

            # # estimations = self.optimizer.estimate(denorm_candidates)
            # # print("Estimations:", estimations)
            # denorm_candidates = denorm_candidates[0]
            
            candidate = result.candidates[0]
            input_keys = self.optimization_parameters.input.get_keys()
            casted_candidates = {}
            for i, key in enumerate(input_keys):
                casted_candidates[key] = int(np.round(candidate[i]))

            # # # Now, round the denorm_candidates and format them as configurations
            # casted_candidates = {
            #     "LR": result.candidates[0][0],
            #     "BATCH_SIZE": int(np.round(result.candidates[0][1])),
            #     "EPOCHS": int(np.round(result.candidates[0][2])),
            # }

            print("Casted Candidates:", casted_candidates)

            # Finally, evaluate the objective function with the new candidates
            # It is expected that the user implements this function
            # It should also include the logic to log the results through yprov4ml
            objective_function(casted_candidates)



