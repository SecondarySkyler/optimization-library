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
        """Method to extract provenance data based on optimization parameters.
           From the optimization parameters, get the input and output keys to extract.
           Calls the ETL module to perform the extraction.

           Returns: Two lists: inputs and outputs extracted from provenance. 
        """
        input, output = self.optimization_parameters._get_keys()
        extractor = ProvenanceExtractor(self.path_to_prov, {"input": input, "output": output})
        return extractor.extract_all()
    
    def run_clustering(self, X, method: str) -> pd.DataFrame:
        """Run clustering on the input data X using the specified method.
           
           Returns: DataFrame with the original data and cluster labels.    
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
                    beta=self.optimizer_config.beta
                ),          
                bounds=bounds,
            )

            # Prepare data for the optimizer
            data = {
                "parameters": inp,
                "metrics": out,
            }

            # Launch the bayesian process to get new candidates
            result = self.optimizer.run(data)

            candidate = result.candidates[0] # Since we are generating one candidate at a time

            # Cast the candidate to appropriate types and format it as a dictionary
            input_keys = self.optimization_parameters.input.get_keys()
            casted_candidate = {}
            for i, key in enumerate(input_keys):
                # casted_candidate[key] = int(np.round(candidate[i]))
                casted_candidate[key] = candidate[i]


            # Evaluate the objective function with the new candidates
            # It is expected that the user implements this function
            evaluation_results = objective_function(casted_candidate)
            print("Evaluated candidate:", casted_candidate, "Result:", evaluation_results)
    
    def results(self) -> pd.DataFrame:
        params, metrics = self._extract_provenance()

        df = pd.DataFrame(data=params, columns=self.optimization_parameters.input.get_keys())
        df[self.optimization_parameters.output] = metrics
        return df
    
    def best_configuration(self, metric_name: str = None) -> Dict[str, Any]:
        """Get the best configuration found during the optimization.

           Args:
               metric_name: The name of the metric to consider for best configuration.
                            If None, combine the metrics.

           Returns: A dictionary with the best input configuration and its corresponding metric value(s).
        """
        results_df = self.results()

        if metric_name is None:
            print("Missing implementation for multi-metric best configuration.")
        else:
            
            if metric_name not in self.optimization_parameters.output:
                raise ValueError(f"Metric {metric_name} not found in optimization outputs.")
            
            # Find the index of the metric_name in self.optimization_parameters.output
            metric_index = self.optimization_parameters.output.index(metric_name)

            # Use the index to determine the optimization direction
            direction = self.optimization_parameters.directions[metric_index]

            if direction == "minimize":
                best_row = results_df.loc[results_df[metric_name].idxmin()]
            else:  # direction == "maximize"
                best_row = results_df.loc[results_df[metric_name].idxmax()]
            best_configuration = {key: best_row[key] for key in self.optimization_parameters.input.get_keys()}
            best_configuration[metric_name] = best_row[metric_name]

            return best_configuration
        




