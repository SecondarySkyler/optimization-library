from typing import Callable, Dict, Any

from ..optimization_parameters.optimization_parameters import OptimizationParameters
from ..optimization_parameters.optimizer_config import OptimizerConfig
from ..etl.extractors.provenance_extractor import ProvenanceExtractor
from ..optimization_parameters.directions import Directions
from ..utils.clustering import perform_clustering

import pandas as pd
import numpy as np
import yprov4ml
from pathlib import Path
import os
from bayesopt.bayesian_handler import BayesianOptimizer, OptimizationConfig


ObjectiveFunctionType = Callable[[Dict[str, Any]], Dict[str, Any]] # Maybe the return type should be more specific e.g. [str, float]

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
        self._runs = None # a pd.DataFrame to store the runs
    
    def _extract_provenance(self):
        """Method to extract provenance data based on optimization parameters.
           From the optimization parameters, get the input and output keys to extract.
           Calls the ETL module to perform the extraction.

           Returns: Two lists: inputs and outputs extracted from provenance. 
        """
        input, output = self.optimization_parameters._get_keys()
        extractor = ProvenanceExtractor(self.path_to_prov, {"input": input, "output": output})
        return extractor.extract_all()
    
    def _validate_evaluation_results(self, results: Dict[str, Any]):
        """Validate that the evaluation results contain all required output metrics.

           Args:
               results: A dictionary containing the evaluation results.

           Raises:
               ValueError: If any required output metric is missing in the results.
        """
        missing =  set(self.optimization_parameters.output) - results.keys()
        if missing:
            raise ValueError(f"Missing output metrics in evaluation results: {missing}")
        
    def _log_config(self, config: Dict[str, Any], results: Dict[str, Any]):
        """Log the configuration and results to provenance.
        """
        yprov4ml.start_run(
            prov_user_namespace="www.example.org",
            experiment_name=f"test_prov_experiment", 
            provenance_save_dir=self.path_to_prov,
            save_after_n_logs=100,
            collect_all_processes=False, 
            disable_codecarbon=True, 
            metrics_file_type=yprov4ml.MetricsType.NETCDF,
        )

        for key, value in config.items():
            yprov4ml.log_param(key, value, yprov4ml.Context.TRAINING)
        
        for key, value in results.items():
            yprov4ml.log_metric(key, value, yprov4ml.Context.TRAINING)
        
        yprov4ml.end_run(
            create_graph=False,
            create_svg=False,
            crate_ro_crate=False
        )
    

    def _generate_unified_log(self):
        """
        Generate a unified log of the experiment
        
        """
        path = Path(self.path_to_prov)
        experiment_dirs = [p for p in path.iterdir() if p.is_dir()]

        yprov4ml.start_run(
            prov_user_namespace="www.example.org",
            experiment_name=f"unified_experiment", 
            provenance_save_dir=self.path_to_prov,
            save_after_n_logs=100,
            collect_all_processes=False, 
            disable_codecarbon=True, 
            metrics_file_type=yprov4ml.MetricsType.NETCDF,
        )

        for experiment_dir in experiment_dirs:
            for file in os.listdir(experiment_dir):
                if file.endswith(".json"):
                    filename = os.path.join(experiment_dir, file)
                    yprov4ml.log_artifact("test", filename, yprov4ml.Context.VALIDATION)

        yprov4ml.end_run(
            create_graph=False,
            create_svg=False,
            crate_ro_crate=False
        )
    

        
    
    def run_clustering(self, X, method: str) -> pd.DataFrame:
        """Run clustering on the input data X using the specified method.
           
           Returns: DataFrame with the original data and cluster labels.    
        """
        model, labels = perform_clustering(X, method)
        df = pd.DataFrame(data=X, columns=self.optimization_parameters.input.get_keys())
        df['cluster'] = labels
        return df
    
    def optimize(self, objective_function: ObjectiveFunctionType, verbose: bool = False):
        """Main method to run the optimization loop.

           Args:
               objective_function: A callable that takes a dictionary of input parameters and evaluates the objective.
               verbose: If True, prints detailed logs during optimization.
            Returns: None  
        """
        
        # Generate the bounds from the Search Space
        bounds = self.optimization_parameters.input.bounds().T
        if verbose:
            print("Optimization Bounds:", bounds)

        for _ in range(self.n_iter):
            # Extract provenance data
            inp, out = self._extract_provenance()
            
            if verbose:
                print(f"Extracted {len(inp)} data points from provenance.")
                # for x, y in zip(inp, out):
                #     print("Input:", x, "Output:", y)


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

            # Now that we have the candidate, we need to:
            # 1. Cast the candidate to appropriate types (since the optimizer works with floats)
            # 2. Format it as a dictionary to be passed to the objective function
            # 3. Evaluate the objective function with the new candidate

            casted_candidate = {}

            for param, generated_val in zip(self.optimization_parameters.input.parameters, candidate):
                casted_candidate[param.name] = param.round(generated_val)


            # Evaluate the objective function with the new candidates
            # It is expected that the user implements this function
            evaluation_results = objective_function(casted_candidate)
            if isinstance(evaluation_results, dict):
                self._validate_evaluation_results(evaluation_results)
            else:
                raise TypeError("Objective function must return a dictionary.")
            print("Evaluated candidate:", casted_candidate, "Result:", evaluation_results)

            # Log the configuration and results to provenance
            self._log_config(casted_candidate, evaluation_results)
        
        # After optimization, generate a unified log of the experiment
        self._generate_unified_log()
    
    def results(self) -> pd.DataFrame:
        params, metrics = self._extract_provenance()

        df = pd.DataFrame(data=params, columns=self.optimization_parameters.input.get_keys())
        df[self.optimization_parameters.output] = metrics
        return df
    
    def best_configuration(self, metric_name: str = None) -> Dict[str, Any]:
        """Get the best configuration found during the optimization.
            If a metric_name is provided, return the configuration that optimizes that metric (accordingly to its direction).
            If no metric_name is provided and multiple outputs exist, return the Pareto front.

           Args:
               metric_name: The name of the metric to consider for best configuration.
                            If None, combine the metrics.

           Returns: A dictionary with the best input configuration and its corresponding metric value(s).
        """
        results_df = self.results()

        if metric_name is None and len(self.optimization_parameters.output) > 1:
            points = results_df[self.optimization_parameters.output]
            # paretoset lib requires the sense to be in the format of list of "max"/"min", so temporarily map directions and take first 3 letters
            senses = list(map(lambda d: d[:3], self.optimization_parameters.directions))
            mask = paretoset(points, sense=senses)
            return points[mask]
        else:
            
            if metric_name not in self.optimization_parameters.output:
                raise ValueError(f"Metric {metric_name} not found in optimization outputs.")
            
            # Find the index of the metric_name in self.optimization_parameters.output
            metric_index = self.optimization_parameters.output.index(metric_name)

            # Use the index to determine the optimization direction
            direction = self.optimization_parameters.directions[metric_index]

            if direction == Directions.MINIMIZE.value:
                best_row = results_df.loc[results_df[metric_name].idxmin()]
            else:  # direction == "maximize"
                best_row = results_df.loc[results_df[metric_name].idxmax()]
            best_configuration = {key: best_row[key] for key in self.optimization_parameters.input.get_keys()}
            best_configuration[metric_name] = best_row[metric_name]

            return best_configuration
    
    def shap_analysis(self):
        """Perform SHAP analysis on the optimization results.
           
           Returns: SHAP values and summary plot.
        """
        import shap

        results_df = self.results()
        X = results_df[self.optimization_parameters.input.get_keys()]
        y = results_df[self.optimization_parameters.output]

        # Train a surrogate model (e.g., Random Forest) to approximate the objective function
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor()
        model.fit(X, y)

        # Create SHAP explainer
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)

        # Plot SHAP summary
        shap.summary_plot(shap_values, X)

        return shap_values 
        




