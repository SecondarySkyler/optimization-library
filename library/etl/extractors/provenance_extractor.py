from pathlib import Path
import os
import numpy as np
from .extractor_factory import ExtractorFactory
import json

"""
optimization_params format
{
    'output': ['ACC_val', 'cpu_usage'],
    'input': ['param_lr', 'param_batch_size', 'param_epochs']   
}
"""
def find_key(data, target_key):
    if isinstance(data, dict):
        for key, value in data.items():
            if key.startswith("yProv4ML:"):
                # Remove the prefix for searching
                key = key[len("yProv4ML:"):]
            if key == target_key:
                return value
            result = find_key(value, target_key)
            if result is not None:
                return result

    elif isinstance(data, list):
        for item in data:
            result = find_key(item, target_key)
            if result is not None:
                return result

    return None

class ProvenanceExtractor:
    def __init__(self, provenance_folder, optimization_params):
        """
        Constructor for the ProvenanceExtractor.
        Parameters:
        - provenance_folder (str): Path to the folder containing provenance data.
        - optimization_params (dict): Dictionary containing input and output optimization parameters.
        """
        self.provenance_folder = Path(provenance_folder)
        self.optimization_params = optimization_params

    
    def _list_experiments(self):
        """
        Returns a list of experiment directories within the provenance folder.
        """
        return [
            p for p in self.provenance_folder.iterdir()
            if p.is_dir() and not p.name.startswith("unified_experiment")
        ]
    
    def _get_metrics_dir(self, experiment_path):
        """
        Returns a list of all files in the metrics*/ directory within the given experiment path.
        Parameters:
        - experiment_path (Path): Path to the experiment directory.
        Returns:
        - List[Path]: List of file paths in the metrics*/ directory.
        Raises:
        - FileNotFoundError: If the metrics*/ directory does not exist.
        """
        for entry in os.listdir(experiment_path):
            full_path = os.path.join(experiment_path, entry)

            if os.path.isdir(full_path) and entry.startswith("metrics"):
                metrics_dir = Path(full_path)
                return list(metrics_dir.glob("*.*"))
        
        raise FileNotFoundError(f"Metrics directory not found in {experiment_path}")

    
    def _extract_experiment_data(self, experiment_path):
        """
        Extracts data from a single experiment directory.
        It uses self.optimization_params to know which parameters to extract.

        Parameters:
        - experiment_path (Path): Path to the experiment directory.

        Returns:
        - Dict[str, Dict]: Dictionary containing extracted parameters and metrics.
        """
        
        params = {}

        # 1 - Find and parse the JSON
        for file in os.listdir(experiment_path):
            if file.endswith(".json"):
                filename = os.path.join(experiment_path, file)
                json_file = json.load(open(filename))
                keys = self.optimization_params["input"] + self.optimization_params["output"]

                # Extract all the params
                for k in keys:
                    value = find_key(json_file, k)

                    if value is not None:
                        params[k] = value
                    else:
                        raise Exception(f"Key {k} not found in experiment {experiment_path}")
                
                # Once found, no need to continue the search
                break

        return params 
    
    def __extract_from_file(self, filepath):
        """
        Method used to extract metric data from a given file.

        Parameters:
        - filepath (str): Path to the file containing metric data.

        Returns:
        - Dict[str, np.ndarray]: Dictionary containing the metric name and its corresponding values.
        """
        extractor = ExtractorFactory.get_extractor(filepath)
        data = extractor.extract(filepath)
        metric_name = data.attrs["_name"]
        metric_value = data["values"].values
        return { metric_name : metric_value }
    
    def __cast_to(self, key, value):
        """
        This method should perform a casting operation on a given metric.
        Currently the cast should go from a str -> float.
        The expected format should be similar to this:
        "key": {
            metric_value: ... <- str
            metric_type: ... <- a type
        }
        """

        if "$" not in value and "type" not in value:
            raise Exception(f"Cannot cast metric {key} with value {value}, probably we don't support this type yet.")

        # For the moment this is hardcoded since the JSON file does not contain the metrics specified as above
        sizes = {
            "small": 824682,
            "medium": 6576330,
            "large": 13130442
        }

        if key == "MODEL_SIZE":
            return np.float64(sizes[value])
        else:
            return np.float64(value["$"])

    def extract_all(self):
        """
        Extracts all input and output parameters from all experiments in the provenance folder.
        
        Returns:
        - Tuple[List[List[float]], List[List[float]]]: A tuple containing two lists:
            - The first list contains input parameters for each experiment.
            - The second list contains output parameters for each experiment.
        
        """
        input = []
        output = []
        for experiment_dir in self._list_experiments():
            params = self._extract_experiment_data(experiment_dir)

            for k, v in params.items():
                # Check v is a dict (probably will be removed) and the dict contains a path, this means we have to parse a file
                if isinstance(v, dict) and "yProv4ML:path" in v:
                    path = v["yProv4ML:path"]
                    data = self.__extract_from_file(path)
                    
                    if len(data[k]) > 1:
                        params[k] = data[k].sum().astype(np.float64)
                    else:
                        params[k] = data[k][0].astype(np.float64)

                else:
                    # Here we work with the values retrieved directly from the JSON (typically hyperparameters)
                    # so here we need to cast accordingly to the type of metric
                    params[k] = self.__cast_to(k, v)

            
            tmp_input = []
            tmp_output = []
            for key in self.optimization_params['input']:
                if key in params:
                    tmp_input.append(params[key])

            input.append(tmp_input)

            for key in self.optimization_params['output']:
                if key in params:
                    tmp_output.append(params[key])
            
            output.append(tmp_output)   
        return input, output
