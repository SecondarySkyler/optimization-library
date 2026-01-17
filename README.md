# Optimization Library
This library provides a set of functionalities for performing hyperparameter tuning leveraging bayesian optimization.

## Installation
First, start by cloning the repository:
```bash
git clone
```

Then, navigate to the project directory:
```bashcd optimization-library
cd optimization-library
```

Finally, install the library:
```bash
pip install .
```

## Usage
To use the optimization library, please refer to the ```examples/``` directory for sample scripts demonstrating how to set up and run hyperparameter tuning tasks.

## Project Structure
```bash
├── LICENSE
├── README.md
├── examples
├── library
│   ├── core
│   │   └── experiment.py
│   ├── etl
│   │   └── extractors
│   │       ├── __init__.py
│   │       ├── base.py
│   │       ├── csv_extractor.py
│   │       ├── extractor_factory.py
│   │       ├── netcdf_extractor.py
│   │       ├── provenance_extractor.py
│   │       └── zarr_extractor.py
│   ├── optimization_parameters
│   │   ├── __init__.py
│   │   ├── categorical_parameter.py
│   │   ├── directions.py
│   │   ├── float_parameter.py
│   │   ├── int_parameter.py
│   │   ├── optimization_parameters.py
│   │   ├── optimizer_config.py
│   │   ├── parameter.py
│   │   └── search_space.py
│   └── utils
│       ├── __init__.py
│       └── clustering.py
├── pyproject.toml
├── requirements.txt
└── tests
```

## Author
- [@Cristian Murtas](https://github.com/SecondarySkyler)