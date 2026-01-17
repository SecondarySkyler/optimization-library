from library import Experiment, FloatParameter, IntParameter, OptimizationParameters, SearchSpace, OptimizerConfig

def test_clustering():
    lr = FloatParameter("LR", 0.0001, 0.001)
    batch_size = IntParameter("BATCH_SIZE", 16, 32)
    epochs = IntParameter("EPOCHS", 5, 10)
    search_space = SearchSpace(parameters=[lr, batch_size, epochs])

    opt_params = OptimizationParameters(
        input=search_space,
        output=["accuracy", "emissions"],
        directions=["maximize", "minimize"]
    )

    optimizer_cfg = OptimizerConfig(
        num_samples=1,
        num_restarts=200,
        acquisition_function="ucb",
        beta=1.0
    )

    exp = Experiment(
        optimization_parameters=opt_params,
        optimizer_config=optimizer_cfg,
        path_to_prov="./tests/prov", 
        n_iter=10
    )

    inp, out = exp._extract_provenance()
    print(exp.run_clustering(inp, "kmeans"))

