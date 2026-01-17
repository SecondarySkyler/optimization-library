from library import Experiment, OptimizationParameters, FloatParameter, IntParameter, SearchSpace

def fake_objective_function(params):
    return

def test_experiment():
    lr = FloatParameter("LR", 0.0001, 0.001)
    batch_size = IntParameter("BATCH_SIZE", 16, 32)
    epochs = IntParameter("EPOCHS", 5, 10)
    search_space = SearchSpace(parameters=[lr, batch_size, epochs])

    opt_params = OptimizationParameters(
        input=search_space,
        output=["test", "val_loss"],
        directions=["minimize", "minimize"]
    )
    exp = Experiment(optimization_parameters=opt_params, path_to_prov="path/to/provenance", n_iter=10)

    exp.optimize(fake_objective_function({"lr": 5.0, "param2": 25}))