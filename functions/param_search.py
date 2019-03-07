from sklearn import model_selection
import functions.train
import tensorflow as tf
from joblib import Parallel, delayed
from pathlib import Path
import main
import time


def run(args, meta):
    experiment = main.prerun(args, exp=True)

    param_grid = [
        {"learning_rate": [0.001, 0.0001], "momentum": [0.99, 0.9], "optimizer": ["rmsprop"]},
        {"learning_rate": [0.001, 0.0001], "optimizer": ["adam"]}
    ]

    experiment.log_other("param_grid", param_grid)

    grid_iterator = model_selection.ParameterGrid(param_grid)

    def do(i, orig, gridpoint):
        tf.enable_eager_execution()

        args_copy = orig.copy()
        args_copy.update(gridpoint)

        args_copy["epochs"] = 1
        args_copy["run_dir"] = str(Path(args["run_dir"], str(i)))

        return functions.train.run(args_copy, meta)

    for i, point in enumerate(grid_iterator):
        do(i, args, point)
    
