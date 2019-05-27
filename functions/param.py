from sklearn import model_selection
import functions.train
import tensorflow as tf
from joblib import Parallel, delayed
from pathlib import Path
import main
import time


def run(args, meta):
    experiment = main.prerun(args, exp=True)

    param_grid = args["param_grid"] 
    experiment.log_other("param_grid", param_grid)

    grid_iterator = model_selection.ParameterGrid(param_grid)

    def do(i, orig, gridpoint):
        args_copy = orig.copy()
        args_copy.update(gridpoint)

        args_copy["run_dir"] = str(Path(args["run_dir"], str(i)))

        return functions.train.run(args_copy, meta)

    for i, point in enumerate(grid_iterator):
        do(i, args, point)
    
