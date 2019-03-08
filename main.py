import arguments
from comet_ml import Experiment, OfflineExperiment
import model
from pathlib import Path
import functions.param
import functions.train
import pandas as pd
import sys
import json

import tensorflow as tf

def prerun(args, exp=True):
    p = Path(args["run_dir"])
    if p.exists():
        raise ValueError("Rundir exists, please remove.")
    p.mkdir()

    with open(Path(p, "args.json"), "w") as fp:
        json.dump(args, fp)
    
    if exp:
        experiment = Experiment(
            api_key="pJ6UYxQwjYoYbCmmutkqP66ni",
            project_name=args["comet_project"],
            workspace="mlippie",
            auto_metric_logging=True,
            auto_param_logging=False,
            log_graph=True,
            disabled=True
        )

        experiment.log_parameters(args)

        return experiment
    else:
        return None


def main():
  
    args = arguments.get_args()

    meta = pd.read_csv(args["meta"])
        
    def summary():
        m = model.model_map(args["model"])(args)
        m.summary()

    def train():
        functions.train.run(args, meta)

    def cv():
        raise NotImplementedError()

    def predict():
        raise NotImplementedError()

    def param():
        functions.param.run(args, meta)

    function_map = {
        "train": train,
        "cv": cv,
        "predict": predict,
        "summary": summary,
        "param": param
    }
    
    function_map[args["function"]]()


if __name__ == "__main__":
    tf.enable_eager_execution()
    main()
