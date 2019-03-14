import main
import functions.train
import tensorflow as tf
from pathlib import Path

def run(args, meta):
    experiment = main.prerun(args, exp=True)
    experiment.log_parameters(args)

    def do(i, orig, d):
        tf.enable_eager_execution()

        args_copy = orig.copy()
        args_copy["split_dir"] = str(d)

        return functions.train.run(args_copy, meta, new_run_dir=False)

    for i, d in enumerate(Path(args["split_dir"]).iterdir()):
        if d.is_dir():
            do(i, args, d)
