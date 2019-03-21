import main
import functions.train
import tensorflow as tf
from pathlib import Path
import preprocessing

def run(args, meta):
    experiment = main.prerun(args, exp=True)
    experiment.log_parameters(args)
    
    data = preprocessing.load_hdf5_to_memory(args, meta["label"].values)

    def do(i, orig, d):
        tf.enable_eager_execution()

        args_copy = orig.copy()
        args_copy["split_dir"] = str(d)

        functions.train.run(args_copy, meta, new_run_dir=False, data=data)

    for i, d in enumerate(Path(args["split_dir"]).iterdir()):
        if d.is_dir():
            do(i, args, d)
