import main
import functions.train
import tensorflow as tf
from pathlib import Path
import preprocessing

def run(args, meta):
    new_run_dir = "skip_n_folds" not in args
    experiment = main.prerun(args, run_dir=new_run_dir, exp=True)
    experiment.log_parameters(args)
    
    data = preprocessing.load_hdf5_to_memory(args, meta["label"].values)

    def do(i, orig, d):
        args_copy = orig.copy()
        args_copy["split_dir"] = str(d)

        functions.train.run(args_copy, meta, new_run_dir=False, data=data, id_=i)

    for i, d in enumerate(Path(args["split_dir"]).iterdir()):
        if not new_run_dir:
            if i < args["skip_n_folds"]:
                continue

        if d.is_dir():
            do(i, args, d)
