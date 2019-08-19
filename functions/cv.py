import main
import functions.train
import tensorflow as tf
from pathlib import Path
import preprocessing
from tensorflow.keras import callbacks as tf_callbacks
import model as modelmod

def run(args, meta, model, experiment, skip_n_folds=0):
    experiment.log_parameters(args)
    
    data = preprocessing.load_hdf5_to_memory(args, meta["label"].values)
    
    def do(i, d):
        args_copy = args.copy()
        args_copy["split_dir"] = str(d)
        
        log_dir = Path(args["run_dir"], str(i))
        log_dir.mkdir()

        callbacks = main.make_callbacks(args, experiment=experiment, run=str(log_dir))
        m = modelmod.build_model(args, m=model)

        functions.train.run(args_copy, meta, m, callbacks, data=data, id_=i, exp=experiment)

    for i, d in enumerate(Path(args["split_dir"]).iterdir()):
        if i < skip_n_folds:
            continue

        if d.is_dir():
            do(i, d)
