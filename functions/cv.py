import main
import functions.train
import tensorflow as tf
from pathlib import Path
import preprocessing
from tensorflow.keras import callbacks as tf_callbacks

def run(args, meta, model, callbacks, experiment, skip_n_folds=0):
    experiment.log_parameters(args)
    
    data = preprocessing.load_hdf5_to_memory(args, meta["label"].values)
    
    def do(i, orig, d):
        args_copy = orig.copy()
        args_copy["split_dir"] = str(d)

        functions.train.run(args_copy, meta, model, callbacks, data=data, id_=i, exp=experiment)

    for i, d in enumerate(Path(args["split_dir"]).iterdir()):
        if i < skip_n_folds:
            continue

        if d.is_dir():
            tb_log_dir = Path(args["run_dir"], str(i))
            tb_log_dir.mkdir()

            # Set log dir or create TensorBoard callback
            found = False
            for cb in callbacks:
                if type(cb)==tf.keras.callbacks.TensorBoard:
                    found = True
                    cb.log_dir = str(tb_log_dir)
            if not found:
                tb = tf_callbacks.TensorBoard(log_dir=tb_log_dir, histogram_freq=1, profile_batch=3, write_graph=True)
                callbacks.append(tb)

            do(i, args, d)
