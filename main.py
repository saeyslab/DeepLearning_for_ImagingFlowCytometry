import arguments
from comet_ml import Experiment, OfflineExperiment
import model
from pathlib import Path
import functions.param
import functions.train
import functions.cv
import functions.embed
import functions.tb_embed
import functions.predict
import pandas as pd
import sys
import json
import socket

import tensorflow as tf
from tensorflow.keras import callbacks as tf_callbacks
import callbacks as my_callbacks
from tensorflow.python.ops import summary_ops_v2

def prerun(args, run_dir=True, exp=True):
    if run_dir:
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
            disabled="maximl" in socket.gethostname() # disable on dev machine
        )

        experiment.log_parameters(args)

        return experiment
    else:
        return None


def make_callbacks(args, experiment=None, run=None):

    if run is None:
        run = args["run_dir"]

    writer = summary_ops_v2.create_file_writer_v2(run)

    def schedule(epoch):
        if epoch < args["warmup_length"]:
            return args["warmup_coeff"]*args["learning_rate"]
        else:
            return args["learning_rate"]

    cb = [
        tf_callbacks.ModelCheckpoint(str(Path(run, "model.hdf5")), verbose=0, save_freq='epoch'),
        tf_callbacks.ModelCheckpoint(
            str(Path(run, "best-model.hdf5")), 
            verbose=0, 
            save_freq='epoch',
            save_best_only=True,
            mode="max",
            monitor="val_balanced_accuracy",
        ),
        tf_callbacks.EarlyStopping(
            monitor="val_balanced_accuracy", 
            patience=args["es_patience"],
            mode="max",
            min_delta=args["es_epsilon"]
        ),
        # tf_callbacks.ReduceLROnPlateau(
        #     monitor="val_balanced_accuracy", min_delta=args["lrplat_epsilon"],
        #     factor=0.5, patience=int(args["lrplat_patience"])
        # ),
        tf_callbacks.LearningRateScheduler(schedule),
        tf_callbacks.CSVLogger(str(Path(run, 'scores.log'))),
        my_callbacks.ImageLogger(writer),
        tf_callbacks.TensorBoard(log_dir=run, write_graph=True, profile_batch=0, histogram_freq=1)
    ]

    if experiment:
        cb.append(my_callbacks.CometLogger(experiment))

    return cb


def main():
  
    args = arguments.get_args()

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            for gpu in gpus:
                print(gpu)
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

    from numpy.random import seed
    seed(1)
    from tensorflow.random import set_seed
    set_seed(2)

    tf.keras.backend.set_image_data_format("channels_first")

    meta = pd.read_csv(args["meta"])
        
    def summary():
        m = model.model_map(args["model"])(args)
        m.summary()

    def train():
        experiment = prerun(args)
        m = model.build_model(args)
        callbacks = make_callbacks(args, experiment)

        functions.train.run(args, meta, m, callbacks, experiment)

    def cv():
        new_run_dir = args["skip_n_folds"] == 0
        experiment = prerun(args, run_dir=new_run_dir)
        m = model.build_model(args)

        functions.cv.run(args, meta, m, experiment, skip_n_folds=args["skip_n_folds"])

    def predict():
        functions.predict.run(args, meta)

    def param():
        functions.param.run(args, meta)

    def embed():
        functions.embed.run(args, meta)

    def tb_embed():
        functions.tb_embed.run(args, meta)

    function_map = {
        "train": train,
        "cv": cv,
        "predict": predict,
        "summary": summary,
        "param": param,
        "embed": embed,
        "tb_embed": tb_embed
    }
    
    function_map[args["function"]]()


if __name__ == "__main__":
    main()
