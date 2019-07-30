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

def make_callbacks_and_model(args, tb=True):
    run = args["run_dir"]

    cb = [
        tf_callbacks.ModelCheckpoint(str(Path(run, "model.hdf5")), verbose=0, save_freq='epoch'),
        my_callbacks.ValidationMonitor(Path(run, "scores.log"), args)
    ]

    m = model.build_model(args)
    
    if tb:
        tb = tf_callbacks.TensorBoard(log_dir=run, histogram_freq=1, profile_batch=3, write_graph=True)
        tb.set_model(m)
        cb.append(tb)

    return cb, m


def main():
  
    args = arguments.get_args()

    gpus = tf.config.experimental.list_logical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)])
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    tf.keras.backend.set_image_data_format("channels_first")

    meta = pd.read_csv(args["meta"])
        
    def summary():
        m = model.model_map(args["model"])(args)
        m.summary()

    def train():
        experiment = prerun(args)
        callbacks, model = make_callbacks_and_model(args)
        functions.train.run(args, meta, model, callbacks, experiment)

    def cv():
        new_run_dir = args["skip_n_folds"] == 0
        experiment = prerun(args, run_dir=new_run_dir)

        callbacks, model = make_callbacks_and_model(args, tb=False)
        functions.cv.run(args, meta, model, callbacks, experiment, skip_n_folds=args["skip_n_folds"])

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
