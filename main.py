import arguments
import model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks as tf_callbacks
import callbacks as my_callbacks
import preprocessing
import metrics
from pathlib import Path
import shutil
import pandas as pd
import numpy as np
import pickle
import schedules
import json


def main():
  
    args = arguments.get_args()

    def rundir_check():
        p = Path(args["run_dir"])
        if p.exists():
            raise ValueError("Rundir exists, please remove.")
        p.mkdir()

    meta = pd.read_csv(args["meta"])
        
    model_map = {
        "simple_nn": model.simple_nn,
        "simple_nn_with_dropout": model.simple_nn_with_dropout,
        "simple_cnn_with_dropout": model.simple_cnn_with_dropout,
        "deepflow": model.deepflow
    }

    optimizer_map = {
        "adam": tf.train.AdamOptimizer(),
        "sgd_mom": tf.train.MomentumOptimizer(args["learning_rate"], args["momentum"])
    }

    lr_schedule_map = {
        "step_decay": schedules.get_step_decay(args)
    }
    
    def train(split=args["split_dir"], run=args["run_dir"], id_=100, cv=False):
        if not cv:
            rundir_check()

        channel_string = "".join([str(c) for c in args["channels"]])

        aug = preprocessing.apply_augmentation if args["augmentation"] else None

        train_ds, val_ds, train_len, validation_len = preprocessing.load_datasets(
            Path(split, "train.txt"), Path(split, "val.txt"),
            "caches/train-%d-%s" % (id_, channel_string), "caches/val-%d-%s" % (id_, channel_string),
            meta, args, aug
        )

        tb = tf_callbacks.TensorBoard(log_dir=run, histogram_freq=None, batch_size=args["batch_size"], write_graph=True, write_grads=True, write_images=True)

        cb = [
            tf_callbacks.ModelCheckpoint(str(Path(run, "model.hdf5")), verbose=0, period=1),
            tb,
            my_callbacks.ValidationMonitor(val_ds, validation_len, Path(run, "scores.log"), args, id_)
        ]

        if "schedule" in args:
            cb += lr_schedule_map[args["schedule"]]

        m = model_map[args["model"]](args)
        optimizer = optimizer_map[args["optimizer"]]

        m.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            metrics=[bal_acc]
        )
        
        m.fit(
            train_ds,
            epochs=args["epochs"], 
            steps_per_epoch=int(np.ceil(train_len/args["batch_size"])),
            callbacks=cb
        )


    def cv():
        rundir_check()

        from os import sep

        split_dir = Path(args["split_dir"])
        for i, fold in enumerate(split_dir.iterdir()):
            if fold.is_dir():
                run = Path(args["run_dir"], str(fold).split(sep)[-1])
                Path(run).mkdir()
                train(fold, run, i, cv=True)


    def predict():
        rundir_check()

        ds, ds_len = preprocessing.load_dataset(None, None, meta, args)
        
        m = keras.models.load_model(args["model_hdf5"], compile=False)
        m.compile(
            optimizer=tf.train.AdamOptimizer(),
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            metrics=[bal_acc]
        )
        
        m.evaluate(
            ds,
            batch_size=args["batch_size"],
            steps_per_epoch=int(np.ceil(ds_len/args["batch_size"])),
        )

    def summary():
        m = model_map[args["model"]](args)
        m.summary()

    function_map = {
        "train": train,
        "cv": cv,
        "predict": predict,
        "summary": summary
    }
    
    bal_acc = metrics.BalancedAccuracy(args["noc"]).balanced_accuracy
    function_map[args["function"]]()


if __name__ == "__main__":
    tf.enable_eager_execution()
    main()
