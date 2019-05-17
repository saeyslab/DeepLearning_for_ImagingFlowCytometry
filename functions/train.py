import main
import model
import preprocessing
import callbacks as my_callbacks

import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks as tf_callbacks
from tensorflow.keras.backend import set_session

import sys

def prerun(args, meta, data):
    split = args["split_dir"]
    aug = preprocessing.apply_augmentation if args["augmentation"] else None

    with tf.device("/cpu:0"): 
        train_ds, val_ds, train_len, validation_len = preprocessing.load_datasets(
            Path(split, "train.txt"), Path(split, "val.txt"), meta, args, aug, data=data
        )

    return train_ds, val_ds, train_len, validation_len

def run(args, meta, id_=100, exp=None, new_run_dir=True, data=None):
    train_ds, val_ds, train_len, validation_len = prerun(args, meta, data)

    run = args["run_dir"]
    
    if exp is None:
        exp = main.prerun(args, run_dir=new_run_dir, exp=True)
    
    tb = tf_callbacks.TensorBoard(log_dir=run, histogram_freq=None, batch_size=args["batch_size"], write_graph=True, write_grads=True, write_images=True)

    cb = [
        tb,
        tf_callbacks.ModelCheckpoint(str(Path(run, "model.hdf5")), verbose=0, period=1),
        my_callbacks.ValidationMonitor(val_ds, validation_len, Path(run, "scores.log"), args, id_, exp)
    ]

    m = model.build_model(args)
    tb.set_model(m)

    hist = m.fit(
        train_ds,
        epochs=args["epochs"], 
        steps_per_epoch=10,#int(np.ceil(train_len/args["batch_size"])),
        callbacks=cb
    )

    return hist
