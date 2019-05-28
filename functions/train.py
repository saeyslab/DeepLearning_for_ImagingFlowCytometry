import main
import model
import preprocessing
import callbacks as my_callbacks

import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

import sys

def prerun(args, meta, data):
    split = args["split_dir"]
    aug = preprocessing.apply_augmentation if args["augmentation"] else None

    with tf.device("/cpu:0"): 
        train_ds, val_ds, train_len, validation_len = preprocessing.load_datasets(
            Path(split, "train.txt"), Path(split, "val.txt"), meta, args, aug, data=data
        )

    return train_ds, val_ds, train_len, validation_len

def run(args, meta, model, callbacks, exp, id_=100, data=None):
    train_ds, val_ds, train_len, validation_len = prerun(args, meta, data)
    
    for cb in callbacks:
        if type(cb)==my_callbacks.ValidationMonitor:
            cb.set(val_ds, validation_len, id_, exp)

    hist = model.fit(
        train_ds,
        epochs=args["epochs"], 
        steps_per_epoch=int(np.ceil(train_len/args["batch_size"])),
        callbacks=callbacks
    )

    return hist
