import functions.train
import tensorflow as tf
from joblib import Parallel, delayed
from pathlib import Path
import main
import time
import preprocessing
import numpy as np
import model

def run(args, meta):
    main.prerun(args, run_dir=False, exp=False)

    with tf.device("/cpu:0"):
        labels = meta["label"].values
        data = preprocessing.load_hdf5_to_memory(args, labels)
        val_idx = np.loadtxt(Path(args["split_dir"], 'val.txt'), dtype=int)
        ds, ds_len = preprocessing.load_dataset(data, val_idx, labels, args, type="val")

    m = model.load_model(args)
    
    preds = m.evaluate(
        ds,
        batch_size=args["batch_size"],
        steps=int(np.ceil(ds_len/args["batch_size"])),
    )

    np.save("predictions.npy", preds)
