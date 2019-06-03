from tensorflow.keras import models
import preprocessing
import numpy as np
import pickle
from pathlib import Path
from tensorflow.keras import callbacks

def run(args, meta):

    tb = callbacks.TensorBoard("tmp", embeddings_freq=1, embeddings_layer_names=["activation_10"])

    data = preprocessing.load_hdf5_to_memory(args, meta["label"])
    idx = np.loadtxt(Path(args["split_dir"], "val.txt"), dtype=int)
    ds, _ = preprocessing.load_dataset(data, idx, meta["label"], args, type="val")

    model = models.load_model(args["model_hdf5"])
    model.predict(
        ds,
        batch_size=128,
        callbacks=[tb]
    )
