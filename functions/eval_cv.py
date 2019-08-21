import functions.train
import tensorflow as tf
from joblib import Parallel, delayed
from pathlib import Path
import main
import time
import preprocessing
import numpy as np
import model
import pickle

def run(args, meta):
    main.prerun(args, run_dir=False, exp=False)
    labels = meta["label"].values
    m = model.load_model(args, load_weights=False)
    data = preprocessing.load_hdf5_to_memory(args, labels)

    paths = [p for p in Path(args["split_dir"]).iterdir() if p.is_dir()]
    print(paths)
    results = {"pred": [None]*len(paths), "true": [None]*len(paths)}

    for d in paths:

        fold = int(d.parts[-1])
        print(fold)

        m.load_weights(args["model_hdf5"] % fold)

        val_idx = np.loadtxt(Path(d, 'val.txt'), dtype=int)
        ds, ds_len = preprocessing.load_dataset(data, val_idx, labels, args, type="pred")
    
        preds = m.predict(
            ds,
            steps=int(np.ceil(ds_len/args["batch_size"])),
        )

        results["pred"][int(d.parts[-1])] = preds
        results["true"][int(d.parts[-1])] = labels[val_idx]

    out_dir = Path(Path(args["model_hdf5"]).parts[:-2])
    with open(str(Path(out_dir, "predictions.pkl")), "wb") as pkl:
        pickle.dump(results, pkl)



    
