import preprocessing
import numpy as np
import pickle
from pathlib import Path
import model

def run(args, meta):

    def generator(embedder, ds, ds_len):
        for i, (im_batch, _) in enumerate(iter(ds)):
            if i % 100==0:
                print("Batch %d" % i)

            embedded_batch = embedder.predict_on_batch(im_batch)

            for embedding in embedded_batch:
                yield embedding
                
    model = model.load_model(args["model_hdf5"])
    outputs = model.get_layer(args["layer"]).output
    embedder = models.Model(inputs=model.inputs, outputs=outputs)

    data = preprocessing.load_hdf5_to_memory(args, meta["label"])

    idx = np.loadtxt(Path(args["split_dir"], "val.txt"), dtype=int)
    ds, ds_len = preprocessing.load_dataset(data, idx, meta["label"], args, type="val")

    out_dir = Path(args["model_hdf5"]).parents[1]
    with open(str(Path(out_dir, "embedding.pkl")), "wb") as pkl:
        for embedding in generator(embedder, ds, ds_len):
            pickle.dump(embedding, pkl)
