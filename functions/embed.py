from tensorflow.keras import models
import preprocessing
import numpy as np
import pickle
from pathlib import Path
import model

def run(args):

    def generator(embedder, ds, ds_len):
        for i, im_batch in enumerate(iter(ds)):
            if i % 100==0:
                print("Batch %d" % i)

            embedded_batch = embedder.predict_on_batch(im_batch)

            for embedding in embedded_batch:
                yield embedding
                
    m = model.load_model(args)
    outputs = m.get_layer(args["layer"]).output
    embedder = models.Model(inputs=m.inputs, outputs=outputs)

    data = preprocessing.load_hdf5_to_memory(args, None)

    ds, ds_len = preprocessing.load_dataset(data, None, None, args, type="pred")

    out_dir = Path(args["model_hdf5"]).parent
    with open(str(Path(out_dir, "embedding.pkl")), "wb") as pkl:
        for embedding in generator(embedder, ds, ds_len):
            pickle.dump(embedding, pkl)
