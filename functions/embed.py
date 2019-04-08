from tensorflow.keras import models
import preprocessing
import numpy as np
import pickle

def run(args, meta):

    def generator(embedder, ds, ds_len):
        for i, (im_batch, _) in enumerate(iter(ds)):
            if i % 100==0:
                print("Batch %d" % i)

            embedded_batch = embedder.predict_on_batch(im_batch)

            for embedding in embedded_batch:
                yield embedding
                

    model = models.load_model(args["model_hdf5"])
    outputs = model.get_layer("flatten_2").output
    embedder = models.Model(inputs=model.inputs, outputs=outputs)

    data = preprocessing.load_hdf5_to_memory(args, meta["label"])
    ds, ds_len = preprocessing.load_dataset(data, np.arange(meta.shape[0]), meta["label"], args, type="val")

    with open(args["embedding_output"], "wb") as pkl:
        for embedding in generator(embedder, ds, ds_len):
            pickle.dump(embedding, pkl)
