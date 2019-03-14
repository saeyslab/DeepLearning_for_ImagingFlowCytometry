import tensorflow as tf
import time
from pathlib import Path
import numpy as np
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import h5py

class dataset_wrapper:
    def __init__(self, h5fp, labels, channels):
        self.labels = labels
        channels = ["channel_%d" % chan for chan in channels]

        shape = tuple([len(channels)] + list(h5fp["channel_1/images"].shape))
        self.images = np.empty(shape=shape, dtype=np.uint16)
        for i, chan in enumerate(channels):
            images = h5fp[chan]["images"]
            masks = h5fp[chan]["masks"]

            self.images[i] = np.multiply(images, masks, dtype=np.float32)/2**16

class generator:
    def __init__(self, data, indices):
        self.data = data
        self.indices = indices

    def __call__(self):
        np.random.shuffle(self.indices) # shuffle happens in-place
        for idx in self.indices:
            yield self.data.images[:, idx, :, :], self.data.labels[idx]

def apply_augmentation(image):

    image = tf.transpose(image, [1, 2, 0])

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(image)
    # Randomly flip the image vertically.
    distorted_image = tf.image.random_flip_up_down(distorted_image)

    distorted_image = tf.transpose(distorted_image, [2, 0, 1])
    return distorted_image

def preprocess_batch(batch, aug):
    return tf.map_fn(
        aug, batch,
        dtype=tf.float32
    )

def load_dataset(data, indices, labels, cache_file, type="train", augment_func = None):

    X = []
    for i in range(8):
        idx = np.where(labels == i)[0]
        X.append(list(idx))

    ds = tf.data.experimental.sample_from_datasets([
        tf.data.Dataset.from_generator(generator(data, x), output_types=(tf.float32, tf.uint8)).repeat()
        for x in X
    ])
    
    ds = ds.batch(batch_size=128)
    ds = ds.map(lambda images, labels: (preprocess_batch(images, apply_augmentation), labels), num_parallel_calls=4)

    return ds


if __name__ == "__main__":
    from collections import Counter
    tf.enable_eager_execution()

    h5 = "/home/maximl/DATA/Experiment_data/PBC/s123.h5"
    meta = pd.read_csv("/home/maximl/DATA/Experiment_data/PBC/train_data_no_images.csv")
    train_indices = np.loadtxt(Path("/home/maximl/DATA/Experiment_data/PBC/s123_5fold/0", "val.txt"), dtype=int)
    train_cache = str(Path("caches", "test"))

    labels = meta["label"].values

    with h5py.File(h5) as h5fp:    
        data = dataset_wrapper(h5fp, labels, [1, 6, 9])
    
        ds = load_dataset(data, train_indices, labels, train_cache, "val", augment_func=apply_augmentation)

        batches = 11
    
        # it = iter(ds.take(batches))
        # next(it)
        # run = []
        # fig, axes = plt.subplots(16, 8, figsize=(50,25))
        # axes = axes.ravel()
    
        # images, labels = next(it)
    
        # for im, ax in zip(images, axes):
        #     print(im)
        #     ax.imshow(im[0])
        # plt.savefig("tmp.png")


        times = []
        for i in range(1):
            it = iter(ds.take(batches))
            next(it)
            run = []
            start = time.time()
            for i, (images, labels) in enumerate(it):
                print(Counter(labels.numpy()))
                run.append(time.time()-start)
                start = time.time()
            times.append(run)

        print(np.mean(times))

