import tensorflow as tf
import time
from pathlib import Path
import numpy as np
import h5py
import tensorflow_addons as tfa

class dataset_wrapper:
    def __init__(self, h5fp, labels, channels):
        self.labels = labels
        channels = ["channel_%d" % chan for chan in channels]

        shape = tuple([len(channels)] + list(h5fp["channel_1/images"].shape))
        self.images = np.empty(shape=shape, dtype=np.float32)
        for i, chan in enumerate(channels):
            ims = h5fp[chan]["images"]
            masks = h5fp[chan]["masks"]

            self.images[i] = np.multiply(ims, masks, dtype=np.float32)

            # per image normalization
            min_ = self.images[i].reshape(self.images[i].shape[0], -1).min(axis=1)
            max_ = self.images[i].reshape(self.images[i].shape[0], -1).max(axis=1)

            max_ = np.where(max_== 0.0, np.ones_like(max_), max_)

            self.images[i] = ((self.images[i].T-min_)/(min_+max_)).T


class generator:
    def __init__(self, data, indices, shuffle=True):
        self.data = data
        self.indices = indices
        self.shuffle = shuffle
    
    def __call__(self):
        if self.shuffle:
            np.random.shuffle(self.indices) # shuffle happens in-place
        for idx in self.indices:
            yield self.data.images[:, idx, :, :], self.data.labels[idx]


class pred_generator:
    def __init__(self, data, indices=None):
        self.data = data

        if indices is not None:
            self.it = indices
        else:
            self.it = range(self.data.images.shape[1])

    def __len__(self):
        try:
            return len(self.it)
        except:
            return self.data.images.shape[1]

    def __call__(self):
        for idx in self.it:
            yield self.data.images[:, idx, :, :]        


def preprocess_batch(batch, aug):
    return tf.map_fn(
        aug, batch,
        dtype=tf.float32
    )

def load_dataset(data, indices, labels, args, type="train", augment_func = None):
    
    if type=="train":
        X = []
        for i in range(args["noc"]):
            idx = np.where(labels == i)[0]
            idx = list(set(idx) & set(indices))
            X.append(
                tf.data.Dataset.from_generator(
                    generator(data, idx), output_types=(tf.float32, tf.uint8)
                ).repeat()
            )

        ds = tf.data.experimental.sample_from_datasets(X)
        ds = ds.batch(batch_size=args["batch_size"])

        if augment_func is not None:
            ds = ds.map(lambda images, labels: (preprocess_batch(images, augment_func), labels), num_parallel_calls=4)
        ds = ds.prefetch(16)

        ds_length = len(indices)
    elif (type=="val") or (type=="pred"):
        if type =="val":
            X = generator(data, indices, shuffle=False)
            ds = tf.data.Dataset.from_generator(
                X, output_types=(tf.float32, tf.uint8)
            )
            ds_length = len(indices)
        else:
            X = pred_generator(data, indices)
            ds = tf.data.Dataset.from_generator(
                X, output_types=tf.float32
            )
            ds_length = len(X)

        ds = ds.batch(batch_size=args["batch_size"])
        ds = ds.prefetch(16)
    else:
        raise RuntimeError("Wrong argument value (%s)" % type)

    return ds, ds_length

def load_hdf5_to_memory(args, labels):
    with h5py.File(args["h5_data"], mode="r", libver="latest", swmr=True) as h5fp:    
        return dataset_wrapper(h5fp, labels, args["channels"])


def load_datasets(train_indices, val_indices, meta, args, augment_func, data=None):

    labels = meta["label"].values
    if data is None:
        data = load_hdf5_to_memory(args, labels)

    train_indices = np.loadtxt(train_indices, dtype=int)
    val_indices = np.loadtxt(val_indices, dtype=int)
    
    train_ds, train_len = load_dataset(data, train_indices, labels, args, "train", augment_func)
    val_ds, val_len = load_dataset(data, val_indices, labels, args, "val")

    return train_ds, val_ds, train_len, val_len


def apply_augmentation(image):
    image = tf.transpose(image, [1, 2, 0])

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(image)
    # Randomly flip the image vertically.
    distorted_image = tf.image.random_flip_up_down(distorted_image)
    # Randomly rotate the image
    angle = tf.random.uniform([], -3.14, 3.14, tf.float32, None, "angle")
    distorted_image = tfa.image.transform_ops.rotate(distorted_image, angle)

    dx = tf.random.uniform([], -6, 6, tf.float32)
    dy = tf.random.uniform([], -6, 6, tf.float32)
    distorted_image = tfa.image.transform_ops.transform(distorted_image, [1, 0, dx, 0, 1, dy, 0, 0])

    distorted_image = tf.transpose(distorted_image, [2, 0, 1])
    return distorted_image


if __name__ == "__main__":
    from collections import Counter
    import arguments
    from pathlib import Path
    import pandas as pd
    # from matplotlib import pyplot as plt

    args = arguments.get_args()
    meta = pd.read_csv(args["meta"])
    
    labels = meta["label"].values

    with h5py.File(args["h5_data"], "r") as h5fp:    
        data = dataset_wrapper(h5fp, labels, [1])

    ds, _= load_dataset(data, np.arange(500), labels, args, "train", augment_func=apply_augmentation)

    it = iter(ds)

    for i in range(5):
        images, labels = next(it)
        print(Counter(labels.numpy()))

    # print(images.shape)
    # plt.imshow(images[0][0])
    # plt.savefig("test.png")

    it = None

