import tensorflow as tf
import time
from pathlib import Path
import numpy as np
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt


def apply_augmentation(image):

    image = tf.transpose(image, [1, 2, 0])

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(image)
    # Randomly flip the image vertically.
    distorted_image = tf.image.random_flip_up_down(distorted_image)

    distorted_image = tf.transpose(distorted_image, [2, 0, 1])
    return distorted_image


def preprocess_image(image):
    image = tf.read_file(image)
    image = tf.image.decode_png(image, channels=1, dtype=tf.uint16)
    image = tf.image.resize_images(image, (90, 90))
    image = tf.squeeze(image, axis=[2])
    image = tf.cast(image, tf.float32)
    image /= 2**16

    return image


def load_and_preprocess_images(paths, aug):
    tmp = tf.map_fn(
        preprocess_image,
        paths,
        dtype=tf.float32
    )
    return aug(tmp)

def load_and_preprocess_batch(batch, aug):
    return tf.map_fn(
        lambda b: load_and_preprocess_images(b, aug),
        batch,
        dtype=tf.float32
    )

def load_dataset(indices_file, cache_file, meta, type="train", augment_func = None):

    if indices_file is not None:
        indices = np.loadtxt(indices_file, dtype=int)
        meta = meta.iloc[indices]

    image_columns = ["image_%s" % str(c) for c in [1]]

    all_image_paths = meta[image_columns].applymap(lambda path: str(Path("/home/maximl/DATA/Experiment_data/eos_data", path)))
    all_image_labels = meta["label"]
    
    X = []
    for i in range(3):
        indices = all_image_labels.index[all_image_labels == i]
        X.append(
            tf.data.Dataset.from_tensor_slices((
                all_image_paths.loc[indices].values, 
                all_image_labels.loc[indices].values
            )).apply(
                tf.data.experimental.shuffle_and_repeat(len(indices), 20)
            )
        )
    
    ds = tf.data.experimental.sample_from_datasets(X).batch(batch_size=128).prefetch(16).map(lambda i, l: (load_and_preprocess_batch(i, apply_augmentation), l), num_parallel_calls=4)

    return ds


if __name__ == "__main__":
    from collections import Counter
    tf.enable_eager_execution()

    meta = pd.read_csv("/home/maximl/DATA/Experiment_data/eos_data/train_data.csv")
    train_indices = Path("/home/maximl/DATA/Experiment_data/eos_meta/s23_5folds/0", "val.txt")
    train_cache = str(Path("caches", "test"))

    times = []
    for i in range(10):
        ds = load_dataset(train_indices, train_cache, meta, "train", augment_func=apply_augmentation)

        batches = 10
        it = iter(ds.take(batches))
        next(it)

        start = time.time()
        for images, labels in it:
            continue
        times.append(time.time()-start)

    print(np.mean(times))
