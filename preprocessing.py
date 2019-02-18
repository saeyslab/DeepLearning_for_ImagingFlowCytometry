import tensorflow as tf
import pandas as pd
import time
from pathlib import Path
import numpy as np


def preprocess_image(image, args):
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.resize_images(image, [args.image_width, args.image_height])
    image = tf.squeeze(image, axis=[2])
    image /= 2**16

    return image

def load_and_preprocess_image(path, args):
    image = tf.read_file(path)
    return preprocess_image(image, args)


def load_and_preprocess_images(paths, args):
    return tf.map_fn(
        lambda j: load_and_preprocess_image(j, args),
        paths,
        dtype=tf.float32
    )

def load_dataset(args):
    meta = pd.read_csv(args.meta)
    train_indices = np.loadtxt(Path(args.split_dir, "train.txt"), dtype=int)
    val_indices = np.loadtxt(Path(args.split_dir, "val.txt"), dtype=int)
    steps_per_epoch = int(np.ceil(len(train_indices)/args.batch_size))
    validation_steps = int(np.ceil(len(val_indices)/args.batch_size))

    image_columns = ["image_%s" % str(c) for c in args.channels]
    all_image_paths = meta[image_columns].applymap(lambda path: str(Path(args.image_base, path)))
    all_image_labels = meta["label"]

    def batch_prefetch(ds):
        return ds.batch(args.batch_size).prefetch(buffer_size=args.batch_size*2)
    
    def select_to_ds(indices):
        return tf.data.Dataset.from_tensor_slices((
            all_image_paths.iloc[indices].values, 
            all_image_labels.iloc[indices].values
        ))

    def load_cache(ds, f):
        print(ds)
        return ds.map(lambda i, l: (load_and_preprocess_images(i, args), l), num_parallel_calls=4).cache(filename=f)

    X_train = []
    for i in range(args.noc):
        indices = np.where(all_image_labels[train_indices] == i)[0]
        X_train.append(select_to_ds(indices).apply(
            tf.data.experimental.shuffle_and_repeat(buffer_size=len(indices))
        ))

    train_ds = tf.data.experimental.sample_from_datasets(X_train)
    train_ds = load_cache(train_ds, "./caches/tf-train-2.cache")

    val_ds = select_to_ds(val_indices)
    val_ds = load_cache(val_ds, "./caches/tf-val.cache").repeat()

    return (
        batch_prefetch(train_ds), 
        batch_prefetch(val_ds),
        steps_per_epoch,
        validation_steps
    )


if __name__ == "__main__":
    tf.enable_eager_execution()

    from collections import Counter
    import arguments
    args = arguments.get_args()

    ds, _, _, _ = load_dataset(args)
    meta = pd.read_csv(args.meta)

    overall_start = time.time()
    # Fetch a single batch to prime the pipeline (fill the shuffle buffer),
    # before starting the timer
    batches = 2*tf.ceil(meta.shape[0]/args.batch_size).numpy()+1
    it = iter(ds.take(batches+1))
    next(it)

    start = time.time()
    for i,(images,labels) in enumerate(it):
        if (i%10 == 0):
            print('.',end='')
            print(Counter(labels.numpy()))
        print()
        end = time.time()
        break

    duration = end-start
    print("{} batches: {} s".format(batches, duration))
    print("{:0.5f} Images/s".format(args.batch_size*batches/duration))
    print("Total time: {}s".format(end-overall_start))
 