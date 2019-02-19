import tensorflow as tf
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

def load_dataset(indices_file, cache_file, meta, args, train):

    indices = np.loadtxt(indices_file, dtype=int)
    meta = meta.iloc[indices]

    steps = int(np.ceil(len(indices)/args.batch_size))
    image_columns = ["image_%s" % str(c) for c in args.channels]

    all_image_paths = meta[image_columns].applymap(lambda path: str(Path(args.image_base, path)))
    all_image_labels = meta["label"]
    
    if train:
        X = []
        for i in range(args.noc):
            indices = all_image_labels.index[all_image_labels == i]
            X.append(
                tf.data.Dataset.from_tensor_slices((
                    all_image_paths.loc[indices].values, 
                    all_image_labels.loc[indices].values
                ))
                .map(lambda i, l: (load_and_preprocess_images(i, args), l), num_parallel_calls=4)
                .cache(filename=cache_file+"-%d" % i)
                .apply(
                    tf.data.experimental.shuffle_and_repeat(buffer_size=len(indices))
                )
            )
        ds = tf.data.experimental.sample_from_datasets(X)
        ds = ds.batch(args.batch_size, drop_remainder=True).prefetch(buffer_size=args.batch_size*2)
    else:
        ds = tf.data.Dataset.from_tensor_slices((
            all_image_paths.values, 
            all_image_labels.values
        ))
        ds = ds.map(lambda i, l: (load_and_preprocess_images(i, args), l), num_parallel_calls=4).cache(filename=cache_file)
        ds = ds.apply(
            tf.data.experimental.shuffle_and_repeat(buffer_size=all_image_labels.shape[0])
        )
        ds = ds.batch(args.batch_size, drop_remainder=True).prefetch(buffer_size=args.batch_size*2)

    return ds, steps 


def load_datasets(train_indices, val_indices, train_cache, val_cache, meta, args):
    train_ds, train_steps = load_dataset(train_indices, train_cache, meta, args, True)
    val_ds, val_steps = load_dataset(val_indices, val_cache, meta, args, False)

    return train_ds, val_ds, train_steps, val_steps


if __name__ == "__main__":
    tf.enable_eager_execution()

    from collections import Counter
    import arguments
    from pathlib import Path
    import pandas as pd

    args = arguments.get_args()
    meta = pd.read_csv(args.meta)
    train_indices = Path(args.split_dir, "val.txt")
    train_cache = str(Path("caches", "test"))

    ds, _ = load_dataset(train_indices, train_cache, meta, args, False)

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

    duration = end-start
    print("{} batches: {} s".format(batches, duration))
    print("{:0.5f} Images/s".format(args.batch_size*batches/duration))
    print("Total time: {}s".format(end-overall_start))
 