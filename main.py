import arguments
import model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks as tf_callbacks
import callbacks as my_callbacks
import preprocessing
import metrics
from pathlib import Path
import shutil
import pandas as pd

def main():
    
    args = arguments.get_args()

    p = Path(args.run_dir)
    if p.exists():
        shutil.rmtree(p)
    p.mkdir()

    meta = pd.read_csv(args.meta)
    
    m = model.simple_nn(args)
    bal_acc = metrics.BalancedAccuracy(args.noc).balanced_accuracy

    m.compile(
        optimizer=tf.train.AdamOptimizer(),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=[bal_acc]
    )

    def train(split=args.split_dir, run=args.run_dir, id_=100):
        train_ds, val_ds, steps_per_epoch, validation_steps = preprocessing.load_datasets(
            Path(split, "train.txt"), Path(split, "val.txt"),
            "caches/train-%d" % id_, "caches/val-%d" % id_,
            meta, args
        )
        
        cb = [
            tf_callbacks.ModelCheckpoint(str(Path(run, "{epoch:02d}-model.hdf5")), verbose=0, period=1),
            tf_callbacks.TensorBoard(log_dir=run, histogram_freq=0, batch_size=args.batch_size, write_graph=True, write_grads=True, write_images=True),
            my_callbacks.ValidationMonitor(val_ds, validation_steps, Path(run, "scores.log"), args)
        ]
        cb[-1].set_fold(id_)
        
        m.fit(
            train_ds,
            epochs=args.epochs, 
            steps_per_epoch=20,#steps_per_epoch,
            callbacks=cb
        )

    def cv():
        from os import sep

        split_dir = Path(args.split_dir)
        for i, fold in enumerate(split_dir.iterdir()):
            if fold.is_dir():
                run = Path(args.run_dir, str(fold).split(sep)[-1])
                Path(run).mkdir()
                # train and validate fold
                train(fold, run, i)
                # flush dataset

    if args.function == "train":
        train()
    elif args.function == "cv":
        cv()


if __name__ == "__main__":
    tf.enable_eager_execution()
    main()
