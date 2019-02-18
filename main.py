import arguments
import model
import tensorflow as tf
import preprocessing
import metrics
from pathlib import Path

def main():
    
    args = arguments.get_args()
    
    m = model.simple_nn(args)
    bal_acc = metrics.BalancedAccuracy(args.noc).balanced_accuracy

    m.compile(
        optimizer=tf.train.AdamOptimizer(),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=[bal_acc]
    )

    def train():
        train_ds, val_ds, steps_per_epoch, validation_steps = preprocessing.load_dataset(args)
        m.fit(
            train_ds,
            epochs=1, 
            steps_per_epoch=steps_per_epoch,
            validation_data=val_ds,
            validation_steps=validation_steps
        )

    if args.function == "train":
        train()
    elif args.function == "cv":
        split_dir = Path(args.split_dir)
        for fold in split_dir.iterdir():
            if fold.is_dir():
                args.split_dir = fold
                train()


if __name__ == "__main__":
    tf.enable_eager_execution()
    main()
