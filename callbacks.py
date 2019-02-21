import tensorflow
from tensorflow import keras
import numpy as np
import metrics
import sklearn
from pathlib import Path
from tabulate import tabulate
from tqdm import tqdm

class ValidationMonitor(keras.callbacks.Callback):

    def __init__(self, ds, ds_size, logfile, args, writer, fold):
        self.ds = ds
        self.ds_size = ds_size
        self.log = open(logfile, mode="wt", buffering=1)
        self.max = None
        self.max_cm = None
        self.epoch = 0
        self.batch = 0
        self.args = args
        self.fold = fold
        self.metrics = []
        self.writer = writer

        self.log.write("TRAINING FOLD %d\n" % self.fold)

    def do(self):

        all_labels = np.empty((self.ds_size,), dtype=int)
        all_preds = np.empty((self.ds_size,), dtype=int)

        print()
        print()

        pos = 0
        for image_batch, label_batch in tqdm(iter(self.ds), total=np.ceil(self.ds_size/self.args.batch_size)):
            preds = self.model.predict_on_batch(image_batch)
            l = preds.shape[0]
            all_labels[pos:pos+l] = label_batch
            all_preds[pos:pos+l] = np.argmax(preds, axis=1)

            pos += l

        print()
        print()

        self.log.write("Epoch: %d, batch: %d\n" % (self.epoch, self.batch))
        
        bal_acc = sklearn.metrics.balanced_accuracy_score(all_labels, all_preds)

        self.metrics.append(bal_acc)

        cm = sklearn.metrics.confusion_matrix(all_labels, all_preds)
        if self.max is None or bal_acc > self.max:
            self.log.write("NEW MAX\n")
            self.max = bal_acc
            self.max_cm = cm

            if self.fold is not None:
                self.model.save(Path(self.args.run_dir, "best-model-fold-%d.h5" % self.fold))
            else:
                self.model.save(Path(self.args.run_dir, "best-model.h5"))

        self.log.write("Bal acc: %.4f\n" % bal_acc)
        self.log.write(tabulate(cm))
        self.log.write("\n")

    def on_epoch_end(self, epoch, logs):
        self.epoch = epoch
        if (
            self.args.freq_type == "epoch"
            and epoch % self.args.update_freq == 0
        ):
            self.do()

    def on_batch_end(self, batch, logs):
        self.batch = batch
        if (
            self.args.freq_type == "batch"
            and batch > 0 
            and batch % self.args.update_freq == 0
        ):
            self.do()

    def on_train_end(self, logs):
        if self.args.function == "cv":
            p = Path(self.args.run_dir, "fold_scores.npy")
            if p.exists():
                d = np.load(p).item()
                d["max_cm"].append(self.max_cm)
                d["metrics"].append(self.metrics)
            else:
                d = {"max_cm": [self.max_cm], "metrics": [self.metrics]}
            np.save(p, d)
        elif self.args.function == "train":
            p = Path(self.args.run_dir, "max_scores.npy")
            np.save(p, [self.max_cm])
