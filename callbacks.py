import tensorflow
from tensorflow import keras
import numpy as np
import metrics
import sklearn
from pathlib import Path
from tabulate import tabulate
from tqdm import tqdm

class ValidationMonitor(keras.callbacks.Callback):

    def __init__(self, ds, steps, logfile, args):
        self.ds = ds
        self.steps = steps
        self.log = open(logfile, mode="wt", buffering=1)
        self.max = None
        self.max_cm = None
        self.epoch = 0
        self.batch = 0
        self.args = args
        self.fold = None

    def set_fold(self, fold):
        self.fold = fold
        self.log.write("TRAINING FOLD %d\n" % self.fold)

    def do(self):
        bs = self.args.batch_size

        all_labels = np.empty((self.steps*bs,), dtype=int)
        all_preds = np.empty((self.steps*bs,), dtype=int)

        print()
        for i, (image_batch, label_batch) in tqdm(enumerate(iter(self.ds)), total=self.steps-1):
            preds = self.model.predict_on_batch(image_batch)
            all_labels[i*bs:(i+1)*bs] = label_batch
            all_preds[i*bs:(i+1)*bs] = np.max(preds, axis=1)
            if self.steps == i+1:
                break
        print()

        self.log.write("Epoch: %d, batch: %d\n" % (self.epoch, self.batch))
        
        bal_acc = sklearn.metrics.balanced_accuracy_score(all_labels, all_preds)
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
            p = Path(self.args.run_dir, "fold-%d_scores.npy" % self.fold)
            if p.exists():
                arr = np.load(p)
                arr.append(self.max_cm)
            else:
                arr = [self.max_cm]
            np.save(p, arr)
        elif self.args.function == "train":
            p = Path(self.args.run_dir, "max_scores.npy")
            np.save(p, [self.max_cm])
