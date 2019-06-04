import tensorflow
from tensorflow import keras
import numpy as np
import metrics
import sklearn
from pathlib import Path
from tabulate import tabulate
from tqdm import tqdm
import pickle

class ValidationMonitor(keras.callbacks.Callback):

    def __init__(self, logfile, args):

        self.epsilon = args["es_epsilon"]
        self.wait = 0
        self.patience = args["es_patience"]
        self.logfile = logfile
        self.args = args
        
    def set(self, ds, ds_size, fold, exp):
        self.log = open(self.logfile, mode="at+", buffering=1)
        self.ds = ds
        self.ds_size = ds_size
        self.fold = fold
        self.experiment = exp
        
        self.max_index = None
        self.max_cm = None
        self.max = None
        
        self.epoch = 0
        self.batch = 0
        self.history = {}
        self.runcount = 0
        
        self.log.write("TRAINING FOLD %d\n" % self.fold)


    def log_in_history(self, logs):
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        if self.experiment is not None:
            logs_copy = logs.copy()
            logs_copy["val_confusion_matrix"] = str(logs_copy["val_confusion_matrix"])

            self.experiment.log_metrics(logs_copy)

    def do(self, logs):
        if not hasattr(self, 'ds'):
            raise ValueError("ValidationMonitor not initialized: validation dataset not set.")

        all_labels = np.empty((self.ds_size,), dtype=int)
        all_preds = np.empty((self.ds_size,), dtype=int)

        pos = 0
        for image_batch, label_batch in tqdm(iter(self.ds), total=int(np.ceil(self.ds_size/self.args["batch_size"]))):
            preds = self.model.predict_on_batch(image_batch)
            l = preds.shape[0]
            all_labels[pos:pos+l] = label_batch
            all_preds[pos:pos+l] = np.argmax(preds, axis=1)

            pos += l

        self.log.write("Epoch: %d, batch: %d\n" % (self.epoch, self.batch))
        
        bal_acc = sklearn.metrics.balanced_accuracy_score(all_labels, all_preds)
        cm = sklearn.metrics.confusion_matrix(all_labels, all_preds)

        logs["val_balanced_accuracy"] = bal_acc
        logs["val_confusion_matrix"] = cm
        
        self.wait += 1

        if self.max is None or (bal_acc - self.max) > self.epsilon:
            self.log.write("NEW MAX\n")
            self.max_index = self.runcount
            self.max = bal_acc
            self.wait = 0

            if self.fold is not None:
                self.model.save_weights(str(Path(self.args["run_dir"], "best-model-fold-%d.h5" % self.fold)))
            else:
                self.model.save_weights(str(Path(self.args["run_dir"], "best-model.h5")))

        self.log.write("Bal acc: %.4f\n" % bal_acc)
        self.log.write(tabulate(cm))
        self.log.write("\n")
        
        self.runcount += 1
        return logs

    def on_epoch_end(self, epoch, logs):
        self.epoch = epoch
        if (
            self.args["freq_type"] == "epoch"
            and epoch % self.args["update_freq"] == 0
        ):
            logs = self.do(logs or {})
            self.log_in_history(logs)

        if self.patience is not None:
            if self.patience <= self.wait:
                self.model.stop_training = True

    def on_batch_end(self, batch, logs):
        self.batch = batch
        if (
            self.args["freq_type"] == "batch"
            and batch > 0 
            and batch % self.args["update_freq"] == 0
        ):
            logs = self.do(logs or {})
            self.log_in_history(logs)

    def on_train_end(self, logs):
        if self.args["function"] == "cv":
            p = Path(self.args["run_dir"], "history.pkl")
            if p.exists():
                with open(p, "rb") as handle:
                    hist = pickle.load(handle)
            else:
                hist = {}
            
            hist.setdefault("max_index", []).append(self.max_index)
            for k, v in self.history.items():
                hist.setdefault(k, []).append(v)
            
            with open(p, "wb") as handle:
                pickle.dump(hist, handle)
                
        elif self.args["function"] == "train" or self.args["function"] == "param":
            p = Path(self.args["run_dir"], "history.pkl")
            hist = {}
            hist["max_index"] = self.max_index
            for k, v in self.history.items():
                hist.setdefault(k, []).append(v)

            with open(p, "wb") as handle:
                pickle.dump(hist, handle)
