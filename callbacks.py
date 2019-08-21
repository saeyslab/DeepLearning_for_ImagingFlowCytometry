import tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np
import metrics
import sklearn
from pathlib import Path
from tabulate import tabulate
from tqdm import tqdm
import pickle
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.eager import context


class ReduceLROnPlateauWithWarmup(keras.callbacks.ReduceLROnPlateau):
    def __init__(self, monitor, min_delta, factor, patience, base_learning_rate, warmup_length, warmup_coeff):
        super(ReduceLROnPlateauWithWarmup, self).__init__(
            monitor=monitor, 
            min_delta=min_delta,
            factor=factor,
            patience=patience,
            mode="max"
        )

        self.warmup_length = warmup_length
        self.warmup_coeff = warmup_coeff
        self.base_learning_rate = base_learning_rate

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.warmup_length:
            new_lr = self.warmup_coeff*self.base_learning_rate
            keras.backend.set_value(self.model.optimizer.lr, new_lr)
        if epoch == self.warmup_length:
            keras.backend.set_value(self.model.optimizer.lr, self.base_learning_rate)
            super()._reset()
            super().on_epoch_end(epoch, logs)
        else:
            super().on_epoch_end(epoch, logs)


class AdamLRLogger(keras.callbacks.Callback):

    def __init__(self, writer):
        super(AdamLRLogger, self).__init__()
        self.writer = writer

    def on_epoch_end(self, epoch, logs=None):
        opt = self.model.optimizer

        lr = K.get_value(opt.lr)
        iterations = K.get_value(opt.iterations)
        beta_2 = K.get_value(opt.beta_2)
        beta_1 = K.get_value(opt.beta_1)
        
        t = K.cast(iterations, K.floatx()) + 1
        lr_t = lr * (tf.sqrt(1. - tf.pow(beta_2, t)) /
                     (1. - tf.pow(beta_1, t)))

        with context.eager_mode(), self.writer.as_default(), summary_ops_v2.always_record_summaries():
            summary_ops_v2.scalar("adam/lr_t", lr_t, step=epoch)
            summary_ops_v2.scalar("adam/lr", lr, step=epoch)
            summary_ops_v2.scalar("adam/beta1", beta_1, step=epoch)
            summary_ops_v2.scalar("adam/beta2", beta_2, step=epoch)
            summary_ops_v2.scalar("adam/iterations", iterations, step=epoch)
            summary_ops_v2.scalar("adam/t", iterations, step=epoch)


class CometLogger(keras.callbacks.Callback):

    def __init__(self, experiment):
        super(CometLogger, self).__init__()

        self.experiment = experiment

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            self.experiment.log_metrics(logs, step=epoch)


class ImageLogger(keras.callbacks.Callback):

    def __init__(self, writer):
        super(ImageLogger, self).__init__()

        self.writer = writer
    
    def make_summary(self, tensor):
        with context.eager_mode(), self.writer.as_default(), summary_ops_v2.always_record_summaries():
            for i in range(self.n_channels):
                summary_ops_v2.image(
                    "image dim %d" % i, 
                    tensor[:, i, :, :, tf.newaxis],
                    max_images=3,
                )

    def set_dataset(self, dataset, n_channels):
        self.dataset = dataset
        self.n_channels = n_channels

    def on_train_begin(self, logs=None):
        self.make_summary(next(iter(self.dataset))[0])


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
            self.experiment.log_metrics(logs)

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
