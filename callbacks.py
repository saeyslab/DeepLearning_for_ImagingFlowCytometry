import tensorflow
from tensorflow import keras

class Report(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs):
        print("Test")