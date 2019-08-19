import tensorflow as tf
from sklearn.metrics import confusion_matrix
import numpy as np
from tensorflow.python.keras import backend as K
import sys


class BalancedAccuracy(tf.keras.metrics.Metric):
    def __init__(self, noc, name="balanced_accuracy", **kwargs):
        super(BalancedAccuracy, self).__init__(name=name, **kwargs)

        self.noc = noc
        self.confusion_matrix = self.add_weight(
            name = "confusion_matrix",
            shape = (noc, noc),
            initializer = "zeros",
            dtype = tf.int32
        )

    def reset_states(self):
        K.batch_set_value([(v, np.zeros(shape=v.get_shape())) for v in self.variables])

    def update_state(self, y_true, y_pred, sample_weight=None):
        confusion_matrix = tf.math.confusion_matrix(y_true, tf.argmax(y_pred, axis=1), num_classes=self.noc)
        return self.confusion_matrix.assign_add(confusion_matrix)

    def result(self):
        diag = tf.linalg.diag_part(self.confusion_matrix)
        rowsums = tf.math.reduce_sum(self.confusion_matrix, axis=1)
        result = tf.math.reduce_mean(diag/rowsums, axis=0)
        return result
