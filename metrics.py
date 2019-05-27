import tensorflow as tf
from sklearn.metrics import confusion_matrix
import numpy as np

class BalancedAccuracy(object):
    def __init__(self, noc):
        self.noc = noc

    def balanced_accuracy(self, y_true, y_pred):
        confusion_matrix = tf.math.confusion_matrix(y_true, tf.argmax(y_pred, axis=1), num_classes=self.noc)
        diag = tf.linalg.diag_part(confusion_matrix)
        rowsums = tf.math.reduce_sum(confusion_matrix, axis=1)
        return tf.math.reduce_mean(diag/rowsums, axis=0)