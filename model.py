import tensorflow as tf
from tensorflow import keras
import metrics
import models.simple
import models.deepflow
import models.resnet

def model_map(key):
    return {
        "simple_nn": models.simple.simple_nn,
        "simple_nn_with_dropout": models.simple.simple_nn_with_dropout,
        "simple_cnn_with_dropout": models.simple.simple_cnn_with_dropout,
        "deepflow": models.deepflow.deepflow,
        "deepflow_narrow": models.deepflow.deepflow_narrow,
        "resnet50": resnet50,
        "resnet18": resnet18
    }[key]


def optimizer_map(key):
    def get_decay_mom(args):
        lr = tf.train.exponential_decay(
            args["learning_rate"],
            tf.train.get_or_create_global_step(),
            args["epochs_per_decay"],
            args["learning_rate_decay"],
            staircase=True
        )

        return tf.train.MomentumOptimizer(lr, args["momentum"])

    def get_mom(args):
        return tf.train.MomentumOptimizer(args["learning_rate"], args["momentum"])

    def get_adam(args):
        return tf.train.AdamOptimizer(learning_rate=args["learning_rate"], beta1=args["beta1"], epsilon=args["epsilon"])

    def get_adam_def(args):
        return tf.train.AdamOptimizer()

    def get_rmsprop(args):
        return tf.train.RMSPropOptimizer(args["learning_rate"], momentum=args["momentum"])

    return {
        "adam": get_adam,
        "adam_def": get_adam_def,
        "mom_decay": get_decay_mom,
        "mom": get_mom,
        "rmsprop": get_rmsprop
    }[key]


def build_model(args):
    m = model_map(args["model"])(args)
    optimizer = optimizer_map(args["optimizer"])(args)

    bal_acc = metrics.BalancedAccuracy(args["noc"]).balanced_accuracy
    m.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=[bal_acc]
    )

    return m

def resnet50(args):
    s = (len(args["channels"]), args["image_width"], args["image_height"])
    model = models.resnet.ResnetBuilder.build_resnet_50(s, args["noc"])

    return model

def resnet18(args):
    s = (len(args["channels"]), args["image_width"], args["image_height"])
    model = models.resnet.ResnetBuilder.build_resnet_18(s, args["noc"])

    return model
