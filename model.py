import tensorflow as tf
from tensorflow import keras
import metrics
import models.simple
import models.deepflow
import models.resnet
import models.densenet
import models.testmodel

def model_map(key):
    return {
        "simple_nn": models.simple.simple_nn,
        "simple_nn_with_dropout": models.simple.simple_nn_with_dropout,
        "simple_cnn_with_dropout": models.simple.simple_cnn_with_dropout,
        "deepflow": models.deepflow.deepflow,
        "deepflow_narrow": models.deepflow.deepflow_narrow,
        "resnet50": resnet50,
        "resnet18": resnet18,
        "densenet": densenet,
        "testmodel": testmodel
    }[key]


def optimizer_map(key):
    # def get_decay_mom(args):
    #     lr = tf.train.exponential_decay(
    #         args["learning_rate"],
    #         tf.train.get_or_create_global_step(),
    #         args["epochs_per_decay"],
    #         args["learning_rate_decay"],
    #         staircase=True
    #     )

    #     return tf.keras.optimizers.Momentum(lr, args["momentum"])

    # def get_mom(args):
    #     return tf.keras.optimizers.Momentum(args["learning_rate"], args["momentum"])

    def get_adam(args):
        return tf.keras.optimizers.Adam(lr=args["learning_rate"], beta_1=args["beta1"], epsilon=args["epsilon"])

    def get_adam_def(args):
        return tf.keras.optimizers.Adam()

    def get_rmsprop(args):
        return tf.keras.optimizers.RMSProp(args["learning_rate"], momentum=args["momentum"])

    return {
        "adam": get_adam,
        "adam_def": get_adam_def,
        # "mom_decay": get_decay_mom,
        # "mom": get_mom,
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

def densenet(args):
    s = [len(args["channels"]), args["image_width"], args["image_height"]]
    return models.densenet.DenseNet(
        input_shape=tuple(s),
        num_of_blocks=args["dense_blocks"],
        num_layers_in_each_block=args["dense_layers"],
        output_classes=args["noc"],
        compression=args["compression"],
        dropout_rate=args["dropout"],
        growth_rate=args["growth_rate"],
        weight_decay=args["l2"],
        pool_initial=False,
        include_top=True,
        bottleneck=False,
        depth_of_model=None,
        data_format=tf.keras.backend.image_data_format()
    )

def testmodel(args):
    m = models.testmodel.TestModel(args)

    return m
