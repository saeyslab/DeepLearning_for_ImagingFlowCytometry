import tensorflow as tf
from tensorflow import keras
import metrics

def model_map(key):
    return {
        "simple_nn": simple_nn,
        "simple_nn_with_dropout": simple_nn_with_dropout,
        "simple_cnn_with_dropout": simple_cnn_with_dropout,
        "deepflow": deepflow
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
        return tf.train.AdamOptimizer(learning_rate=args["learning_rate"])

    def get_rmsprop(args):
        return tf.train.RMSPropOptimizer(args["learning_rate"], momentum=args["momentum"])

    return {
        "adam": get_adam,
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


def simple_nn(args):
    model = keras.Sequential([
        keras.layers.Flatten(
            data_format="channels_first", 
            input_shape=(len(args["channels"]), args["image_width"], args["image_height"])
        ),
        keras.layers.Dense(384, activation=tf.nn.relu),
        keras.layers.Dense(args["noc"], activation="softmax")
    ])

    return model


def simple_nn_with_dropout(args):
    model = keras.Sequential([
        keras.layers.Flatten(
            data_format="channels_first", 
            input_shape=(len(args["channels"]), args["image_width"], args["image_height"])
        ),
        keras.layers.Dropout(args["dropout"]["visible"]),
        keras.layers.Dense(384, activation=tf.nn.relu), #kernel_regularizer=keras.regularizers.l2(l=args["l2"])),
        keras.layers.Dropout(args["dropout"]["hidden"]),
        keras.layers.Dense(args["noc"], activation="softmax")
    ])

    return model

def simple_cnn_with_dropout(args):
    model = keras.Sequential([
        keras.layers.Conv2D(
            filters=16,
            kernel_size=(3,3),
            strides=2,
            padding="valid",
            activation="relu",
            data_format="channels_first",
            input_shape=(len(args["channels"]), args["image_width"], args["image_height"])
        ),
        keras.layers.MaxPooling2D(
            pool_size=(2, 2), 
            strides=None, 
            padding='valid', 
            data_format="channels_first"
        ),
        keras.layers.Flatten(data_format="channels_first"),
        keras.layers.Dense(256, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(l=args["l2"])),
        keras.layers.Dropout(args["dropout"]["hidden"]),
        keras.layers.Dense(args["noc"], activation="softmax")
    ])

    return model

def deepflow(args):

    def _dual_factory(inp, f_out_1, f_out_2):
        conv1 = keras.layers.Conv2D(f_out_1, 1, padding="valid", kernel_regularizer=keras.regularizers.l2(l=args["l2"]))(inp)
        conv1 = keras.layers.BatchNormalization(axis=1, scale=False)(conv1)
        conv1 = keras.layers.ReLU()(conv1)

        conv2 = keras.layers.Conv2D(f_out_2, 3, padding="same", kernel_regularizer=keras.regularizers.l2(l=args["l2"]))(inp)
        conv2 = keras.layers.BatchNormalization(axis=1, scale=False)(conv2)
        conv2 = keras.layers.ReLU()(conv2)

        return keras.layers.Concatenate(axis=3)([conv1, conv2])

    def _dual_downsample_factory(inp, f_out):
        conv1 = keras.layers.Conv2D(f_out, 3, strides=[2, 2], padding="same", kernel_regularizer=keras.regularizers.l2(l=args["l2"]))(inp)
        conv1 = keras.layers.BatchNormalization(axis=1, scale=False)(conv1)
        conv1 = keras.layers.ReLU()(conv1)
        
        pool1 = keras.layers.MaxPooling2D(pool_size=[3, 3], strides=[2, 2], padding="same")(inp)

        return keras.layers.Concatenate(axis=3)([conv1, pool1])

    inp = keras.layers.Input(shape=(len(args["channels"]), args["image_width"], args["image_height"]))

    conv1 = keras.layers.Conv2D(96, 3, strides=[2, 2], padding="same", kernel_regularizer=keras.regularizers.l2(l=args["l2"]))(inp)
    conv1 = keras.layers.BatchNormalization(axis=1, scale=False)(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    in3a = _dual_factory(conv1, 32, 32)
    in3b = _dual_factory(in3a, 32, 48)
    in3c = _dual_downsample_factory(in3b, 80)

    in4a = _dual_factory(in3c, 112, 48)
    in4b = _dual_factory(in4a, 96, 64)
    in4c = _dual_factory(in4b, 80, 80)
    in4d = _dual_factory(in4c, 48, 96)
    in4e = _dual_downsample_factory(in4d, 96)

    in5a = _dual_factory(in4e, 176, 160)
    in5b = _dual_factory(in5a, 176, 160)

    in6a = _dual_downsample_factory(in5b, 96)
    in6b = _dual_factory(in6a, 176, 160)
    in6c = _dual_factory(in6b, 176, 160)

    flatten = keras.layers.Flatten(data_format="channels_first")(in6c)
    fc1 = keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(l=args["l2"]))(flatten)
    soft = keras.layers.Dense(args["noc"], activation="softmax", kernel_regularizer=keras.regularizers.l2(l=args["l2"]))(fc1)

    model = keras.models.Model(inputs=inp, outputs=soft)

    return model
