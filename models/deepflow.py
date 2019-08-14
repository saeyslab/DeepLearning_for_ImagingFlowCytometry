from tensorflow import keras
import tensorflow as tf

def deepflow(args):

    initializer = "he_uniform"

    def _dual_factory(inp, f_out_1, f_out_2):
        with tf.name_scope("Dual") as scope:

            with tf.name_scope("Left") as scope:
                conv1 = keras.layers.Conv2D(
                    f_out_1, 1, padding="valid", kernel_regularizer=keras.regularizers.l2(l=args["l2"]),
                    kernel_initializer=initializer, bias_initializer=initializer)(inp)
                conv1 = keras.layers.BatchNormalization(axis=1, scale=False)(conv1)
                conv1 = keras.layers.ReLU()(conv1)

            with tf.name_scope("Right") as scope:
                conv2 = keras.layers.Conv2D(
                    f_out_2, 3, padding="same", kernel_regularizer=keras.regularizers.l2(l=args["l2"]),
                    kernel_initializer=initializer, bias_initializer=initializer)(inp)
                conv2 = keras.layers.BatchNormalization(axis=1, scale=False)(conv2)
                conv2 = keras.layers.ReLU()(conv2)

            return keras.layers.Concatenate(axis=1)([conv1, conv2])

    def _dual_downsample_factory(inp, f_out):
        with tf.name_scope("Dual downsample") as scope:
            conv1 = keras.layers.Conv2D(
                f_out, 3, strides=[2, 2], padding="same", kernel_regularizer=keras.regularizers.l2(l=args["l2"]),
                kernel_initializer=initializer, bias_initializer=initializer)(inp)
            conv1 = keras.layers.BatchNormalization(axis=1, scale=False)(conv1)
            conv1 = keras.layers.ReLU()(conv1)
            
            pool1 = keras.layers.MaxPooling2D(pool_size=[3, 3], strides=[2, 2], padding="same")(inp)

            return keras.layers.Concatenate(axis=1)([conv1, pool1])

    inp = keras.layers.Input(shape=(len(args["channels"]), args["image_width"], args["image_height"]))

    with tf.name_scope("Top") as scope:
        conv1 = keras.layers.Conv2D(
            96, 3, strides=[2, 2], padding="same", kernel_regularizer=keras.regularizers.l2(l=args["l2"]),
            kernel_initializer=initializer, bias_initializer=initializer)(inp)
        conv1 = keras.layers.BatchNormalization(axis=1, scale=False)(conv1)
        conv1 = keras.layers.ReLU()(conv1)

    with tf.name_scope("Module 1") as scope:
        in3a = _dual_factory(conv1, 32, 32)
        in3b = _dual_factory(in3a, 32, 48)
        in3c = _dual_downsample_factory(in3b, 80)

    with tf.name_scope("Module 2") as scope:
        in4a = _dual_factory(in3c, 112, 48)
        in4b = _dual_factory(in4a, 96, 64)
        in4c = _dual_factory(in4b, 80, 80)
        in4d = _dual_factory(in4c, 48, 96)
        in4e = _dual_downsample_factory(in4d, 96)

    with tf.name_scope("Module 3") as scope:
        in5a = _dual_factory(in4e, 176, 160)
        in5b = _dual_factory(in5a, 176, 160)

    with tf.name_scope("Module 4") as scope:
        in6a = _dual_downsample_factory(in5b, 96)
        in6b = _dual_factory(in6a, 176, 160)
        in6c = _dual_factory(in6b, 176, 160)

    with tf.name_scope("Classifier") as scope:
        flatten = keras.layers.Flatten(data_format="channels_first")(in6c)
        fc1 = keras.layers.Dense(
            256, activation="relu", kernel_regularizer=keras.regularizers.l2(l=args["l2"]),
            kernel_initializer=initializer, bias_initializer=initializer)(flatten)
        soft = keras.layers.Dense(
            args["noc"], activation="softmax", kernel_regularizer=keras.regularizers.l2(l=args["l2"]),
            kernel_initializer=initializer, bias_initializer=initializer)(fc1)

    model = keras.models.Model(inputs=inp, outputs=soft)

    return model


def deepflow_narrow(args):

    def _dual_factory(inp, f_out_1, f_out_2):
        f_out_1 = int(f_out_1/2)
        f_out_2 = int(f_out_2/2)

        conv1 = keras.layers.Conv2D(f_out_1, 1, padding="valid", kernel_regularizer=keras.regularizers.l2(l=args["l2"]))(inp)
        conv1 = keras.layers.BatchNormalization(axis=1, scale=False)(conv1)
        conv1 = keras.layers.ReLU()(conv1)

        conv2 = keras.layers.Conv2D(f_out_2, 3, padding="same", kernel_regularizer=keras.regularizers.l2(l=args["l2"]))(inp)
        conv2 = keras.layers.BatchNormalization(axis=1, scale=False)(conv2)
        conv2 = keras.layers.ReLU()(conv2)

        return keras.layers.Concatenate(axis=1)([conv1, conv2])

    def _dual_downsample_factory(inp, f_out):
        f_out = int(f_out/2)

        conv1 = keras.layers.Conv2D(f_out, 3, strides=[2, 2], padding="same", kernel_regularizer=keras.regularizers.l2(l=args["l2"]))(inp)
        conv1 = keras.layers.BatchNormalization(axis=1, scale=False)(conv1)
        conv1 = keras.layers.ReLU()(conv1)
        
        pool1 = keras.layers.MaxPooling2D(pool_size=[3, 3], strides=[2, 2], padding="same")(inp)

        return keras.layers.Concatenate(axis=1)([conv1, pool1])

    inp = keras.layers.Input(shape=(len(args["channels"]), args["image_width"], args["image_height"]))

    conv1 = keras.layers.Conv2D(48, 3, strides=[2, 2], padding="same", kernel_regularizer=keras.regularizers.l2(l=args["l2"]))(inp)
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