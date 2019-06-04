from tensorflow.keras import layers
import tensorflow

class TestModel(tensorflow.keras.Model):

  def __init__(self, args, **kwargs):
    super(TestModel, self).__init__(**kwargs)

    input_shape = (len(args["channels"]), args["image_width"], args["image_height"])

    self.conv = layers.Conv2D(16, 3, padding="same", input_shape=input_shape)
    self.flatten = layers.Flatten()
    self.dense_1 = layers.Dense(64, activation='relu')
    self.dense_2 = layers.Dense(args["noc"])

  @tensorflow.function
  def call(self, inputs):
    x = self.conv(inputs)
    x = self.flatten(x)
    x = self.dense_1(x)
    return self.dense_2(x)
        