import math
from tensorflow import keras

def get_step_decay(args):

    def step_decay(epoch):
        initial_lrate = args["learning_rate"]
        drop = args["learning_rate_decay"]
        epochs_drop = args["epochs_per_decay"]
        lrate = initial_lrate * math.pow(drop,  
                math.floor((1+epoch)/epochs_drop))
        return lrate

    return keras.callbacks.LearningRateScheduler(step_decay)