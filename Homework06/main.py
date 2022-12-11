import get_data
from model import MyCNN
import training_loop
import tensorflow_datasets as tfds
import keras_cv

import tensorflow as tf

# HYPERPARAMETERS global
config = "Original" # Path parameter for the csv,logs and Plots folder
BATCH = 64 #64 Default
LEARNING_RATE = 0.001
optimizer = tf.optimizers.Adam(learning_rate = LEARNING_RATE)
EPOCHS = 15
MODE = None # "dense" or "res"
NORMALIZATION = False # boolean
DROPOUT = None # int between 0 and 1
REG = tf.keras.regularizers.L2(0.001) # regularizer l2 int between 0 and 1


aug_model = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
    ])

# get and prepare data and model
(training_data, test_data ) , ds_info = get_data.load_data(False) # True
#training_data = training_data.take(2000) # 
training_data = get_data.data_preprocess(training_data, batch_size = BATCH,  augment = None) # augment should be used on train data only
test_data = get_data.data_preprocess(test_data)

model = MyCNN(optimizer,10,mode=MODE,normalization=NORMALIZATION,dropout_rate = DROPOUT,regularizer=REG)

training_loop.training_cifar(training_data, test_data, model, epochs = EPOCHS, plot=True, save = True, config_name = config) # True