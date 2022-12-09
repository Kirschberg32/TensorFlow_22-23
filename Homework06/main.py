import get_data
from model import MyCNN
import training_loop
import tensorflow_datasets as tfds

import tensorflow as tf

# HYPERPARAMETERS global
LEARNING_RATE = 0.001
optimizer = tf.optimizers.Adam(learning_rate = LEARNING_RATE)
EPOCHS = 15
MODE = None # "dense" or "res"
NORMALIZATION = False # boolean
DROPOUT = 0 # int between 0 and 1
REG = 0 # regularizer l2 int between 0 and 1

# get and prepare data and model
(training_data, test_data ) , ds_info = get_data.load_data(False) # True
#training_data = training_data.take(2000) # 
training_data = training_data.apply(get_data.data_preprocess)
test_data = test_data.apply(get_data.data_preprocess)

model = MyCNN(optimizer,10,mode=MODE,normalization=NORMALIZATION,dropout = DROPOUT,regularizer=REG)

training_loop.training_cifar(training_data, test_data, model, epochs = EPOCHS, plot=True, save = True) # True