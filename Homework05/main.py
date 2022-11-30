import get_data
from model import MyCNN
import training_loop

import tensorflow as tf

# HYPERPARAMETERS global
LEARNING_RATE = 0.001
optimizer = tf.optimizers.Adam(learning_rate = LEARNING_RATE)
EPOCHS = 15

# get and prepare data and model
(training_data, test_data ) , _ = get_data.load_data(False)
training_data = training_data.apply(get_data.data_preprocess)
test_data = test_data.apply(get_data.data_preprocess)

model = MyCNN(optimizer,10)

training_loop.training_cifar(training_data, test_data, model, epochs = EPOCHS)