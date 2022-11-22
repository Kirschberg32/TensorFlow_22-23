import tensorflow as tf
import training

SUBTASK = 0 # optino 0 oder 1 (0 being subtask 1 and 1 being subtask 2)
LEARNING_RATE = 0.001
OPTIMIZER = tf.optimizers.Adam() #tf.optimizers.SGD(learning_rate = LEARNING_RATE)

training.training_mnist(SUBTASK,OPTIMIZER)