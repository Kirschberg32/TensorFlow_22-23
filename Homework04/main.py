import tensorflow as tf
import training


# in a notebook, load the tensorboard extension, not needed for scripts
%load_ext tensorboard

SUBTASKS = [0, 1] # optino 0 oder 1 (0 being subtask 1 and 1 being subtask 2)
LEARNING_RATE = 0.001
OPTIMIZERS = [[tf.optimizers.Adam(), tf.optimizers.SGD(learning_rate = LEARNING_RATE), tf.optimizers.SGD(learning_rate = LEARNING_RATE, momentum = 0.9), tf.keras.optimizers.experimental.RMSprop(learning_rate = LEARNING_RATE), tf.keras.optimizers.experimental.Adagrad(learning_rate = LEARNING_RATE)], 
            [tf.optimizers.Adam(), tf.optimizers.SGD(learning_rate = LEARNING_RATE), tf.optimizers.SGD(learning_rate = LEARNING_RATE, momentum = 0.9), tf.keras.optimizers.experimental.RMSprop(learning_rate = LEARNING_RATE), tf.keras.optimizers.experimental.Adagrad(learning_rate = LEARNING_RATE)]]
for SUBTASK in SUBTASKS:
    for OPTIMIZER in OPTIMIZERS[SUBTASK]:
        training.training_mnist(SUBTASK,OPTIMIZER)
    
%tensorboard --logdir logs/