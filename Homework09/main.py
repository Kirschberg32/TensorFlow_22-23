import get_data
from model import MyCNN, MyGenerator
from training_loop import training_candles

import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np

config = "run-test"
category = 'candle'
epochs = 2 # around 10
batch_size = 64
latent_space = 100
sample_size = 10000 # 10000
train_size = 0.8

optimizer = tf.optimizers.Adam(learning_rate=0.001)
loss = tf.losses.BinaryCrossentropy()

# get and prepare data and model

training_data, o_val_data = get_data.load_data(category, sample_size, train_size = train_size)
#training_data = training_data.take(10) # 
training_data = get_data.data_preprocess(training_data, batch_size = batch_size)
val_data = get_data.data_preprocess(o_val_data, batch_size = batch_size)

discriminator = MyCNN(1,filter_start = 24, regularizer=tf.keras.regularizers.L2(0.001))
generator = MyGenerator(filter_start = 24,final_shape = 28, batch_size = 64)

# compile
discriminator.compile(optimizer = optimizer, loss=loss)
generator.compile(optimizer = optimizer,loss=loss)

# train
example_images = training_candles(training_data, val_data, discriminator, generator, epochs, config_name = config,latent_space = latent_space, batch_size=batch_size, sample_size = sample_size, train_size = train_size)

# save the trained model
discriminator.save(f"saved_discriminator/{config}")
generator.save(f"saved_generator/{config}")

# plot the example_images
f, axes = plt.subplots(epochs,1)
for i in range(epochs):
    axes[i].imshow(example_images[i][0][0])
    number = np.around(example_images[i][1][0].numpy(),decimals=2)[0]
    axes[i].set_title(f"E: {i}; P: {number}")
plt.savefig(f"Plots/{config}_example_images.png")
plt.show()