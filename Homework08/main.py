import get_data
from model import MyCNN, MyAutoencoder
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt

epochs = 1
batch_size = 64
noise_std = 0.2
embedding = 10

optimizer = tf.optimizers.Adam()

# get and prepare data and model
(training_data, val_data ) , ds_info = get_data.load_data(False) # True
#training_data = training_data.take(2000) # 
training_data = get_data.data_preprocess(training_data, batch_size = batch_size,noisy = noise_std)
val_data = get_data.data_preprocess(val_data, batch_size = batch_size, noisy = noise_std)

encoder = MyCNN(optimizer,embedding,regularizer=tf.keras.regularizers.L2(0.001))
# decoder = 