import get_data
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt

epochs = 1
batch_size = 64
noise_std = 0.2

# get and prepare data and model
(training_data, val_data ) , ds_info = get_data.load_data(False) # True
#training_data = training_data.take(2000) # 
training_data = get_data.data_preprocess(training_data, batch_size = batch_size,noisy = noise_std)
val_data = get_data.data_preprocess(val_data, batch_size = batch_size, noisy = noise_std)

