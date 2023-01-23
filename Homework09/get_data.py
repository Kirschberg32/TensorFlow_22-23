import tensorflow as tf
import urllib
import os
import numpy as np

def download_data(category = 'candle'):
    categories = [line.rstrip(b'\n') for line in urllib.request.urlopen('https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt')]
    #print(categories[:10])

    # Creates a folder to download the original drawings into.
    # We chose to use the numpy format : 1x784 pixel vectors, with values going from 0 (white) to 255 (black). We reshape them later to 28x28 grids and 
    # normalize the pixel intensity to [-1, 1]

    if not os.path.isdir("npy_files"):
        os.mkdir("npy_files")

    url = f'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{category}.npy'  
    urllib.request.urlretrieve(url, f'npy_files/{category}.npy')

def load_data(category = 'candle', sample_size = 10000, train_size = 0.8):
    """ loads the quickdraw-dataset from google"""

    #download_data(category) # only if not downloaded yet

    images = np.load(f'npy_files/{category}.npy')
    #print(f'{len(images)} images to train on')

    # limited the amount of images used
    images = images[:sample_size]

    # define test and train images you want to use
    train_len = int(sample_size * train_size)
    test_len = int(sample_size - train_len)
    train_images, test_images = images[:train_len], images[train_len:]

    # Notice that this to numpy format contains 1x784 pixel vectors, with values going from 0 (white) to 255 (black). We reshape them later to 28x28 grids and normalize the pixel intensity to [-1, 1]
    train = tf.reshape(train_images, [train_len, 28, 28, 1])
    test = tf.reshape(test_images, [test_len, 28, 28, 1])
    train_ds = tf.data.Dataset.from_tensor_slices(train)
    test_ds = tf.data.Dataset.from_tensor_slices(test)

    return train_ds, test_ds

def data_preprocess(data, batch_size = 64):
    """ creates a data pipeline to preprocess the tensorflow datasets mnst dataset
    
    Parameters: 
        data (tf.data.Dataset) = the dataset to preprocess
    """

    # cast to float
    data = data.map(lambda img: tf.cast(tf.convert_to_tensor(img), tf.float32))
    # normalize the image values
    data = data.map(lambda img: (img/127)-1) # create in between -1 and 1

    # add target to the images
    data = data.map(lambda img: (img,tf.constant(1))) # one being that it is a true image
    
    #cache shuffle, batch, prefetch
    data = data.cache()
    data = data.shuffle(3000)
    #data = data.batch(batch_size)
        
    data = data.prefetch(tf.data.AUTOTUNE)
    return data

def create_latent_space(l_size : tf.int32 = 100, data_size : tf.int32 = 10000, batch_size : int = 64):
    """
    creates a random latent space as input for a generator
    """

    # create random noise for the generator and make a Dataset out of it
    noise = tf.random.normal(shape=(data_size, l_size)) # we have 10000 training samples

    # make dataset out of it and add an target
    noise = tf.data.Dataset.from_tensor_slices(noise).map(lambda noise_sample: (noise_sample,tf.constant(0))).batch(batch_size, drop_remainder = True)
    return noise