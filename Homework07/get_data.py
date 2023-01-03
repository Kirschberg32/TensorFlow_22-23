import tensorflow_datasets as tfds
import tensorflow as tf

# preprocessing the data
def load_data(info : bool = False):
    """ loads the mnst dataset from tensorflow datasets 
    
    Parameters: 
        info (bool) = wether you want some info to be displayed and also additionaly returned
    """

    ( train_ds , test_ds ) , ds_info = tfds.load("mnist", split =[ "train", "test"], as_supervised = True , with_info = True )

    if(info):
        print(ds_info)
        tfds.show_examples(train_ds, ds_info)
        return (train_ds, test_ds) , ds_info

    return (train_ds, test_ds) , ds_info

def data_preprocess(data, batch_size = 64, sequence = 6):
    """ creates a data pipeline to preprocess the tensorflow datasets mnst dataset
    
    Parameters: 
        data (tensorflow.data.Dataset) = the dataset to preprocess
    """

    # cast to float
    data = data.map(lambda img, target: (tf.cast(img, tf.float32), target))
    # flatten
    #data = data.map(lambda img, target: (tf.reshape(img, (-1,)), target))
    # normalize the image values
    data = data.map(lambda img, target: ((img/128.)-1., target))
    # creating one-hot-vectors 
    data = data.map(lambda img, target: (img, tf.one_hot(target, depth=sequence)))

    # alternate positive, negative target values
    range_vals = tf.range(sequence)

    data = data.map(lambda img, target:(img, tf.where(tf.math.floormod(range_vals,2)==0, target, -target)))

    data = data.map(lambda img, target:(img, (tf.math.cumsum(target))))
    
    #cache shuffle, batch, prefetch
    data = data.cache()
    data = data.shuffle(1000)
    data = data.batch(batch_size) 
        
    data = data.prefetch(tf.data.AUTOTUNE)
    return data