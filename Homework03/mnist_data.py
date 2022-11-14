import tensorflow_datasets as tfds
import tensorflow as tf

def load_mnst(info : bool = False):
    """ loads the mnst dataset from tensorflow datasets 
    
    Parameters: 
        info (bool) = wether you want some info to be displayed and also additionaly returned
    """

    ( train_ds , test_ds ) , ds_info = tfds . load("mnist", split =[ "train", "test"], as_supervised = True , with_info = True )

    if(info):
        print(ds_info)
        tfds.show_examples(train_ds, ds_info)
        return (train_ds, test_ds) , ds_info

    return (train_ds, test_ds) , ds_info

def mnist_preprocess(mnist):
    """ creates a data pipeline to preprocess the tensorflow datasets mnst dataset
    
    Parameters: 
        mnst (tensorflow.data.Dataset) = the dataset to preprocess
    """

    # cast to float
    mnist = mnist.map(lambda img, target: (tf.cast(img, tf.float32), target))
    # flatten
    mnist = mnist.map(lambda img, target: (tf.reshape(img, (-1,)), target))
    # normalize the image values
    mnist = mnist.map(lambda img, target: ((img/128.)-1., target))
    # creating one-hot-vectors 
    mnist = mnist.map(lambda img, target: (img, tf.one_hot(target, depth=10)))
    # cache shuffle, batch, prefetch
    mnist = mnist.cache()
    mnist = mnist.shuffle(1000)
    mnist = mnist.batch(64)
    mnist = mnist.prefetch(20)
    return mnist