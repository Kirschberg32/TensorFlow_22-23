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

    return train_ds, test_ds , ds_info

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
    return mnist

def mnist_final_process(mnist):
    """ does the caching, shuffling, batching and prefetching of the processing 
    
    Parameters: 
        mnst (tensorflow.data.Dataset) = the dataset to preprocess
    """

    mnist = mnist.cache()
    mnist = mnist.shuffle(1000)
    mnist = mnist.batch(64)
    mnist = mnist.prefetch(20)
    return mnist
    

def get_processed_data(subtask):
    """ loads and preprocesses the mnist dataset, with always two samples merged together, the target is decided by the target function dependent on the subtask given """
    
    training_data, test_data, _ = load_mnst(info=False)
    fn1 = lambda x,y: (x[0], y[0], x[1] + y[1] >= 5)
    fn2 = lambda x,y: (x[0],y[0],x[1]-y[1])
    target_function = fn1 if subtask == 0 else fn2

    training_data = training_data.apply(mnist_preprocess)
    test_data = test_data.apply(mnist_preprocess)

    # change dataset for certain condition
    double_training = tf.data.Dataset.zip((training_data.shuffle(4000),training_data.shuffle(2000)))
    double_test = tf.data.Dataset.zip((test_data.shuffle(4000),test_data.shuffle(2000)))
    double_training = double_training.map(target_function)
    double_test = double_test.map(target_function)

    final_training = double_training.apply(mnist_final_process)
    final_test = double_test.apply(mnist_final_process)

    return final_training, final_test