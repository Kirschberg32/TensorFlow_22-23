import tensorflow_datasets as tfds
import tensorflow as tf

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

def data_preprocess(data, batch_size = 64,noisy = 0.1,targets = False):
    """ creates a data pipeline to preprocess the tensorflow datasets mnst dataset
    
    Parameters: 
        data (tensorflow.data.Dataset) = the dataset to preprocess
        noisy (float) = standard deviation of the noise to add around 0
    """

    # cast to float
    data = data.map(lambda img, target: (tf.cast(img, tf.float32), target))
    # normalize the image values
    data = data.map(lambda img, target: ((img/255), target)) # create in between 0 and 1

    # create the specific data we want: 

    # add a color channel dimension (the color channel dimension already exists in mnist)
    # data = data.map(lambda img, target: ( tf.expand_dims(img,axis=-1), target ) )

    if targets:
        targets_data = data.map(lambda img, target: target)

    # remove the targets, add noise instead
    data = data.map(lambda img, target: (tf.random.normal(shape=img.shape,mean=0,stddev=noisy), img) )


    # add noise to images and save in new dataset
    data = data.map(lambda noise, img: (tf.add(noise,img),img))
    # keep image in the right area
    data = data.map(lambda noise, img: (tf.clip_by_value(noise,clip_value_min=-1,clip_value_max=1),img))

    if targets: 
        data = tf.data.Dataset.zip((data,targets_data))
        data = data.map(lambda images, targets: (images[0],images[1],targets))
    
    #cache shuffle, batch, prefetch
    data = data.cache()
    data = data.shuffle(3000)
    data = data.batch(batch_size)
        
    data = data.prefetch(tf.data.AUTOTUNE)
    return data
