import tensorflow as tf

def load_data(path):
    with open(path) as f:
        data = f.readlines()
    return data

def data_preprocess(data, batch_size :int = 64):
    """ creates a data pipeline to preprocess the tensorflow datasets mnst dataset
    
    Parameters: 
        data (tf.data.Dataset) = the dataset to preprocess
        batch_size (int) = size of the batches
    """
    #cache shuffle, batch, prefetch
    data = data.cache()
    data = data.shuffle(3000)
    data = data.batch(batch_size)     
    data = data.prefetch(tf.data.AUTOTUNE)
    return data

def get_preprocessed_data(path : str, most_common_size : int = 10000,window_size : int = 2, train_part : float = 0.8):
    """ loads and fully prepares the dataset, returns the tokanizer for later usage"""
    
    data = load_data(path)
    #tokenization
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=most_common_size,
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n1234567890',
    )
    print("After getting tokenizer")
    tokenizer.fit_on_texts(data)
    print("After fit on text")
    tokens = tokenizer.texts_to_sequences(data)
    print("after text to sequence")
    tokens = [i for sublst in tokens for i in sublst if i]
    print("flattend")

    # pairing
    pairs, targets = tf.keras.preprocessing.sequence.skipgrams(tokens, most_common_size, window_size)# default shuffle here and creates negative samples

    print("done pairing")

    # split in train and test set
    pairs_len = len(pairs)
    print("Pairs length: ", pairs_len)
    train_len = int(pairs_len * train_part)
    train_pairs, test_pairs = pairs[:train_len], pairs[train_len:]
    train_targets, test_targets = targets[:train_len], targets[train_len:]
    print("done splitting")

    # tf Dataset and Preprocess
    train_ds = tf.data.Dataset.from_tensor_slices(train_pairs)
    test_ds = tf.data.Dataset.from_tensor_slices(test_pairs)
    train_targets = tf.data.Dataset.from_tensor_slices(train_targets)
    test_targets = tf.data.Dataset.from_tensor_slices(test_targets)

    train_ds = tf.data.Dataset.zip((train_ds,train_targets))
    test_ds = tf.data.Dataset.zip((test_ds,test_targets))


    print("done as Datasets")
    train_ds = data_preprocess(train_ds)
    test_ds = data_preprocess(test_ds)
    print("Done preprocessing")

    return (train_ds, test_ds), tokenizer