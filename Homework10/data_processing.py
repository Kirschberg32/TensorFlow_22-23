import tensorflow as tf

def load_data(path):
    with open(path) as f:
        data = f.read()
    return data

def get_tokenizer(num_words : int = 10000):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=num_words,
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n1234567890',
    )
    return tokenizer

def nlp_preprocess(data, most_common_size :int = 10000):
    """ takes a text and returns the tokens, or the most_common_size most common tokens 
    retuns a flat list with all the tokens of the text in order
    """
    
    if (most_common_size != None):
        tokenizer = get_tokenizer(most_common_size)
    else:
        tokenizer = get_tokenizer()
    tokenizer.fit_on_texts(data)
    data = tokenizer.texts_to_sequences(data)
    tokens = [i for sublst in data for i in sublst if i]
    return tokens, tokenizer

def pairing(tokens, vocabulary_size, win_len):
    """ creates pairs of words that are next ot each other in a window size of win_len """
    pairs, labels = tf.keras.preprocessing.sequence.skipgrams(tokens, vocabulary_size, win_len)
    return pairs, labels

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
    #preprocesse
    tokens, tokenizer = nlp_preprocess(data,most_common_size)
    pairs, targets = pairing(tokens, window_size) # default shuffle here and creates negative samples

    # split in train and test set
    pairs_len = len(pairs)
    train_len = int(pairs_len * train_part)
    train_pairs, test_pairs = pairs[:train_len], pairs[train_len:]
    train_targets, test_targets = targets[:train_len], targets[train_len:]

    # tf Dataset and Preprocess
    train_ds = tf.data.Dataset.from_tensor_slices((train_pairs,train_targets))
    test_ds = tf.data.Dataset.from_tensor_slices((test_pairs,test_targets))
    train_ds = data_preprocess(train_ds)
    test_ds = data_preprocess(test_ds)

    return (train_ds, test_ds), tokenizer