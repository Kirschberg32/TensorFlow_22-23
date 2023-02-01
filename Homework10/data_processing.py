import re
import collections
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
    return tokens

def pairing(tokens, win_len):
    """ creates pairs of words that are next ot each other in a window size of win_len """
    pairs = []
    for i, word in enumerate(tokens):
        if i <= len(tokens) - win_len:
            for j in range(1, win_len):
                pairs.append([word, tokens[i + j]])
    return pairs

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

def get_preprocessed_data(path : str, ):
    data = load_data(path)
    pass

