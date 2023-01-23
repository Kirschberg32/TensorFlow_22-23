import tensorflow as tf
import tqdm
import datetime

import get_data

def training_candles(training_data, test_data, discriminator, generator,epochs, config_name = "Default",latent_space =100, batch_size = 64,sample_size = 10000,train_size = 0.8):
    """ a whole training circle
    
    Parameters: 
        training_data = the dataset to train on
        test_data = the dataset to validate with
        discriminator = the discriminator
        generator = the generator of images
        epochs = how many epochs to train for
        config_name (String) = the folders where we save stuff will be called like that
        latent_space (int) = the size of the latent space
        batch_size
        sample_size
        train_size
    """

    # create directory for logs
    time_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    train_file_path = f"logs/{config_name}/{time_string}/train"
    test_file_path = f"logs/{config_name}/{time_string}/test"
    train_summary_writer = tf.summary.create_file_writer(train_file_path)
    test_summary_writer = tf.summary.create_file_writer(test_file_path)

    example_images = []

    # create int32 values for train size and test size
    train_len = tf.cast(sample_size * train_size, tf.int32)
    test_len = tf.cast(sample_size - train_len, tf.int32)

    for e in range (epochs):
        
        # create training full dataset
        noise_train = get_data.create_latent_space(latent_space, train_len, batch_size)
        # feed noise to generator
        generated_images_train = noise_train.map(lambda noise_batch,targets_batch: (generator(noise_batch,training=False),targets_batch)).unbatch()

        # zip generated_images and original images to one 
        full_dataset_train = training_data.concatenate(generated_images_train).shuffle(4000).batch(batch_size, drop_remainder = True)

        # create testing full dataset
        noise_test = get_data.create_latent_space(latent_space,test_len, batch_size)
        # feed noise to generator
        generated_images_test = noise_test.map(lambda noise_batch,targets_batch: (generator(noise_batch,training=False),targets_batch)).unbatch()

        # zip generated_images and original images to one 
        full_dataset_test = test_data.concatenate(generated_images_test).shuffle(3000).batch(batch_size, drop_remainder = True)

        # training the discriminator
        history = discriminator.fit(full_dataset_train, validation_data = full_dataset_test)
        metrics = history.history

        # training the generator
        for n in tqdm.tqdm(noise_train,position=0,leave=True):
            generator.train_step_indi(n,discriminator)

        # log in tensorboard and print
        with train_summary_writer.as_default():
            [tf.summary.scalar(name = m.name, data = m.result(),step=e) for m in discriminator.metrics]

        [ tf.print(f"Epoch {e} {k}: {v[0]}") for (k,v) in metrics.items() ]

        #discriminator.reset_metrics()

        # testing the discriminator
        #metrics = discriminator.evaluate(full_dataset_test,return_dict = True)

        # log in tensorboard and print
        #with test_summary_writer.as_default():
            #[tf.summary.scalar(name = m.name, data = m.result(),step=e) for m in discriminator.metrics]

        #[ tf.print(f"V Epoch {e} {k}: {v.numpy()}") for (k,v) in metrics.items() ]

        for n_test in noise_test:
            n, _ = n_test
            ex_images = generator(n)
            break
        output = discriminator(ex_images)
        example_images.append([ex_images , output])

        discriminator.reset_metrics()

    return example_images