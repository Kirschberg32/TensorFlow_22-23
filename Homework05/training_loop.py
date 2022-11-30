import tensorflow as tf
import tqdm
import datetime

def training_cifar(training_data, test_data, model,epochs):
    """ a whole training cicle on mnist
    
    Parameters: 
        training_data = the dataset to train on
        test_data = the dataset to validate with
        model = the model to train
        epochs = how many epochs to train for
    """

    # create directory for logs
    optimizer_name = model.optimizer.__dict__.get('_name')
    if (optimizer_name == None):
        optimizer_name = model.optimizer.__dict__.get('name')
    elif(optimizer_name == "SGD"):
        optimizer_name += "_Momentum-" + str(model.optimizer.__dict__.get('_hyper')['momentum'])
        
    print("Optimizer: " + optimizer_name)

    time_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    train_file_path = f"logs/{optimizer_name}/{time_string}/train"
    test_file_path = f"logs/{optimizer_name}/{time_string}/test"
    train_summary_writer = tf.summary.create_file_writer(train_file_path)
    test_summary_writer = tf.summary.create_file_writer(test_file_path)

    for e in range (epochs):

        # training
        for data in tqdm.tqdm(training_data,position=0,leave=True):
            metrics = model.train_step(data)

        # log in tensorboard and print
        with train_summary_writer.as_default():
            [ tf.summary.scalar(name = m.name, data = m.result(),step=e) for m in model.metrics ]

        [ tf.print(f"Epoch {e} {k}: {v.numpy()} ") for (k,v) in metrics.items() ]

        model.reset_metrics()

        # validate
        for data in tqdm.tqdm(test_data,position=0,leave=True):
            metrics = model.test_step(data)

        # tensorboard and print
        with test_summary_writer.as_default():
            [ tf.summary.scalar(name = "v_" + m.name, data = m.result(),step=e) for m in model.metrices]

        [ tf.print(f"Epoch {e} Validation {k}: {v.numpy()} ") for (k,v) in metrics.items() ]
        tf.print()

        model.reset_metrics()