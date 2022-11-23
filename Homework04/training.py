import tensorflow as tf
import tqdm

from model import MyModel
import mnist_data as mnist

def training_mnist(subtask,optimizer = tf.optimizers.SGD()):
    """ a whole training cicle on mnist
    
    Parameters: 
        subtask (int) = which subtask of the homework sheet to do (0 or 1)
        optimizer = the optimizer to use
    """

    # create directory for logs
    config_name = "SUB_" + str(subtask + 1)
    optimizer_name = optimizer.__dict__.get('_name')
    if (optimizer_name == None):
        optimizer_name = optimizer.__dict__.get('name')
    elif(optimizer_name == "SGD"):
        optimizer_name += "_Momentum-" + str(optimizer.__dict__.get('_hyper')['momentum'])
        
    print("Optimizer: " + optimizer_name)
    
    train_file_path = f"logs/{config_name}/{optimizer_name}/train"
    test_file_path = f"logs/{config_name}/{optimizer_name}/test"
    train_summary_writer = tf.summary.create_file_writer(train_file_path)
    test_summary_writer = tf.summary.create_file_writer(test_file_path)
    
    # choose which loss function and output activation depending on the subtask to solve
    epochs = 100
    loss_function = tf.losses.BinaryCrossentropy() if subtask == 0 else tf.losses.MeanSquaredError()
    output_activ = tf.nn.sigmoid if subtask == 0 else tf.keras.activations.linear

    # get data and model
    training_data, test_data = mnist.get_processed_data(subtask)
    model = MyModel(loss_function,optimizer,[256,256,256,128],1,output_activation = output_activ)

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