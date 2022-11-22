import tensorflow as tf
import datetime
import tqdm

from model import MyModel
import mnist_data as mnist

def training_mnist(subtask,optimizer = tf.optimizers.SGD()):

    config_name = "Run-42"
    #current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_file_path = f"logs/{config_name}/train"
    test_file_path = f"logs/{config_name}/test"
    train_summary_writer = tf.summary.create_file_writer(train_file_path)
    test_summary_writer = tf.summary.create_file_writer(test_file_path)
    
    epochs = 100
    loss_function = tf.losses.BinaryCrossentropy() if subtask == 0 else tf.losses.MeanSquaredError()
    output_activ = tf.nn.sigmoid if subtask == 0 else tf.keras.activations.linear

    training_data, test_data = mnist.get_processed_data(subtask)

    model = MyModel(loss_function,optimizer,[256,256,256,128],1,output_activation = output_activ)

    for e in range (epochs):

        # training
        for data in tqdm.tqdm(training_data,position=0,leave=True):
            metrics = model.train_step(data)

        # log in tensorboard
        with train_summary_writer.as_default():
            [ tf.summary.scalar(name = m.name, data = m.result(),step=e) for m in model.metrics ]

        [ tf.print(f"Epoch {e} {k}: {v.numpy()} ") for (k,v) in metrics.items() ]

        model.reset_metrics()

        # validate
        for data in tqdm.tqdm(test_data,position=0,leave=True):
            metrics = model.test_step(data)

        # tensorboard
        with test_summary_writer.as_default():
            [ tf.summary.scalar(name = "v_" + m.name, data = m.result(),step=e) for m in model.metrices]

        [ tf.print(f"Epoch {e} Validation {k}: {v.numpy()} ") for (k,v) in metrics.items() ]
        tf.print()

        model.reset_metrics()