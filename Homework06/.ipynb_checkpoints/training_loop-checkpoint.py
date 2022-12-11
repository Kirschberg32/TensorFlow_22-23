import tensorflow as tf
import tqdm
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def training_cifar(training_data, test_data, model,epochs,plot = False, save = False, config_name = "Default"):
    """ a whole training circle on cifar-10
    
    Parameters: 
        training_data = the dataset to train on
        test_data = the dataset to validate with
        model = the model to train
        epochs = how many epochs to train for
        plot (boolean) = whether to plot the results additionaly to tensorboard with matplotlib at the end
    """

    # create directory for logs
    optimizer_name = model.optimizer.__dict__.get('_name')
    if (optimizer_name == None):
        optimizer_name = model.optimizer.__dict__.get('name')
    elif(optimizer_name == "SGD"):
        optimizer_name += "_Momentum-" + str(model.optimizer.__dict__.get('_hyper')['momentum'])
        
    print("Optimizer: " + optimizer_name)

    time_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # custom instead of time_string
    config = config_name
    
    train_file_path = f"logs/{config_name}/{time_string}/train"
    test_file_path = f"logs/{config_name}/{time_string}/test"
    train_summary_writer = tf.summary.create_file_writer(train_file_path)
    test_summary_writer = tf.summary.create_file_writer(test_file_path)

    # create dataframe for plotting the data after training
    data_frame = pd.DataFrame(data=None,columns=["mode","tag","step","value"])

    for e in range (epochs):

        # training
        for data in tqdm.tqdm(training_data,position=0,leave=True):
            metrics = model.train_step(data)

        # log in tensorboard and print
        with train_summary_writer.as_default():
            [tf.summary.scalar(name = m.name, data = m.result(),step=e) for m in model.metrics]


        # safe in data_frame for plotting at the end
        for (k,v) in metrics.items():
            data_frame.loc[len(data_frame)] = ["train",k,e,v.numpy()]

        [ tf.print(f"Epoch {e} {k}: {v.numpy()} ") for (k,v) in metrics.items() ]

        model.reset_metrics()

        # validate
        for data in tqdm.tqdm(test_data,position=0,leave=True):
            metrics = model.test_step(data)

        # tensorboard and print
        with test_summary_writer.as_default():
            [ tf.summary.scalar(name = "v_" + m.name, data = m.result(),step=e) for m in model.metrices]

        # safe in data_frame for plotting at the end
        for (k,v) in metrics.items():
            data_frame.loc[len(data_frame)] = ["test",k,e,v.numpy()]

        [ tf.print(f"Epoch {e} Validation {k}: {v.numpy()} ") for (k,v) in metrics.items() ]
        tf.print()

        model.reset_metrics()

    # before plotting save results in a csv
    if save:
        os.makedirs(f"csvs/{config_name}", exist_ok = True)
        data_frame.to_csv(f"csvs/{config_name}/{time_string}")

    if plot:
        
        os.makedirs(f"Plots/{config_name}", exist_ok = True)
        # from: https://www.tensorflow.org/tensorboard/dataframe_api
        # sadly the three following lines do not work for the locally saved data yet 
        # when they will or slightly modified, we do not need to save everything in the data frame while training
        # experiment = tb.data.experimental.ExperimentFromDev(f"logs/{optimizer_name}/{time_string}")
        # data_frame = experiment.get_scalars()#pivot=True)
        # training_vs_testing = data_frame.run.apply(lambda run: run.split(",")[0])

        plt.figure(figsize=(16, 6))
        sns.lineplot(data=data_frame,x="step",y="value",hue="tag",style="mode").set_title("results")
        plt.savefig(f"Plots//{config_name}/{time_string}.png")
        plt.show()