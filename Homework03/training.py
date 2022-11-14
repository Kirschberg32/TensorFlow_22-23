import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from model import MyModel

def training_mnist(model : MyModel, training_data, test_data, loss_function = tf.keras.losses.CategoricalCrossentropy(), optimizer = tf.keras.optimizers.SGD(learning_rate = 0.1,momentum=0), epochs : int = 10,visualization_training_loss = list(),visualization_training_accuracy = list(),visualization_test_loss = list(),visualization_test_accuracy = list()):
    """ trains a model given the data (created for training on mnist) using backpropagation
    
    Parameters: 
        model (MyModel) = the model to train
        training_data = the dataset to train on
        test_data = the dataset to test on
        loss_function = the loss function
        optimizer = the optimizer to use for the training
        epochs (int) = how many epochs to train for
        visualization_training_loss = the training loss will be saved in here
        visualization_training_accuracy = the training accuracy will be saved in here
        visualization_test_loss = the test loss will be saved in here
        visualization_test_accuracy = the test accuracy will be saved in here
    """

    # get performance before training
    before_training_training_loss,before_training_training_accuracy = testing_mnist(model,training_data,loss_function)
    visualization_training_loss.append(before_training_training_loss)
    visualization_training_accuracy.append(before_training_training_accuracy)
    before_training_test_loss, before_training_test_accuracy = testing_mnist(model,test_data,loss_function)
    visualization_test_loss.append(before_training_test_loss)
    visualization_test_accuracy.append(before_training_test_accuracy)

    for e in range (epochs):
        print("Epoch: ", e +1)

        epoch_losses = list()
        epoch_accuracy = list()
        for (images, targets) in training_data:

            with tf.GradientTape() as tape: 

                predictions = model(images)
                loss = loss_function(targets, predictions)

            # get the gradients
            gradients = tape.gradient(loss,model.trainable_variables)

            # apply the gradient
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # save loss and accuracy
            epoch_losses.append(loss)
            accuracy = np.mean( np.argmax(targets,axis=1) == np.argmax(predictions,axis=1) )
            epoch_accuracy.append(accuracy)

        visualization_training_loss.append(tf.reduce_mean(epoch_losses))
        visualization_training_accuracy.append(tf.reduce_mean(epoch_accuracy))

        # get performance after each epoch for the test data
        during_training_test_loss, during_training_test_accuracy = testing_mnist(model,test_data,loss_function)
        visualization_test_loss.append(during_training_test_loss)
        visualization_test_accuracy.append(during_training_test_accuracy)

    return visualization_training_loss, visualization_training_accuracy, visualization_test_loss, visualization_test_accuracy


def testing_mnist(model : MyModel,data,loss_function):
    """ calculates loss and accuracy of a model on a given data
    
    Parameters: 
        model (MyModel) = the model to test
        data = the data to test on (inputs,targets) for each sample
        loss_function = the loss function to calculate the loss with
    """

    test_accuracy_list = list()
    test_loss_list = list()

    for (img,target) in data:

        prediction = model(img)

        loss = loss_function(target, prediction)
        accuracy = np.mean( np.argmax(target,axis=1) == np.argmax(prediction,axis=1) )

        test_accuracy_list.append(accuracy)
        test_loss_list.append(loss)

    return tf.reduce_mean(test_loss_list) , tf.reduce_mean(test_accuracy_list)

def plot_training_results(training_loss, training_accuracy, test_loss, test_accuracy):
    """ Prints the final loss and accuracy and plots them over the whole training procedure.
    All inputs need to have the same length (1D)
    
    Parameters: 
        training_loss = training loss for each epoch
        training_accuracy = training accuracy for each epoch
        test_loss = test loss for each epoch
        test_accuracy = test accuracy for each epoch
    """

    # print the results of the last epoch
    print("Training loss: ", training_loss[-1].numpy())
    print("Training accuracy: ", training_accuracy[-1].numpy())
    print("Test loss: ", test_loss[-1].numpy())
    print("Test accuarcy: ", test_accuracy[-1].numpy())

    epochs = len(training_loss) # one more than for training because one measurement before training started
    plt.plot(range(epochs),training_loss,"g", label="training loss")
    plt.plot(range(epochs),test_loss,"b",label="test loss")
    plt.plot(range(epochs),training_accuracy,"r",label="training accuracy")
    plt.plot(range(epochs),test_accuracy,label = "test accuracy")
    

    plt.legend(loc="upper right")
    plt.xlabel("Epochs")
    plt.ylabel("Loss/Accuracy")
    plt.show()


