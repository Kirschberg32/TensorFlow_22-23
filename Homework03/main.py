import mnist_data
from model import MyModel
import training

import tensorflow as tf
import time

### • How many training/test images are there?
#- 60,000 training images and 10,000 test images

### • What’s the image shape?
#- The image shapes are pixelated handwritten single digits
#- The image is in shape 28/28/1

### • What range are pixel values in?
#- They are in the range between 0 - 255 

# Hyperparameters
EPOCHS = 10
LEARNING_RATE = 0.1
MOMENTUM = 0

loss_function = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.SGD(learning_rate = LEARNING_RATE,momentum=MOMENTUM)

# Loading and preprocessing the data
( train_ds , test_ds ), _ = mnist_data.load_mnst(info = False)

train_processed = train_ds.apply(mnist_data.mnist_preprocess)
test_processed = test_ds.apply(mnist_data.mnist_preprocess)

# Model creation and training
model = MyModel([256,256,],10)
#model = MyModel([128,],10) # smallest good model which is comparable to us

start = time.time()
training_loss,training_accuracies,test_loss,test_accuracies = training.training_mnist(model,train_processed,test_processed,loss_function,optimizer,EPOCHS)
end = time.time()

print("Time needed for training: ", end - start , " sec")
training.plot_training_results(training_loss,training_accuracies,test_loss,test_accuracies)