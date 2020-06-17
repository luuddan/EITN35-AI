#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

import os
import random
import gc  # garbage collector for cleaning deleted data from memory

from keras.models import Model
from tensorflow.keras import models, layers, regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout
import sys
import math
import time
import getopt

print ('Nbr of arguments: ' , len(sys.argv) , 'total')
print ('Arguments list: ' , str(sys.argv))

learning_rate_input =float(sys.argv[1])
drop_param =float(sys.argv[2])
reg_param = float(sys.argv[3])
run_number =float(sys.argv[4])
data_fraction = float(sys.argv[5])
batch_size = int(sys.argv[6])

train_dir = 'C:/Users/eitn35/Documents/EITN35_EVOLVE/image_frames/data_set_' + str(data_fraction) + '/train_set_' + str(data_fraction) + '/'
val_dir = 'C:/Users/eitn35/Documents/EITN35_EVOLVE/image_frames/data_set_' + str(data_fraction) + '/val_set_' + str(data_fraction) + '/'
test_dir = 'C:/Users/eitn35/Documents/EITN35_EVOLVE/image_frames/data_set_' + str(data_fraction) + '/test_set_' + str(data_fraction) + '/'
save_dir = 'C:/Users/eitn35/Documents/EITN35_EVOLVE/models_and_weights_EVOLVE/models/saved_models_and_weights/'
work_dir = 'C:/Users/eitn35/PycharmProjects/EITN35-AI/ImageRecognition/'

test_imgs = [test_dir + '{}'.format(i) for i in os.listdir(test_dir)]  # get test images
val_imgs = [val_dir + '{}'.format(i) for i in os.listdir(val_dir)]  # get val images
train_imgs = [train_dir + '{}'.format(i) for i in os.listdir(train_dir)]  # get test images

# shuffle it randomly
random.shuffle(train_imgs)
random.shuffle(val_imgs)
random.shuffle(test_imgs)

gc.collect()  # collect garbage to save memory

import matplotlib.image as mpimg

# Lets declare our image dimensions
# we are using coloured images.
nrows = 224
ncolumns = 224
channels = 3  # change to 1 if you want to use grayscale image


# A function to read and process the images to an acceptable format for our model
def read_and_process_image(list_of_images):
    """
    Returns two arrays:
        X is an array of resized images
        y is an array of labels
    """
    X = []  # images
    y = []  # labels
    i = 0
    for image in list_of_images:
        # ändra här mellan COLOR och GRAYSCALE beroende på antal channels
        X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (nrows, ncolumns),
                            interpolation=cv2.INTER_CUBIC))  # Read the image
        # get the labels
        if 'persons_1' in image:
            y.append(1)
        elif 'dogs_1' in image:
            y.append(2)
        elif 'bikes_1' in image:
            y.append(3)
        else:
            y.append(0)
        i += 1
    return X, y

class_names = ['empty', 'person', 'dogs', 'bikes']

X_train, y_train = read_and_process_image(train_imgs)
X_val, y_val = read_and_process_image(val_imgs)
X_test, y_test = read_and_process_image(test_imgs)

import seaborn as sns

gc.collect()

# Convert list to numpy array
X_train = np.array(X_train)
y_train = np.array(y_train)
X_val = np.array(X_val)
y_val = np.array(y_val)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Lets split the data into train and test set
print("Shape of train images is:", X_train.shape)
print("Shape of train labels is:", y_train.shape)
print("Shape of validation images is:", X_val.shape)
print("Shape of validation labels is:", y_val.shape)
print("Shape of test images is:", X_test.shape)
print("Shape of test labels is:", y_test.shape)

gc.collect()

# get the length of the train and validation data
ntrain = len(X_train)
nval = len(X_val)

# We will use a batch size of 32. Note: batch size should be a factor of 2.***4,8,16,32,64...***
#batch_size = float(sys.argv[6])

my_callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)] #was 20

# Change number of layers for correct model saving
no_layers = 6

# Model Setup
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), padding = 'same', activation='relu', input_shape=(ncolumns, nrows, 3),kernel_regularizer = tf.keras.regularizers.l2(l=reg_param)))  # input ska var (150, 150, 3)
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(drop_param))
model.add(layers.Conv2D(32, (3, 3), padding = 'same', activation='relu',kernel_regularizer = tf.keras.regularizers.l2(l=reg_param)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(drop_param))
model.add(layers.Conv2D(64, (3, 3), padding = 'same', activation='relu',kernel_regularizer = tf.keras.regularizers.l2(l=reg_param)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(drop_param))
model.add(layers.Conv2D(64, (3, 3), padding = 'same', activation='relu',kernel_regularizer = tf.keras.regularizers.l2(l=reg_param)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(drop_param))
model.add(layers.Dense(256, activation='relu',kernel_regularizer = tf.keras.regularizers.l2(l=reg_param)))
model.add(layers.Dense(4))  # Sigmoid function at the end because we have just two classes

# Model summary
model.summary()

# Compilation of model
opt = tf.keras.optimizers.SGD(learning_rate=learning_rate_input) #which optimizer should med used
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=opt, metrics=['accuracy'])

# Lets create the augmentation configuration
# This helps prevent overfitting, since we are using a small dataset
#train_datagen = ImageDataGenerator(rescale=1. / 255,  # Scale the image between 0 and 1
                                   #rotation_range=40,
                                   #width_shift_range=0.2,
                                   #height_shift_range=0.2,
                                   #shear_range=0.2,
                                  #zoom_range=0.2,
                                  #horizontal_flip=True, )

#val_datagen = ImageDataGenerator(rescale=1. / 255)  # We do not augment validation data. we only perform rescale

# Override current setup of ImageDataGenerator
train_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()

# Create the image generators
train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)

start_time = time.time()

history = model.fit_generator(train_generator, steps_per_epoch=ntrain // batch_size, epochs=300,
                              validation_data=val_generator, validation_steps=nval // batch_size,
                              callbacks=my_callbacks)
end_time = time.time()

# Save the model
os.chdir(save_dir)
model.save_weights('model_layers_' + str(no_layers) + '_data_frac_'+ str(data_fraction) + '_weights_run_'+ str(run_number)+ '.h5')
model.save('model_layers_' + str(no_layers) + '_data_frac_'+ str(data_fraction) + '_run_'+ str(run_number)+ '.h5')

os.chdir(work_dir)

# lets plot the train and val curve
# get the details form the history object
acc = history.history['acc']
acc = acc[1:]
val_acc = history.history['val_acc']
val_acc = val_acc[1:]
loss = history.history['loss']
loss = loss[1:]
val_loss = history.history['val_loss']
val_loss = val_loss[1:]

epochs = range(1, len(acc) + 1)

# Train and validation accuracy
plt.plot(epochs, acc, 'b', label='Training accurarcy')
plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
plt.title('Training and Validation accurarcy')
plt.legend()

plt.figure()
# Train and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()

#plt.show()

# Probability of test set
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(X_test)


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    #plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(4))
    plt.yticks([])
    thisplot = plt.bar(range(4), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions[i], y_test, X_test)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions[i], y_test)
plt.tight_layout()
#plt.show()

# Probability of test set
print("Shape of labels is:", predictions.shape)
threshold = 0.6
total_correct = 0
dogs_correct = 0
bikes_correct = 0
persons_correct = 0
empty_correct = 0
for i in range(len(y_test)):
    index_max = np.argmax(predictions[i])
    if y_test[i] == index_max and predictions[i][index_max] > threshold:
        total_correct += 1
        if index_max == 1:
            persons_correct += 1
        if index_max == 2:
            dogs_correct += 1
        if index_max == 3:
            bikes_correct += 1
        if index_max == 0:
            empty_correct += 1

#hard coded percentages for each category, i.e. 0.403 * test_set = test_set_persons
total_acc = total_correct / len(X_test)
person_acc = persons_correct / (len(X_test)*0.403)
dog_acc = dogs_correct / (len(X_test)*0.113)
bike_acc = bikes_correct / (len(X_test)*0.08)
empty_acc = empty_correct / (len(X_test)*0.403)

print("Total accuracy ", total_acc)
print("Person accuracy ", person_acc)
print("Dog accuracy ", dog_acc)
print("Bike accuracy ", bike_acc)
print("Empty accuracy ", empty_acc)

#project_directory = '/Users/august/PycharmProjects/EITN35/'
results = work_dir + 'training_results/'

def round_sig(x, sig=2):
    return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)


def print_to_file(acc, val_acc, loss, val_loss, total_acc, person_acc, dog_acc, bike_acc, empty_acc, epochs, reg_param):
    # New CSV file
    os.chdir(results)
    current_results = os.listdir(results)
    current_results.sort()
    working_file = current_results[len(os.listdir(results)) - 1]
    print("Latest file is: " + str(working_file))


    final_acc = acc[len(acc) - 1]
    final_val_acc = val_acc[len(val_acc)-1]
    final_loss = loss[len(loss) - 1]
    final_val_loss = val_loss[len(val_loss) - 1]
    var = final_acc - final_val_acc
    var = round_sig(var, 4)

    total_time = end_time - start_time
    total_time = round(total_time)

    file = open(working_file, "a+")
    file.write("\nRun_Result," + str(run_number))
    file.write("\nLayers," + str(no_layers))
    file.write("\nTrainable_param," + str(3.278116))
    file.write("\nData_set_fraction," + str(data_fraction))
    file.write("\nBatch_size," + str(batch_size))
    file.write("\nTraining," + str(final_acc))
    file.write("\nValidation," + str(final_val_acc))
    file.write("\nVariance," + str(var))
    file.write("\nTotal_test_acc," + str(total_acc))
    file.write("\nPerson_test_acc," + str(person_acc))
    file.write("\nDog_test_acc," + str(dog_acc))
    file.write("\nBike_test_acc," + str(bike_acc))
    file.write("\nEmpty_test_acc," + str(empty_acc))
    file.write("\nEpochs," + str(len(epochs)+1))
    file.write("\nL2_reg," + str(reg_param))
    file.write("\nDrop_rate," + str(drop_param))
    file.write("\nLearning_rate," + str(learning_rate_input))
    file.write("\nRun_time," + str(total_time))
    file.write("\n")
    file.close()


print_to_file(acc, val_acc, loss, val_loss, total_acc, person_acc, dog_acc, bike_acc, empty_acc, epochs, reg_param)





