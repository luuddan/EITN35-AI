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





learning_rate_input = 0.001
drop_param = 0
reg_param = 0
run_number = 1
data_fraction = 1
batch_size = 32
test_dir = 'C:/Users/eitn35/Documents/EITN35/video_files/frames/CycleGan/TestSetCycleNight/GeneratedDayFinal/'
#test_dir = 'C:/Users/eitn35/Documents/EITN35/video_files/frames/CycleGan/TestSetCycleNight/Night/NightAllFinal/'
#test_dir = 'C:/Users/eitn35/Documents/EITN35/video_files/frames/CycleGan/TestSetCycleNight/Night/NightAllFinal/'
model_dir = 'C:/Users/eitn35/Documents/EITN35_EVOLVE/models_and_weights_EVOLVE/models/saved_models_and_weights/'
#test_dir = 'C:/Users/eitn35/Documents/EITN35_EVOLVE/image_frames/data_set_' + str(1.0) + '/test_set_' + str(1.0) + '/'

test_imgs = [test_dir + '{}'.format(i) for i in os.listdir(test_dir)]  # get test images


# shuffle it randomly

random.shuffle(test_imgs)


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

X_test, y_test = read_and_process_image(test_imgs)

import seaborn as sns

gc.collect()

# Convert list to numpy array

X_test = np.array(X_test)
y_test = np.array(y_test)

# Lets split the data into train and test set

print("Shape of test images is:", X_test.shape)
print("Shape of test labels is:", y_test.shape)

gc.collect()

# get the length of the train and validation data

# We will use a batch size of 32. Note: batch size should be a factor of 2.***4,8,16,32,64...***
#batch_size = float(sys.argv[6])

my_callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)] #was 20

# Change number of layers for correct model saving
no_layers = 6

# Model Setup
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), padding = 'same', activation='relu', input_shape=(ncolumns, nrows, 3),kernel_regularizer = tf.keras.regularizers.l2(l=reg_param)))  # input ska var (150, 150, 3)
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), padding = 'same', activation='relu',kernel_regularizer = tf.keras.regularizers.l2(l=reg_param)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), padding = 'same', activation='relu',kernel_regularizer = tf.keras.regularizers.l2(l=reg_param)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), padding = 'same', activation='relu',kernel_regularizer = tf.keras.regularizers.l2(l=reg_param)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(drop_param))
model.add(layers.Dense(128, activation='relu',kernel_regularizer = tf.keras.regularizers.l2(l=reg_param)))
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

model.load_weights(model_dir + 'model_layers_6_data_frac_1.0_weights_run_6.0.h5')

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
threshold = 0.1
total_correct = 0
dogs_correct = 0
bikes_correct = 0
persons_correct = 0
empty_correct = 0
confusion = np.zeros((4,4))
for i in range(len(y_test)):
    index_max = np.argmax(predictions[i])
    if y_test[i] == index_max and predictions[i][index_max] > threshold:
        total_correct += 1
        if index_max == 1:
            persons_correct += 1
            confusion[1, 1] += 1
        if index_max == 2:
            dogs_correct += 1
            confusion[2, 2] += 1
        if index_max == 3:
            bikes_correct += 1
            confusion[3, 3] += 1
        if index_max == 0:
            empty_correct += 1
            confusion[0, 0] += 1
    else:
        if y_test[i] == 0:
            if(index_max == 1):
                confusion[1,0] += 1
            if (index_max == 2):
                confusion[2,0] += 1
            if (index_max == 3):
                confusion[3,0] += 1
        if y_test[i] == 1:
            if(index_max == 0):
                confusion[0,1] += 1
            if (index_max == 2):
                confusion[2,1] += 1
            if (index_max == 3):
                confusion[3,1] += 1
        if y_test[i] == 2:
            if(index_max == 0):
                confusion[0,2] += 1
            if (index_max == 1):
                confusion[1,2] += 1
            if (index_max == 3):
                confusion[3,2] += 1
        if y_test[i] == 3:
            if(index_max == 0):
                confusion[0,3] += 1
            if (index_max == 1):
                confusion[1,3] += 1
            if (index_max == 2):
                confusion[2,3] += 1


        print(str(test_imgs[i]))
        print('Correct class: ' + str(y_test[i]))
        print('Guess class: ' + str(index_max))
        print('Confidence of guess: '+ str(predictions[i][index_max]))
        print('--------------------------------------------')

#hard coded percentages for each category, i.e. 0.403 * test_set = test_set_persons
total_acc = total_correct / len(X_test)
person_acc = persons_correct / (len(X_test)*0.3897)
dog_acc = dogs_correct / (len(X_test)*0.0787)
bike_acc = bikes_correct / (len(X_test)*0.22058823)
empty_acc = empty_correct / (len(X_test)*0.3897)

print("Total accuracy ", total_acc)
print("Person accuracy ", person_acc)
print("Dog accuracy ", dog_acc)
print("Bike accuracy ", bike_acc)
print("Empty accuracy ", empty_acc)

print("persons correct ", persons_correct)
print("empty correct ", empty_correct)
print("Bike correct ", bikes_correct)
print("test_lenght ", len(X_test))
print("persons correct ", persons_correct)
print("empty correct ", empty_correct)
print("Bike correct ", bikes_correct)
print("test_lenght ", len(X_test))

work_dir = 'C:/Users/eitn35/PycharmProjects/EITN35-Asses/'
#project_directory = '/Users/august/PycharmProjects/EITN35/'
results = work_dir

def round_sig(x, sig=2):
    return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)

def extractRunNo(elem):
    if('Model' in elem):
        return int(elem.split('_')[1].split('.')[0])
    else:
        return 0



def print_confusion():
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)

    # training_results folder
    os.chdir(results)
    current_results = os.listdir(results)

    working_file_2 = 'results_confusionM.txt'
    print("Confusion file found: " + str(working_file_2))

    file = open(working_file_2, "a+")
    file.write("\nConfusion Matrix Run at " + str(current_time))
    file.write("\n")
    for i in range(4):
        for j in range(4):
            file.write(str(confusion[i][j]) + ',')
        file.write("\n")

    file.write("\n")
    file.close()


print_confusion()

print(confusion)






