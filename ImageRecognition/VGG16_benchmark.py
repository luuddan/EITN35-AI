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
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import decode_predictions
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

random.shuffle(test_imgs)

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


# Convert list to numpy array
X_test = np.array(X_test)
y_test = np.array(y_test)

# Lets split the data into train and test set
print("Shape of test images is:", X_test.shape)
print("Shape of test labels is:", y_test.shape)


# Change number of layers for correct model saving
no_layers = 16

# Model Setup
model = VGG16()

# Model summary
model.summary()

# Compilation of model
#opt = tf.keras.optimizers.SGD(learning_rate=learning_rate_input) #which optimizer should med used
#model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=opt, metrics=['accuracy'])

os.chdir(work_dir)

print("Came to probability!")
# Probability of test set
#probability_model = tf.keras.Sequential([model,
#                                         tf.keras.layers.Softmax()])

print("Came to predictions! Wheoo!")
predictions = model.predict(X_test)

#for i in range(len(predictions)):
    #print(predictions[i]);

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
# plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
# for i in range(num_images):
#     plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
#     plot_image(i, predictions[i], y_test, X_test)
#     plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
#     plot_value_array(i, predictions[i], y_test)
# plt.tight_layout()
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
    label = decode_predictions(predictions,top=1)

    for j in range(len(label)):
        print(label[j])
    #print("This was the labels " + label)
    #insert code for translation between VGG16 prediction and our counter

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
    file.write("\nTrainable_param," + str(134.276932))
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





