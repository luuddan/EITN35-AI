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
from keras_applications.vgg16 import decode_predictions
from tensorflow.keras import models, layers, regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout
import sys
import math
import time
import getopt

# Model Setup
model = VGG16()

# Model summary
model.summary()