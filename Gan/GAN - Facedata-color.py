#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, RMSprop
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn
from numpy.random import randint
from numpy.random import rand
from numpy import zeros
from numpy import ones
from matplotlib import pyplot
import random
from numpy import vstack
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Reshape


#Input dimension generator
input_dim = 100


# In[2]:



# Optimizer
adam = Adam(lr=0.0002, beta_1=0.5)

def define_discriminator(in_shape=(128,128,3)):
    discriminator = Sequential()
    discriminator.add(Conv2D(64, (3,3), padding='same', data_format="channels_last", strides=(2, 2), input_shape=in_shape , activation=LeakyReLU(alpha=0.2)))
    discriminator.add(Dropout(0.4))
    discriminator.add(Conv2D(64, (3,3),padding='same', data_format="channels_last", strides=(2, 2), activation=LeakyReLU(alpha=0.2)))
    discriminator.add(Dropout(0.4))
    discriminator.add(Conv2D(64, (3,3),padding='same', data_format="channels_last", strides=(2, 2), activation=LeakyReLU(alpha=0.2)))
    discriminator.add(Dropout(0.4))
    discriminator.add(Flatten())
    discriminator.add(Dense(1, activation='sigmoid'))  
    discriminator.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    return discriminator


# In[3]:


discriminator = define_discriminator()
discriminator.summary()


# In[4]:


def define_generator(latent_dim):
    model = Sequential()
    # foundation for 8x8 image
    n_nodes = 128 * 8 * 8
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((8, 8, 128)))
    # upsample to 16x16
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 32x32
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsampling to 64x64
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(3, (7,7), activation='tanh', padding='same'))
    return model


# In[5]:


generator = define_generator(100)
generator.summary()


# In[6]:


#GAN-model
discriminator.trainable = False
inputs = Input(shape=(input_dim, ))
hidden = generator(inputs)
output = discriminator(hidden)
gan = Model(inputs, output)
gan.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
gan.summary()


# In[7]:


# create a data generator
datagen = ImageDataGenerator()

x_train = datagen.flow_from_directory('C:/Users/eitn35/Documents/EITN35/video_files/frames/CycleGan/DayPersons_medium/'
                                       ,class_mode=None, color_mode = 'rgb',target_size=(128, 128), batch_size=38)


# In[8]:



def newImages(x_train=x_train):
    x = next(x_train)
    for i in range(38):
        x[i] = x[i].astype('float32')
        x[i] = x[i] / 127.5 - 1
    return x


# In[9]:


x= newImages()


# In[10]:



# select real samples
def generate_real_samples(x_train, n_samples):
    # choose random instances
    ix = randint(0, x_train.shape[0], n_samples)
    # retrieve selected images
    X = x_train[ix]
    # generate 'real' class labels (1)
    X.reshape(n_samples, 128,128,3)
    y = ones((n_samples, 1))
    return X, y
   
# generate n noise samples with class labels
def generate_noise_samples(n_samples):
    # generate uniform random numbers in [0,1]
    X = rand(16384* 3 * n_samples)
    # reshape into a batch of grayscale images
    X = X.reshape((n_samples, 16384*3))
    X = X.reshape((n_samples, 128, 128, 3))
    # generate 'fake' class labels (0)
    y = zeros((n_samples, 1))
    return X, y

    # generate points in latent space as input for the generator
def generate_latent_points(input_dim, n_samples):
    # generate points in the latent space
    x_input = randn(input_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, input_dim)
    return x_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(model, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    x = model.predict(x_input)
    # create 'fake' class labels (0)
    y = zeros((n_samples, 1))
    return x, y


# In[11]:



def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
    # prepare real samples
    x_real, y_real = generate_real_samples(dataset, n_samples)
    # evaluate discriminator on real examples
    _, acc_real = d_model.evaluate(x_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    # evaluate discriminator on fake examples
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))


# In[12]:


def plot_result(k):
    n_samples = 5
    latent_points = generate_latent_points(input_dim, n_samples)
    x = generator.predict(latent_points)
    x = (x + 1) / 2.0
    for i in range(n_samples):
        pyplot.subplot(2, n_samples, 1  + i)
        pyplot.axis('off')
        pyplot.imshow(x[i])
    # save plot to file
    filename1 = 'generated_plot_%06d.png' % ((k + 1))
    pyplot.savefig(filename1)
    pyplot.close()
    

# save the generator models to file
def save_models(i, g_model_AtoB):
	# save the first generator model
	filename1 = 'g_model%06d.h5' % (i+1)
	g_model_AtoB.save(filename1)
	# save the second generator model
	print('>Saved: %s' % (filename1))


# In[13]:


#Train discriminator
def train_discriminator(model, number_iteration, batch_size, x_train):

    half_batch = int(batch_size/2)
    for i in range(number_iteration):
        #x_train = newImages()
        #Sample from real images
        x_real, y_real = generate_real_samples(x_train, half_batch)
        #Sample from noise
        x_noise, y_noise = generate_noise_samples((half_batch))
        #Train with real images
        _, real_acc = model.train_on_batch(x_real, y_real)
        #Train with noise
        _, noise_acc = model.train_on_batch(x_noise, y_noise)
        print('>%d real=%.0f%% noise=%.0f%%' % (i+1, real_acc*100, noise_acc*100))


# In[14]:


def train_gan(generator, discriminator, gan, input_dim, n_epochs, batch_size):
    x_train = newImages()
    half_batch = int(batch_size/2)
    batch_per_epoch = int(136/x_train.shape[0])
    for i in range(n_epochs):
        if(i%20 == 0):
            plot_result(i)
            save_models(i, generator)
        x_train = datagen.flow_from_directory('C:/Users/eitn35/Documents/EITN35/video_files/frames/CycleGan/DayPersons_medium/'
                                       ,class_mode=None, color_mode = 'rgb',target_size=(128, 128), batch_size=38)
        for k in range(batch_per_epoch):
            x_train = newImages()
            x_real, y_real = generate_real_samples(x_train, half_batch)
            x_fake, y_fake = generate_fake_samples(generator, input_dim, half_batch)
            x, y = vstack((x_real, x_fake)), vstack((y_real, y_fake))
            discriminator_loss = discriminator.train_on_batch(x, y)
            x_gan = generate_latent_points(input_dim, batch_size)
            y_gan = ones((batch_size, 1))
            generator_loss = gan.train_on_batch(x_gan, y_gan)
            print('>%d, d=%.3f, g=%.3f' % (i+1, discriminator_loss[0], generator_loss[0]))
            #if (j==1):
            #summarize_performance(i, generator, discriminator, x_train, input_dim)


# In[15]:


x_train = newImages()
train_discriminator(discriminator, 25, 34, x)

#15ggr


# In[16]:


train_gan(generator,discriminator,gan,input_dim,5000, 34)


# In[19]:





