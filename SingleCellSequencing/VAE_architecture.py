#Dependancies

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras import layers
from loadData import loadData
from tensorflow import keras
import tensorflow as tf
import os
from os.path import expanduser
import pandas as pd




def create_layers():
    layers = []
    size = 32 
  
    # Q: How do we decide the layers? Conv2D, padding, MaxPooling2D, Upsampling 

    #encoder layers
    for i in range(0, 3):
        x = Conv2D(size, (3, 3), activation='relu', padding='same')
        layers += [x] 
        x = MaxPooling2D((2, 2), padding='same')
        layers += [x]
        size = size // 2 #rounds to nearest whole number
  

    #deocder layers 
    for i in range(0, 3):

        size = size * 2
        if i == 2:
            x = Conv2D(size, (3, 3), activation='relu')
        else:
            x = Conv2D(size, (3, 3), activation='relu', padding='same')

        layers += [x]
        x = UpSampling2D((2, 2))
        layers += [x]
    
    
    x = Conv2D(3, (3, 3), activation='sigmoid', padding='same')
    layers += [x]
  
    return layers


def getAutoencoder(c,w,h):

    input_img = Input(shape=(c, w, h))  

    layers = create_layers()

    #create the auto encoder network 
    x = input_img

    for layer in layers:
        x = layer(x)
    
    autoencoder = Model(input_img, x)
    autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy')
  
    #create the encoder network
    x = input_img

    for layer in layers[0:6]:
        x = layer(x)
    
    encoder = Model(input_img, x)
  
    #create the decoder network
    input_encoded = Input(shape = (4, 4, 8))
    
    x = input_encoded

    for layer in layers[6:]:
        x = layer(x)

    decoder = Model(input_encoded, x)

    return autoencoder, encoder, decoder 

def VAE_encoder(w,h,c,latent_dim):

    #latent_dim = 2
    encoder_inputs = keras.Input(shape=(w, h, c))
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()
    return encoder

def VAE_decoder(latent_dim):
    #latent_dim = 2
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(75 * 75 * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((75, 75, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()
    return decoder
