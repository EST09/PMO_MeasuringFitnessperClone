'''
## Description

'''

# Commenting Saeed's code and keeping important parts
# Making it run on a clone by clone basis

'''
## Resources
'''

# https://blog.keras.io/building-autoencoders-in-keras.html
# https://keras.io/examples/generative/vae/
# https://github.com/keras-team/keras-io/blob/master/examples/generative/vae.py
# https://learnopencv.com/variational-autoencoder-in-tensorflow/
# https://towardsdatascience.com/generating-fake-fifa-19-football-players-with-variational-autoencoders-and-tensorflow-aff6c10016ae
# https://github.com/mmeendez8/Fifa
# https://towardsdatascience.com/variational-autoencoders-vaes-for-dummies-step-by-step-tutorial-69e6d1c9d8e9
# https://casser.io/autoencoder/
# https://stackabuse.com/autoencoders-for-image-reconstruction-in-python-and-keras
# https://github.com/AKASH2907/Introduction_to_Deep_Learning_Coursera/blob/master/Week_4_PA1/Autoencoders_task.ipynb


'''
## Dependancies
'''

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from os.path import expanduser
import pandas as pd

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import backend as K

from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

from loadData import loadData

'''
## File paths
'''

current_directory = os.path.dirname(os.path.realpath(__file__))
home = expanduser("~")
path2Images = home + "/smaller_test_imgs" # the file that makes these is miscellaneous.py

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" #not sure what this is
#os.environ["CUDA_VISIBLE_DEVICES"] = "5" 

'''
# Going to try to deconstruct code
'''

'''
# X_input
# https://machinelearningmastery.com/image-augmentation-deep-learning-keras/
'''

## Get Images

# takes in path2images which is a folder with 100 pathways 
# with width:300, height: 300
# uses cv2 to read in creating 3 color channels (300, 300, 3)
# splits into 70:30 but not randomly so far 
# writes image names of validation set to a csv to be used as labels later
x_train,x_test = loadData(path2Images,300,300)

# converts to range 0-1 
# typically used to help optimiser e.g. sigmoid works better between 0-1
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

## Augment Images

# generates batches of tensor image data with real time data augmentation 
# ImageDataGenerator is a class which is designed to be appliec to your data in real time
datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=90,
    horizontal_flip=True,
    vertical_flip=True)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
# essentially here we are fitting our ImagedataGenerator class to our actual data
datagen.fit(x_train)

# print(x_train.shape) # (71, 300, 300, 3)
# print(x_test.shape) # (30, 300, 300, 3)

# returns batches of images when requested
traingen = datagen.flow(x_train, batch_size=32)

# Question for Saeed: do we need to call fit_generator? 
# Question for Saeed: do we need to iterate through images or is traingen a collection of batches now?

'''
# Create Sampling Layer
'''

'''
# Keras Terminology: 

# Layers.layer: A layer encapsulates both a state (the layer's "weights") 
#              and a transformation from inputs to outputs (a "call", the layer's forward pass).
# keras.Model: The outer container, the thing you want to train, is a Model. A Model is just like a Layer, 
#            but with added training and serialization utilities.

'''

class Sampling(layers.Layer): 
    """
    Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.
    """

    # using def call allows us to simply type Sampling()([inputs]) and output of call will be returned
    def call(self, inputs):
        
        # these will be two dense layers from the encoder
        # dimensionality = latent_dim 
        # input is rest of encoder layers
        z_mean, z_log_var = inputs
        # tf.shape returns a tensor containing the shape of the input tensor
        
        batch = tf.shape(z_mean)[0]  # batch 32     
        dim = tf.shape(z_mean)[1] # dim 2
        
        # this creates an epsilson (noise samples) 
        # tensor with normal distribution of values
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        
        # https://stats.stackexchange.com/questions/486158/reparameterization-trick-in-vaes-how-should-we-do-this 
        # constaining to positive numbers
        # var^0.5 = sd (so (z = mu + eps * sd) == (z = mu + eps * exp(0.5log(var))))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon # reaparmeterisation trick (question for aden?)


'''
# Create encoder
'''

def VAE_encoder(w,h,c,latent_dim):

    # instantiate tensor
    encoder_inputs = keras.Input(shape=(w, h, c)) # (None, 300, 300, 3) where None is just no. of images unknown
    # Question for Saeed: do I need to provide an input shape? the blog didn't?

    # This specifies 32 convolution filters/kernels
    # kernel is 3X3 (single integer specifies same value in all spatial dimensions)
    # padding same ensures output has same h, w dimension as input
    # Question for Saeed: padding ensures same output as input but 300.300.3 != 150.150.32? 
    # Question for Saeed: How do we choose how many kernels to use? Does this affect the dimensions of the final result? 
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs) # (None, 150, 150, 32)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x) # (None, 75, 75, 64)
    # flatten the input 
    x = layers.Flatten()(x) # (None, 360000) = 75.75.64
    # regularly connected NN layer
    # 16 = dimensionality of output space
    x = layers.Dense(16, activation="relu")(x)

    #Question for Saeed: I don't really understand why this layer is z_mean and the other layer is z_log_var? They should be the same no?
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    
    z = Sampling()([z_mean, z_log_var])
    
    # Model groups layers into an object with training and inference features.
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    # Prints a string summary of the network.
    encoder.summary()

    return encoder


"""
## Build the decoder
"""

def VAE_decoder(latent_dim):

    latent_inputs = keras.Input(shape=(latent_dim,))
    # specify dimensionality of output space
    x = layers.Dense(75 * 75 * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((75, 75, 64))(x)
    # deconvolution
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x) # (None, 300, 300, 3)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()

    return decoder


"""
## Define the VAE as a `Model` with a custom `train_step`
"""

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        # Computes the (weighted) mean of the given values.
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    # allows backwards compatibility
    # https://www.programiz.com/python-programming/property
    # Question for Saeed: Still not massively sure about decorators?
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        # https://keras.io/guides/customizing_what_happens_in_fit/
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`
        # train_step is an expected customisable function
        # override the training step function of the Model class

        # Record operations for automatic differentiation.
        with tf.GradientTape() as tape:           
            
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            # reconstruction loss

            # Question for Saeed: My data isn't binary - am I okay to replace binary cross entropy?
            # reconstruction_loss = tf.reduce_mean(
            #     tf.reduce_sum(
            #         keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
            #     )
            # )
            # https://keras.io/api/losses/
            # Question for Saeed: can I do this? How do I learn more about this?
            mse = tf.keras.losses.MeanSquaredError()        
            reconstruction_loss = mse(data, reconstruction)

            # kl_loss
            # a measure of how one probability distribution is different from a second
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            # Question for Saeed: why do we reduce sum and mean?
            # Computes the mean of elements across dimensions of a tensor.
            # axis = 1 mean computed horizontally
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        
            # total_loss
            total_loss = reconstruction_loss + kl_loss

        # computes the gradient
        grads = tape.gradient(total_loss, self.trainable_weights)
        # Ask the optimizer to apply the processed gradients
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
        
    def call(self, data):
        
        [z_mean,z_log_var,z]  = self.encoder(data)
        y_pred = self.decoder(z)
        return y_pred

encoder = VAE_encoder(300,300, 3, 2)
decoder = VAE_decoder(2)
vae = VAE(encoder, decoder)

# Configures the model for training
# Adam optimization is a stochastic gradient descent method that is 
# based on adaptive estimation of first-order and second-order moments.
# default lr = 0.001
vae.compile(optimizer=keras.optimizers.Adam())

# Trains the model for a fixed number of epochs (iterations on a dataset).
# verbose is how much you want to see whilst training
# Callback to save the Keras model or model weights at some frequency.
# Question for Saeed: won't work if I set save weights only as false? Does this matter?
history = vae.fit(traingen, epochs=100, shuffle=True, validation_data= (x_test, x_test), callbacks=[ModelCheckpoint(os.path.join(current_directory,'../modelsVAE_3D/','model.h5'), monitor='reconstruction_loss', verbose=1, save_best_only=True, save_weights_only=True)], verbose=2)

z_mean, _, _ = vae.encoder.predict(x_test)

pickle.dump(z_mean, open(os.path.join(current_directory, '../LatentSpaceVAE_3D/', 'model.pickle'), 'wb'))

"""
## Display loss
"""

# Test doesn't appear to be doing anything
# Only train is plotting anything
# Question for Saeed: I don't really understand why? 
# Is it because I don't specify validation split?

plt.plot(history.history['loss'])
plt.plot(history.history['val_total_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# show original and reconstructed image

'''
# Visualisation
'''

def visualize(img,encoder,decoder):
    """Draws original, encoded and decoded images"""
    # img[None] will have shape of (1, 300, 300, 3) which is the same as the model input
    print("img", img.shape)
    
    # reference the first of the batch
    # encoder.predict expects images in teh form of batches 
    # even tho here we are doing one image at a time 
    # hence we much reference the first of None 
    # where None is the undefined shape of the batch size
    code = encoder.predict(img[None])[0]
    
    print("code", code[None].shape)
    # Input 0 is incompatible with layer decoder: expected shape=(None, 2), found shape=(None, 1, 2)
    # Question for Saeed: Do you think I'm referencing the right thing here? 
    # Not sure why I get None, 1, 2?
    reco = decoder.predict(code)[0]

    print("reco", reco.shape)

    plt.subplot(1,3,1)
    plt.title("Original")
    plt.imshow(img)

    plt.subplot(1,3,2)
    plt.title("Code")
    # not really sure what this is doing - probably need to change to make more informative
    plt.imshow(code.reshape([code.shape[-1]//2,-1]))

    plt.subplot(1,3,3)
    plt.title("Reconstructed")
    plt.imshow(reco)
    plt.show()

for i in range(5):
    img = x_test[i]
    visualize(img,encoder,decoder)

#plot latent space
#need to add labels


z_mean, _, _ = encoder.predict(x_test)
plt.figure(figsize=(6, 6))

#plot z_mean
# Question for Saeed: I don't really understand why I only have 30 means?
print(z_mean[:, 0])
plt.scatter(z_mean[:, 0], z_mean[:, 1])
plt.show()


# Display a 2D manifold of the images

n = 10  # figure with 10x10 images
digit_size = 300
figure = np.zeros((digit_size * n, digit_size * n, 3))
# We will sample n points within [10, 10] standard deviations
grid_x = np.linspace(0, 10, n)
grid_y = np.linspace(0, 10, n)

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded[0].reshape((digit_size, digit_size, 3))
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size, 
               :3] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure)
plt.show()


K.clear_session()


#######################################################

# """
# ## Create a sampling layer
# """

# '''
# # Keras Terminology: 

# # Layers.layer: A layer encapsulates both a state (the layer's "weights") 
# #              and a transformation from inputs to outputs (a "call", the layer's forward pass).
# # keras.Model: The outer container, the thing you want to train, is a Model. A Model is just like a Layer, 
# #            but with added training and serialization utilities.

# '''

# class Sampling(layers.Layer): 
#     """
#     Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.
#     """

#     def call(self, inputs):
        
#         z_mean, z_log_var = inputs
#         batch = tf.shape(z_mean)[0]
#         dim = tf.shape(z_mean)[1]
#         epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        
#         return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# """
# ## Build the encoder

# # keras.Input
# # instantiate a keras tensor
# # shape: A shape tuple (integers), not including the batch size. 
# # For instance, shape=(32,) indicates that the expected input will be batches of 32-dimensional vectors. 
# # Elements of this tuple can be None; 'None' elements represent dimensions where the shape is not known.

# # layers.Conv2D

# """

# def VAE_encoder(w,h,c,latent_dim):

#     encoder_inputs = keras.Input(shape=(w, h, c))
#     x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
#     x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
#     x = layers.Flatten()(x)
#     x = layers.Dense(16, activation="relu")(x)
#     z_mean = layers.Dense(latent_dim, name="z_mean")(x)
#     z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
#     z = Sampling()([z_mean, z_log_var])
#     encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
#     encoder.summary()

#     return encoder

# """
# ## Build the decoder
# """

# def VAE_decoder(latent_dim):

#     latent_inputs = keras.Input(shape=(latent_dim,))
#     x = layers.Dense(75 * 75 * 64, activation="relu")(latent_inputs)
#     x = layers.Reshape((75, 75, 64))(x)
#     x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
#     x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
#     decoder_outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)
#     decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
#     decoder.summary()

#     return decoder

# """
# ## Define the VAE as a `Model` with a custom `train_step`
# """

# class VAE(keras.Model):
#     def __init__(self, encoder, decoder, **kwargs):
#         super(VAE, self).__init__(**kwargs)
#         self.encoder = encoder
#         self.decoder = decoder
#         self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
#         self.reconstruction_loss_tracker = keras.metrics.Mean(
#             name="reconstruction_loss"
#         )
#         self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

#     @property
#     def metrics(self):
#         return [
#             self.total_loss_tracker,
#             self.reconstruction_loss_tracker,
#             self.kl_loss_tracker,
#         ]

#     def train_step(self, data):
#         with tf.GradientTape() as tape:
#             z_mean, z_log_var, z = self.encoder(data)
#             reconstruction = self.decoder(z)
#             reconstruction_loss = tf.reduce_mean(
#                 tf.reduce_sum(
#                     keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
#                 )
#             )
#             kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
#             kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
#             total_loss = reconstruction_loss + kl_loss
#         grads = tape.gradient(total_loss, self.trainable_weights)
#         self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
#         self.total_loss_tracker.update_state(total_loss)
#         self.reconstruction_loss_tracker.update_state(reconstruction_loss)
#         self.kl_loss_tracker.update_state(kl_loss)
#         return {
#             "loss": self.total_loss_tracker.result(),
#             "reconstruction_loss": self.reconstruction_loss_tracker.result(),
#             "kl_loss": self.kl_loss_tracker.result(),
#         }
        
#     def call(self, data):
#         [z_mean,z_log_var,z]  = self.encoder(data)
#         y_pred = self.decoder(z)
#         return y_pred

# '''
# ## Load the Data and Train the VAE
# '''

# encoder = VAE_encoder(300,300, 3, 2)
# decoder = VAE_decoder(2)
# vae = VAE(encoder, decoder)
# vae.compile(optimizer=keras.optimizers.Adam())

        
# x_train,x_test = loadData(path2Images,300,300)

# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.

# datagen = ImageDataGenerator(
#     featurewise_center=False,
#     featurewise_std_normalization=False,
#     rotation_range=90,
#     horizontal_flip=True,
#     vertical_flip=True)

# datagen.fit(x_train)

# print(x_train.shape)
# print(x_test.shape)

# traingen = datagen.flow(x_train,batch_size=32)

# history = vae.fit(traingen, epochs=10, shuffle=True, validation_data= (x_test, x_test), callbacks=[ModelCheckpoint(os.path.join(current_directory,'../modelsVAE_3D/','model.h5'), monitor='reconstruction_loss', verbose=1, save_best_only=True, save_weights_only=True)], verbose=2)

# z_mean, _, _ = vae.encoder.predict(x_test)
# print(z_mean.shape)

# pickle.dump(z_mean, open(os.path.join(current_directory, '../LatentSpaceVAE_3D/', 'model.pickle'), 'wb'))


# """
# ## Display loss
# """
# # Test doesn't appear to be doing anything

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_total_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# """
# ## Display reconstructed images
# """

# # show original and reconstructed image

# # def visualize(img,encoder,decoder):
# #     """Draws original, encoded and decoded images"""
# #     # img[None] will have shape of (1, 32, 32, 3) which is the same as the model input
# #     print("img_None", img[None].shape)
    
# #     code = encoder.predict(img[None])[0]
    
# #     print(code[None].shape)

# #     reco = decoder.predict(code)[0]

# #     plt.subplot(1,3,1)
# #     plt.title("Original")
# #     plt.imshow(img)

# #     plt.subplot(1,3,2)
# #     plt.title("Code")
# #     plt.imshow(code.reshape([code.shape[-1]//2,-1]))

# #     plt.subplot(1,3,3)
# #     plt.title("Reconstructed")
# #     plt.imshow(reco)
# #     plt.show()

# # for i in range(5):
# #     img = x_test[i]
# #     visualize(img,encoder,decoder)

# #plot latent space
# #need to add labels

# # x_test_encoded = encoder.predict(x_test, batch_size=32)
# # plt.figure(figsize=(6, 6))
# # #plot z_mean
# # plt.scatter(x_test_encoded[0][:, 0], x_test_encoded[0][:, 1])
# # plt.show()

# #Not working for colour yet

# # # Display a 2D manifold of the digits
# # n = 10  # figure with 15x15 digits
# # digit_size = 300
# # figure = np.zeros((digit_size * n, digit_size * n, 3))
# # # We will sample n points within [-15, 15] standard deviations
# # grid_x = np.linspace(0, 10, n)
# # grid_y = np.linspace(0, 10, n)

# # for i, yi in enumerate(grid_x):
# #     for j, xi in enumerate(grid_y):
# #         z_sample = np.array([[xi, yi]])
# #         x_decoded = decoder.predict(z_sample)
# #         digit = x_decoded[0].reshape((digit_size, digit_size, 3))
# #         figure[i * digit_size: (i + 1) * digit_size,
# #                j * digit_size: (j + 1) * digit_size, 
# #                :3] = digit

# # plt.figure(figsize=(10, 10))
# # plt.imshow(figure)
# # plt.show()


# K.clear_session()



#########

# To show that my image isn't binary and can't use cross entropy

# plt.imshow(x_train[0])
# plt.show()
# image = x_train[0]

# # tuple to select colors of each channel line
# colors = ("red", "green", "blue")
# channel_ids = (0, 1, 2)

# # create the histogram plot, with three lines, one for
# # each color
# plt.xlim([0, 0.9])
# for channel_id, c in zip(channel_ids, colors):
#     histogram, bin_edges = np.histogram(
#         image[:, :, channel_id], bins=100, range=(0, 0.9)
#     )
#     plt.plot(bin_edges[0:-1], histogram, color=c)

# plt.xlabel("Color value")
# plt.ylabel("Pixels")

# plt.show()