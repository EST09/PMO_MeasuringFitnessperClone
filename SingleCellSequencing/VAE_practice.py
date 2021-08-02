#https://blog.keras.io/building-autoencoders-in-keras.html

#Today two interesting practical applications of autoencoders are data denoising (which we feature later in this post), 
# and dimensionality reduction for data visualization.

# We'll start simple, with a single fully-connected neural layer as encoder and as decoder:

import keras
from keras import layers
from keras import regularizers
from keras.preprocessing.image import load_img, img_to_array

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from os.path import expanduser
from shutil import copyfile

#this just creates a test dataset to work with

# home = expanduser("~")
# path22D = home + "/2D-SNU-668"

# for image in os.listdir(os.path.join(path22D, "Clone_0.00281812_ID107695")): 
#     if image.endswith(".png"):
#         copyfile(os.path.join(path22D, "Clone_0.00281812_ID107695", image), os.path.join(home, "test_images", image) )

img = load_img("/Users/esthomas/test_images/ABC-family_proteins_mediated_transport.png")
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
print(img_array)
print(img_array.shape)


# # This is the size of our encoded representations
# encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# # This is our input tensor

# '''
# # In Keras, the input layer itself is not a layer, but a tensor. 
# # It's the starting tensor you send to the first hidden layer. 
# # This tensor must have the same shape as your training data.
# # Example: if you have 30 images of 50x50 pixels in RGB (3 channels), 
# # the shape of your input data is (30,50,50,3). 
# # Then your input layer tensor, must have this shape (see details in the "shapes in keras" section)
# '''
# input_img = keras.Input(shape=(784,))
# # "encoded" is the encoded representation of the input
# encoded = layers.Dense(encoding_dim, activation='relu',
#                                     activity_regularizer=regularizers.l1(10e-5))(input_img)
# # "decoded" is the lossy reconstruction of the input
# decoded = layers.Dense(784, activation='sigmoid')(encoded)
# # This model maps an input to its reconstruction
# autoencoder = keras.Model(input_img, decoded)

# ##Let's also create a separate encoder model:

# # This model maps an input to its encoded representation
# encoder = keras.Model(input_img, encoded)

# ##As well as the decoder model:

# # This is our encoded (32-dimensional) input
# encoded_input = keras.Input(shape=(encoding_dim,))
# # Retrieve the last layer of the autoencoder model
# decoder_layer = autoencoder.layers[-1]
# # Create the decoder model
# decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

# ##Now let's train our autoencoder to reconstruct MNIST digits.
# ##First, we'll configure our model to use a per-pixel binary crossentropy loss, and the Adam optimizer:

# autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# ##Let's prepare our input data. 
# ##We're using MNIST digits, and we're discarding the labels 
# ##(since we're only interested in encoding/decoding the input images).

# from keras.datasets import mnist
# import numpy as np
# (x_train, _), (x_test, _) = mnist.load_data()

# #We will normalize all values between 0 and 1 and we will flatten the 28x28 images into vectors of size 784.

# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
# print(x_train.shape)
# print(x_test.shape)

# #Now let's train our autoencoder for 50 epochs:

# autoencoder.fit(x_train, x_train,
#                 epochs=100,
#                 batch_size=256,
#                 shuffle=True,
#                 validation_data=(x_test, x_test))

# #After 50 epochs, the autoencoder seems to reach a stable train/validation loss value of about 0.09. 
# #We can try to visualize the reconstructed inputs and the encoded representations. We will use Matplotlib.



# # Encode and decode some digits
# # Note that we take them from the *test* set

# encoded_imgs = encoder.predict(x_test)
# decoded_imgs = decoder.predict(encoded_imgs)

# # Use Matplotlib (don't ask)
# import matplotlib.pyplot as plt

# n = 10  # How many digits we will display
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     # Display original
#     ax = plt.subplot(2, n, i + 1)
#     plt.imshow(x_test[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)

#     # Display reconstruction
#     ax = plt.subplot(2, n, i + 1 + n)
#     plt.imshow(decoded_imgs[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()


