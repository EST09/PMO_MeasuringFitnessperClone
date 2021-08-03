# Taking Saeed's code and converting it to make it more intuitive and commented 
# Making it run on a clone by clone basis

'''
## Resources
'''

# https://blog.keras.io/building-autoencoders-in-keras.html
# https://keras.io/examples/generative/vae/
# https://github.com/keras-team/keras-io/blob/master/examples/generative/vae.py
# https://learnopencv.com/variational-autoencoder-in-tensorflow/
# https://towardsdatascience.com/generating-fake-fifa-19-football-players-with-variational-autoencoders-and-tensorflow-aff6c10016ae
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

"""
## Create a sampling layer
"""

class Sampling(layers.Layer): 
    """
    Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.
    """

    def call(self, inputs):
        
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

"""
## Build the encoder
"""

def VAE_encoder(w,h,c,latent_dim):

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

"""
## Build the decoder
"""

def VAE_decoder(latent_dim):

    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(75 * 75 * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((75, 75, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)
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
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
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

'''
## Load the Data and Train the VAE
'''

encoder = VAE_encoder(300,300, 3, 2)
decoder = VAE_decoder(2)
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())

        
x_train,x_test = loadData(path2Images,300,300)

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=90,
    horizontal_flip=True,
    vertical_flip=True)

datagen.fit(x_train)

print(x_train.shape)
print(x_test.shape)

traingen = datagen.flow(x_train,batch_size=32)

history = vae.fit(traingen, epochs=10, shuffle=True, validation_data= (x_test, x_test), callbacks=[ModelCheckpoint(os.path.join(current_directory,'../modelsVAE_3D/','model.h5'), monitor='reconstruction_loss', verbose=1, save_best_only=True, save_weights_only=True)], verbose=2)

z_mean, _, _ = vae.encoder.predict(x_test)
print(z_mean.shape)

pickle.dump(z_mean, open(os.path.join(current_directory, '../LatentSpaceVAE_3D/', 'model.pickle'), 'wb'))


"""
## Display loss
"""
# Test doesn't appear to be doing anything

plt.plot(history.history['loss'])
plt.plot(history.history['val_total_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

"""
## Display reconstructed images
"""

image = x_test[0]
print("x_test", x_test[0], x_test[0].shape)
code = encoder.predict(image[None])
print("code", code)
code_t = encoder.predict(image[None])[0]
print("code_t", code_t)


reco_t = decoder.predict(code_t)[0]
print("reco_t", reco_t)








#ValueError: Input 0 is incompatible with layer decoder: expected shape=(None, 2), found shape=(None, 1, 2)
def visualize(img,encoder,decoder):
    """Draws original, encoded and decoded images"""
    # img[None] will have shape of (1, 32, 32, 3) which is the same as the model input
    print("img_None", img[None].shape)
    
    code = encoder.predict(img[None])[0]
    
    print(code[None].shape)

    reco = decoder.predict(code)[0]

    plt.subplot(1,3,1)
    plt.title("Original")
    plt.imshow(img)

    plt.subplot(1,3,2)
    plt.title("Code")
    plt.imshow(code.reshape([code.shape[-1]//2,-1]))

    plt.subplot(1,3,3)
    plt.title("Reconstructed")
    plt.imshow(reco)
    plt.show()

for i in range(5):
    img = x_test[i]
    visualize(img,encoder,decoder)




K.clear_session()