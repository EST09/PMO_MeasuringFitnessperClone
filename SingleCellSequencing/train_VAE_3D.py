#https://blog.keras.io/building-autoencoders-in-keras.html

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

current_directory = os.path.dirname(os.path.realpath(__file__))
#path2Identities = os.path.join(current_directory, "../Identity")
#going to hard code path to SNU-668 because I don't want it uploaded to google drive
home = expanduser("~")
path2Identities = home + "/smaller_test_imgs"


#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" #not sure what this is
#os.environ["CUDA_VISIBLE_DEVICES"] = "5" 


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


#I'm not sure what layers.Layer is here? 

class Sampling(layers.Layer): 
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# How do you choose all these layers and strides?

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

encoder = VAE_encoder(300,300,3,3)
decoder = VAE_decoder(3)
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
        
# To train it, use the original MNIST digits with shape (samples, 3, 28, 28),
# and just normalize pixel values between 0 and 1

#path2Identities = '../Identity'
#path2Identities = os.path.join(current_directory, "../Identity")
Identities = os.listdir(path2Identities)
for f,identity in enumerate(Identities):
    #print('item {}'.format(f))
    if os.path.exists(os.path.join(current_directory, '../LatentSpaceVAE_3D',identity+'.pickle')):
        #print('item {} already exists'.format(f))    
        continue

    else:
        print('item {} not exists'.format(identity))
        #continue
        path2Images = os.path.join(path2Identities,identity) #'../Identity/ABC_transporter_disorders'
        #identity = 'ABC_transporter_disorders'

        x_train,x_test = loadData(path2Identities,300,300)

        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.

    datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=90,
        horizontal_flip=True,
        vertical_flip=True)

    datagen.fit(x_train)
    #x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))    # adapt this if using 'channels_first' image data format
    #x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))       # adapt this if using 'channels_first' image data format

    # open a terminal and start TensorBoard to read logs in the autoencoder subdirectory
    # tensorboard --logdir=autoencoder

    #vae.fit(mnist_digits, epochs=2, batch_size=128)
    #autoencoder,encoder,decoder = getAutoencoder(300,300,3)
    #print(vae.summary())
    print(identity)
    print('item {} is now running {}'.format(f,identity))
    
    print(x_train.shape)
    print(x_test.shape)
    if (x_train.shape[0] == 0) or (x_test.shape[0] == 0):
        continue
    #history = autoencoder.fit(x_train, x_train, epochs=100, batch_size=32, shuffle=True, validation_data=(x_test, x_test,callbacks=[ModelCheckpoint('../models/'+identity+'.h5', monitor='val_loss', verbose=0, save_best_only=False,save_weights_only=False)], verbose=2)
    #encoder = VAE_encoder(300,300,3,2)
    #decoder = VAE_decoder(2)
    #vae = VAE(encoder, decoder)
    #vae.compile(optimizer=keras.optimizers.Adam())
    #vae.fit(x_train,epochs=1, batch_size = 1)
    #vae.fit_generator(datagen.flow(x_train, x_train,batch_size=32), epochs=2)
    traingen = datagen.flow(x_train,batch_size=32)
    #print(traingen)
    history = vae.fit(traingen, epochs=10, shuffle=True, validation_data= (x_test,x_test),callbacks=[ModelCheckpoint(os.path.join(current_directory,'../modelsVAE_3D/',identity+'.h5'), monitor='reconstruction_loss', verbose=1, save_best_only=True,save_weights_only=True)], verbose=2)

    # take a look at the reconstructed digits
    #decoded_imgs = vae.decoder.predict(x_test)
    #print(decoded_imgs.shape)
    '''
    n = 10
    plt.figure(figsize=(10, 4), dpi=100)
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.set_axis_off()

        # display reconstruction
        ax = plt.subplot(2, n, i + n + 1)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.set_axis_off()

    plt.save('reconstruction.png')
    '''
    # take a look at the 128-dimensional encoded representation
    # these representations are 8x4x4, so we reshape them to 4x32 in order to be able to display them as grayscale images

    #encoder = Model(input_img, encoded)
    z_mean, _, _ = vae.encoder.predict(x_test)
    print(z_mean.shape)
    
    pickle.dump(z_mean, open(os.path.join(current_directory, '../LatentSpaceVAE_3D/', identity+'.pickle'), 'wb'))
    '''
    n = 10
    plt.figure(figsize=(10, 4), dpi=100)
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
        plt.imshow(encoded_imgs[i].reshape(4, 4 * 8).T)
        plt.gray()
        ax.set_axis_off()

    plt.savefig('latentSpace.pnd')
    '''
    K.clear_session()

