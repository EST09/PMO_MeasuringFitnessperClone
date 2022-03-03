from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from keras import backend as K
import numpy as np
#import matplotlib.pyplot as plt
import pickle
from tensorflow.keras import layers
from loadData import loadData, loadDataGIF
from tensorflow import keras
import tensorflow as tf
import os
import pandas as pd

#K.set_image_data_format('channels_first')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
#path2Images ='../Identity/ABC_transporter_disorders'
#identity = 'ABC_transporter_disorders'
def create_layers():
    layers = []
    size = 32 
  
    #encoder layers
    for i in range(0, 3):
        x = Conv2D(size, (3, 3), activation='relu', padding='same')
        layers += [x] 
        x = MaxPooling2D((2, 2), padding='same')
        layers += [x]
        size = size // 2
  

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

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def VAE_encoder(w,h,c,l,latent_dim):
    #latent_dim = 2
    encoder_inputs = keras.Input(shape=(w, h, c,l))
    x = layers.Conv3D(32, (3,3,3), activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv3D(64, (3,3,3), activation="relu", strides=2, padding="same")(x)
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
    x = layers.Dense(75 * 75 * 10 * 6, activation="relu")(latent_inputs)
    x = layers.Reshape((75, 75, 10, 6))(x)
    x = layers.Conv3DTranspose(64, (3,3,3), activation="relu", strides=2, padding="same")(x)
    x = layers.Conv3DTranspose(32, (3,3,3), activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv3DTranspose(3, (3,3,3), activation="sigmoid", padding="same")(x)
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

encoder = VAE_encoder(300,300,40,3,3)
decoder = VAE_decoder(3)
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.1 ))
        


path2Identities = '../Identity_gif'
Identities = os.listdir(path2Identities)
for f,identity in enumerate(Identities):
    #print('item {}'.format(f))
    if os.path.exists(os.path.join('../LatentSpaceVAE_3D_gif',identity+'.pickle')):
        #print('item {} already exists'.format(f))    
        continue

    else:
        print('item {} not exists'.format(identity))
        #continue
        path2Images = os.path.join('../Identity',identity) #'../Identity/ABC_transporter_disorders'
        path2GIF = os.path.join(path2Identities,identity)
        #identity = 'ABC_transporter_disorders'

        x_train,x_test = loadData(path2Images,300,300)
        x_trainGIF,x_testGIF = loadDataGIF(path2GIF,300,300)

        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        x_trainGIF = x_trainGIF.astype('float32')/ 255.
        x_testGIF = x_testGIF.astype('float32')/ 255.
        
        print(x_train.shape)
        print(x_test.shape)
        print(x_trainGIF.shape)
        print(x_testGIF.shape) 
    
    data_gen_args = dict(featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=90,
        horizontal_flip=True,
        vertical_flip=True)

    
    datagen = ImageDataGenerator(**data_gen_args)
    datagenGIF = ImageDataGenerator(**data_gen_args)

    print(identity)
    print('item {} is now running {}'.format(f,identity))
    
    x_train = np.expand_dims(x_train,4)
    x_test = np.expand_dims(x_test,4)

    x_train = np.concatenate((x_train,x_trainGIF),axis = 4)
    x_test = np.concatenate((x_test,x_testGIF),axis = 4)
    x_train = np.rollaxis(x_train,4,3)
    x_test = np.rollaxis(x_test,4,3)
    print(x_train.shape)
    print(x_test.shape)

    if (x_train.shape[0] == 0) or (x_test.shape[0] == 0):
        continue

    if not os.path.exists('../modelsVAE_3D_gif'):
        os.makedirs('../modelsVAE_3D_gif')
    history = vae.fit(x_train, batch_size = 4,epochs=50, shuffle=True, validation_data= (x_test,x_test),callbacks=[ModelCheckpoint('../modelsVAE_3D_gif/'+identity+'.h5', monitor='reconstruction_loss', verbose=1, save_best_only=True,save_weights_only=False)], verbose=2)

    z_mean, _, _ = vae.encoder.predict(x_test)
    print(z_mean.shape)
    if not os.path.exists('../LatentSpaceVAE_3D_gif'):
        os.makedirs('../LatentSpaceVAE_3D_gif')

    pickle.dump(z_mean, open('../LatentSpaceVAE_3D_gif/'+identity+'.pickle', 'wb'))
    K.clear_session()

