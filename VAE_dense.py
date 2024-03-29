from gc import callbacks
from tabnanny import verbose
import os
import numpy as np
import pickle
import tensorflow as tf

from tensorflow import keras
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils

from itertools import chain

from config import Config
from data_prep import DataPrep, Vec2Img

"""
Variatonal Autoencoder with Dense layers. Creates: Encoder, Decoder, Sampling layer.

 - Reads data using DataPrep class
 - Dimensions of inputs, outputs, layer neurons given from Config class
 - As long as layers are regular Dense layers, users can change their size/number from Config

All models (encoder, decoder, full vae) are saved in /saved_models subdirectory
"""

def loss_func(encoder_mu, encoder_log_variance):
    def vae_reconstruction_loss(y_true, y_predict):
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.binary_crossentropy(y_true, y_predict), axis=-1)
                )
        return reconstruction_loss

    def vae_kl_loss(encoder_mu, encoder_log_variance):
        kl_loss = -.5 * (1 + encoder_log_variance - tf.square(encoder_mu) - tf.exp(encoder_log_variance))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=-1))
        return kl_loss

    def vae_loss(y_true, y_predict):
        reconstruction_loss = vae_reconstruction_loss(y_true, y_predict)
        kl_loss = vae_kl_loss(y_true, y_predict)

        loss = reconstruction_loss + kl_loss
        return loss

    return vae_loss

def sampling(mu_log_variance):
    x_mean, x_logvar = mu_log_variance
    batch = tf.shape(x_mean)[0]
    dim = tf.shape(x_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))

    return x_mean + tf.exp(0.5 * x_logvar) * epsilon


if __name__ == "__main__":

    checkpoint_path = 'saved_models/training/cp.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True,verbose=1)

    const = Config()
    neurons = const.VAE_neurons

    # Data loading and preparation
    data = DataPrep(file = const.file_path,
                    num_settings = const.num_settings,
                    num_sensors = const.num_sensors,
                    num_units = const.num_units[0],
                    prev_step_units = const.prev_step_units[0],
                    step = const.step[0],
                    normalization_type="01")

    df = data.ReadData()
    #print(df.shape)

    # Build encoder
    latent_dim = const.latent_dim

    encoder_inputs = tf.keras.Input(shape=(const.image_size,))
    x = encoder_inputs

    for i in range(len(neurons)):
        if i == 0:
            x = tf.keras.layers.Dense(neurons[i], activation='relu', name="dense_layer_" + str(i))(x)
        elif (i > 0) & (i <= len(neurons)):
            x = tf.keras.layers.Dense(neurons[i], activation='relu', name="dense_layer_" + str(i))(x)
        else:
            x = tf.keras.layers.Dense(neurons[i], activation='relu', name="dense_layer_final")(x)

    x_mean = tf.keras.layers.Dense(const.latent_dim, activation='sigmoid', name="x_mean")(x)
    x_logvar = tf.keras.layers.Dense(const.latent_dim, name="x_logvar")(x)
    
    mean_logvar_model = tf.keras.models.Model(x, (x_mean, x_logvar), name="mean_logvar_model")

    encoder_output = tf.keras.layers.Lambda(sampling, name="encoder_output")([x_mean, x_logvar])

    encoder = tf.keras.models.Model(encoder_inputs, encoder_output, name="encoder")
    encoder.summary()
    
    # Build decoder
    latent_inputs = tf.keras.Input(shape=(const.latent_dim,))
    x = latent_inputs

    for i in range(len(neurons)-1,-1,-1):
        if i == len(neurons) - 1:
            x = tf.keras.layers.Dense(neurons[i], activation='relu', name="latent_layer")(x)
        else:
            x = tf.keras.layers.Dense(neurons[i], activation='relu', name="dense_layer_" + str(i))(x)
    decoder_outputs = tf.keras.layers.Dense(const.image_size, activation="sigmoid", name="decoder_output")(x)

    decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()
    
    # Build VAE
    vae_input = tf.keras.layers.Input(shape=(const.image_size,), name="VAE_input")
    vae_encoder_output = encoder(vae_input)
    vae_decoder_output = decoder(vae_encoder_output)
    vae = tf.keras.models.Model(vae_input, vae_decoder_output, name="VAE")
    vae.summary()
    
    # Train VAE
    x_train = df[list(chain(*[['NormTime'], data.setting_measurement_names]))]
    #print(x_train)
    #x_train = x_train[1:].expanding(axis=0).mean()
    #print(x_train)
    
    vae.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005), loss=loss_func(x_mean, x_logvar))
    vae.fit(x_train, x_train,\
        epochs=30,
        batch_size=64, 
        callbacks=[cp_callback]
    )
    

    # Save models (decoder and/or encoder)

    encoder.save('./saved_models/encoder')
    decoder.save('./saved_models/decoder')

    vae.save('./saved_models/vae')
    """
    with open('decoder.pkl', 'wb') as f:
        pickle.dump(decoder, f)
    vae.build((None,) + (const.image_size,))
    vae.save('./saved_models/model', save_format='tf')
    with open('model.pkl', 'wb') as f:
        pickle.dump(vae, f)
    """