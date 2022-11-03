from gc import callbacks
from tabnanny import verbose
import os
import numpy as np
import pickle
import sys
import tensorflow as tf

from tensorflow import keras
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils

from itertools import chain

sys.path.insert(1, '../')

from config import Config
from data_prep import DataPrep, Vec2Img




class VAE_LSTM(object):

    def __init__(self,
                 df,
                 image_size,
                 timesteps,
                 latent_dim,
                 neurons):

        super().__init__()

        self.df = df
        self.image_size = image_size
        self.timesteps = timesteps
        self.latent_dim = latent_dim
        self.neurons = neurons

    
    def models(self):
        # Build encoder
        
        def sampling(args):
            x_mean, x_logvar = args
            batch = tf.shape(x_mean)[0]
            dim = tf.shape(x_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))

            return x_mean + tf.exp(0.5 * x_logvar) * epsilon

        encoder_inputs = tf.keras.Input(shape=(self.timesteps, self.image_size,))

        x = tf.keras.layers.LSTM(self.neurons[0], name="LSTM_encoder_layer")(encoder_inputs)
        x = tf.keras.layers.Dense(self.neurons[1], name="dense_layer")(x)
        self.x_mean = tf.keras.layers.Dense(self.latent_dim, activation='sigmoid', name="x_mean")(x)
        self.x_logvar = tf.keras.layers.Dense(self.latent_dim, name="x_logvar")(x)
        
        encoder_output = tf.keras.layers.Lambda(sampling, name="encoder_output")([self.x_mean, self.x_logvar])

        self.encoder = tf.keras.models.Model(encoder_inputs, encoder_output, name="encoder")
        self.encoder.summary()

    
        # Build decoder
        latent_inputs = tf.keras.Input(shape=(self.latent_dim,))
        x = tf.keras.layers.RepeatVector(self.timesteps)(latent_inputs)
        x = tf.keras.layers.LSTM(self.neurons[0], return_sequences=True, name="LSTM_decoder_layer")(x)
        decoder_outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))(x)

        self.decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")
        self.decoder.summary()


        # Build VAE
        vae_input = tf.keras.Input(shape=(self.timesteps, self.image_size,), name="VAE_input")
        vae_encoder_output = self.encoder(vae_input)
        vae_decoder_output = self.decoder(vae_encoder_output)
        vae = tf.keras.models.Model(vae_input, vae_decoder_output, name="VAE")
        vae.summary()

        return vae

    def train_models(self):
        
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

        # Train VAE
        x_train = self.df
        #print(x_train)
        #x_train = x_train[1:].expanding(axis=0).mean()
        #print(x_train)

        vae = self.models()
        
        vae.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005), loss=loss_func(self.x_mean, self.x_logvar))
        vae.fit(x_train, x_train,\
            epochs=30,
            batch_size=64, 
            callbacks=[cp_callback]
        )

        # Save models (decoder and/or encoder)

        self.encoder.save('./saved_models/lstm_encoder')
        self.decoder.save('./saved_models/lstm_decoder')

        vae.save('./saved_models/lstm_vae') 

def gen_seq(id_df, seq_length):

    data_matrix =  id_df
    num_elements = data_matrix.shape[0]

    for start, stop in zip(range(0, num_elements-seq_length, 1), range(seq_length, num_elements, 1)):
        
        yield data_matrix[stop-sequence_length:stop].values

if __name__ == "__main__":

    checkpoint_path = './saved_models/training/cp.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True,verbose=1)

    const = Config()
    neurons = const.VAE_neurons

    # Data loading and preparation
    data = DataPrep(file = os.path.join("../",const.file_path),
                    num_settings = const.num_settings,
                    num_sensors = const.num_sensors,
                    num_units = const.num_units[0],
                    prev_step_units = const.prev_step_units[0],
                    step = const.step[0],
                    normalization_type="01")

    df = data.ReadData()
    #print(df.shape)
    const.image_size = len(df.columns)-1
    const.sequence_length = 20


    sequence_length = const.sequence_length

    sequence_input = []

    for seq in gen_seq(df[list(chain(*[['NormTime'], data.setting_measurement_names]))], sequence_length):
        sequence_input.append(seq)
        
    sequence_input = np.asarray(sequence_input)

    #print(sequence_input.shape)

    vae = VAE_LSTM(sequence_input, const.image_size, const.sequence_length, const.latent_dim, [const.units, const.dense_neurons])
    vae.train_models()
