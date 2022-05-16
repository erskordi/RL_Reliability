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

def unpack(model, training_config, weights):
    restored_model = deserialize(model)
    if training_config is not None:
        restored_model.compile(
            **saving_utils.compile_args_from_training_config(
                training_config
            )
        )
    restored_model.set_weights(weights)
    return restored_model

# Hotfix function
def make_keras_picklable():

    def __reduce__(self):
        model_metadata = saving_utils.model_metadata(self)
        training_config = model_metadata.get("training_config", None)
        model = serialize(self)
        weights = self.get_weights()
        return (unpack, (model, training_config, weights))

    cls = tf.keras.Model
    cls.__reduce__ = __reduce__

class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        """
        Returns latent x using reparameterization trick (x = mu + sigma*epsilon)
        """
        x_mean, x_logvar = inputs
        batch = tf.shape(x_mean)[0]
        dim = tf.shape(x_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))

        return x_mean + tf.exp(0.5 * x_logvar) * epsilon


class VAE(tf.keras.Model):
    """
    Variational Autoencoder;
    """

    def __init__(self, 
                 encoder=None,
                 decoder=None) -> None:
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        # Losses: Total loss = reconstruction_loss + KL_loss
        self.total_loss_tracker = tf.keras.metrics.Mean(name="Total loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="Reconstruction loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="KL loss")

        # Load options
        self.save_options = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')

    @property
    def metrics(self) -> list:
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    @tf.function
    def train_step(self, data) -> dict:
        with tf.GradientTape() as tape:
            x_mean, x_logvar, x = self.encoder(data)
            reconstruction = self.decoder(x)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(data, reconstruction), axis=-1)
                    )
            kl_loss = -.5 * (1 + x_logvar - tf.square(x_mean) - tf.exp(x_logvar))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=-1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result()
        }


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
    x = Sampling()([x_mean, x_logvar])

    encoder = tf.keras.Model(encoder_inputs, [x_mean, x_logvar, x], name="encoder")
    #encoder.summary()

    # Build decoder
    latent_inputs = tf.keras.Input(shape=(const.latent_dim,))
    x = latent_inputs

    for i in range(len(neurons)-1,0,-1):
        if i == len(neurons) - 1:
            x = tf.keras.layers.Dense(neurons[i], activation='relu', name="latent_layer")(x)
        else:
            x = tf.keras.layers.Dense(neurons[i], activation='relu', name="dense_layer_" + str(i))(x)
    decoder_outputs = tf.keras.layers.Dense(const.image_size, activation="sigmoid", name="decoder_output")(x)

    decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")
    #decoder.summary()

    # Train VAE
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=tf.keras.optimizers.Adam())
    vae.fit(df[list(chain(*[['NormTime'], data.setting_measurement_names]))],\
        epochs=100,
        batch_size=64, 
        callbacks=[cp_callback]
    )
    
    decoder.save('./saved_models/decoder')
    with open('decoder.pkl', 'wb') as f:
        pickle.dump(decoder, f)
    """
    vae.build((None,) + (const.image_size,))
    vae.save('./saved_models/model', save_format='tf')
    with open('model.pkl', 'wb') as f:
        pickle.dump(vae, f)
    """
