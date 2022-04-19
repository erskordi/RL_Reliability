from gc import callbacks
from tabnanny import verbose
import os
import numpy as np
import pickle
import tensorflow as tf

from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils

from itertools import chain

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


class VAE(tf.keras.Model):
    """
    Variational Autoencoder;
    It provides methods for developing encoder/decoder
    """

    def __init__(self, 
                 latent_dim=2,
                 image_size=21) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.upper_bound = 10.0

        # Losses: Total loss = reconstruction_loss + KL_loss
        self.total_loss_tracker = tf.keras.metrics.Mean(name="Total loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="Reconstruction loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="KL loss")

        # Load options
        self.save_options = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')
    
    def Sampling(self, inputs) -> np.ndarray:
        """
        Returns latent x using reparameterization trick (x = mu + sigma*epsilon)
        """
        x_mean, x_logvar = inputs
        batch = tf.shape(x_mean)[0]
        dim = tf.shape(x_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))

        return x_mean + tf.exp(0.5 * x_logvar) * epsilon

    def Encoder(self, neurons) -> tf.keras.Model:

        encoder_inputs = tf.keras.Input(shape=(self.image_size,))
        x = encoder_inputs

        for i in range(len(neurons)):
            if i == 0:
                x = tf.keras.layers.Dense(neurons[i], activation='relu', name="dense_layer_" + str(i))(x)
            elif (i > 0) & (i <= len(neurons)):
                x = tf.keras.layers.Dense(neurons[i], activation='relu', name="dense_layer_" + str(i))(x)
            else:
                x = tf.keras.layers.Dense(neurons[i], activation='relu', name="dense_layer_final")(x)
        
        x_mean = tf.keras.layers.Dense(self.latent_dim, activation='sigmoid', name="x_mean")(encoder_inputs)
        x_mean = x_mean * self.upper_bound # x is between 0 - 10
        x_logvar = tf.keras.layers.Dense(self.latent_dim, name="x_logvar")(encoder_inputs)
        x = self.Sampling([x_mean, x_logvar])

        self.encoder = tf.keras.Model(encoder_inputs, [x_mean, x_logvar, x], name="encoder")
        #self.encoder.summary()

        return self.encoder

    def Decoder(self, neurons) -> tf.keras.Model:

        latent_inputs = tf.keras.Input(shape=(self.latent_dim,))
        x = latent_inputs

        for i in range(len(neurons)-1,0,-1):
            if i == len(neurons) - 1:
                x = tf.keras.layers.Dense(neurons[i], activation='relu', name="latent_layer")(x)
            else:
                x = tf.keras.layers.Dense(neurons[i], activation='relu', name="dense_layer_" + str(i))(x)
        decoder_outputs = tf.keras.layers.Dense(self.image_size, activation="sigmoid", name="decoder_output")(x)

        self.decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")
        #self.decoder.summary()

        return self.decoder

    @property
    def metrics(self) -> list:
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    @tf.function
    def train_step(self, data) -> dict:
        with tf.GradientTape() as tape:
            x_mean, x_logvar, x = self.encoder(data)
            reconstruction = self.decoder(x)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.binary_crossentropy(data, reconstruction), axis=-1))
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

    def save_models(self, encoder, decoder):
        #encoder.save('saved_models/encoder')
        decoder.save('./saved_models/decoder.h5', options=self.save_options)

    def load_models(self):
        decoder = tf.keras.models.load_model('saved_models/decoder.h5', compile=False, options=self.save_options)
        with open('model.pkl', 'wb') as f:
            pickle.dump(decoder, f)
        return decoder

def get_model():
    return VAE(latent_dim=1,image_size=25)

if __name__ == "__main__":

    file_path = "CMAPSSData/train_FD002.txt"
    num_settings = 3
    num_sensors = 21
    num_units = 200
    step = "VAE"

    neurons = [256, 128, 64, 32, 16, 8]

    # Data prep
    data = DataPrep(file=file_path,
                    num_settings=num_settings, 
                    num_sensors=num_sensors, 
                    num_units=num_units, 
                    step=step,
                    normalization_type="01")
    
    df = data.ReadData()
    
    n = get_model()
    encoder = n.Encoder(neurons)
    decoder = n.Decoder(neurons)
    decoder.compile()
    n.compile(optimizer=tf.keras.optimizers.Adam())

    checkpoint_path = 'saved_models/training/cp.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True,verbose=1)
    n.fit(df[list(chain(*[['NormTime'], data.setting_measurement_names]))], epochs=10, batch_size=64, callbacks=[cp_callback])
    
    # Save decoder to use later as RL environment
    n.save_models(encoder, decoder)

