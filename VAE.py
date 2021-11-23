import numpy as np
import tensorflow as tf

from data_prep import DataPrep, Vec2Img


class VAE(tf.keras.Model):
    """
    Convolutional Variational Autoencoder;
    It provides methods for developing encoder/decoder
    """

    def __init__(self, 
                 latent_dim=2,
                 image_size=21) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

        # Losses: Total loss = reconstruction_loss + KL_loss
        self.total_loss_tracker = tf.keras.metrics.Mean(name="Total loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="Reconstruction loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="KL loss")
    
    def Sampling(self, inputs) -> np.ndarray:
        """
        Returns latent x using reparameterization trick (x = mu + sigma*epsilon)
        """
        x_mean, x_logvar = inputs
        batch = tf.shape(x_mean)[0]
        dim = tf.shape(x_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))

        return x_mean + tf.exp(0.5 * x_logvar) * epsilon

    def Encoder(self, filters, kernel, stride, neurons) -> tf.keras.Model:

        encoder_inputs = tf.keras.Input(shape=(self.image_size,self.image_size,1))

        x = tf.keras.layers.Conv2D(filters=filters[0], kernel_size=kernel, strides=stride, activation='relu', padding='same', name="conv2D_layer_1")(encoder_inputs)
        x = tf.keras.layers.Conv2D(filters=filters[1], kernel_size=kernel, strides=stride, activation='relu', padding='same', name="conv2D_layer_2")(x)
        x = tf.keras.layers.Flatten(name="flattened")(x)
        z = tf.keras.layers.Dense(neurons, activation='relu', name="dense_layer")(x)

        x_mean = tf.keras.layers.Dense(self.latent_dim, name="x_mean")(z)
        x_logvar = tf.keras.layers.Dense(self.latent_dim, name="x_logvar")(z)
        x = self.Sampling([x_mean, x_logvar])

        self.encoder = tf.keras.Model(encoder_inputs, [x_mean, x_logvar, x], name="encoder")
        #self.encoder.summary()

        return self.encoder

    def Decoder(self, filters, kernel, stride, neurons) -> tf.keras.Model:

        op_filters, sens_filters = filters

        latent_inputs = tf.keras.Input(shape=(self.latent_dim,))
        x = tf.keras.layers.Dense(self.encoder.get_layer('flattened').output_shape[1], activation='relu', name="latent_layer")(latent_inputs)
        x = tf.keras.layers.Reshape(self.encoder.get_layer('conv2D_layer_2').output_shape[1:])(x)
        x = tf.keras.layers.Conv2DTranspose(filters=filters[1], kernel_size=kernel, strides=stride, activation='relu', padding='same', name="conv2DTranspose_layer_1")(x)
        x = tf.keras.layers.Conv2DTranspose(filters=filters[0], kernel_size=kernel, strides=stride, activation='relu', padding='same', name="conv2DTranspose_layer_2")(x)
        decoder_outputs = tf.keras.layers.Conv2DTranspose(1, kernel_size=kernel, strides=1, activation=None, padding='same', name="conv2DTranspose_layer_out")(x)

        self.decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")
        #self.decoder.summary()

        return self.decoder

    @property
    def metrics(self) -> list:
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data) -> dict:
        with tf.GradientTape() as tape:
            x_mean, x_logvar, x = self.encoder(data)
            reconstruction = self.decoder(x)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.binary_crossentropy(data, reconstruction), axis=(1,2)))
            kl_loss = -.5 * (1 + x_logvar - tf.square(x_mean) - tf.exp(x_logvar))
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
            "kl_loss": self.kl_loss_tracker.result()
        }

def img_norm(data):
    images = np.expand_dims(data, -1).astype("float32") / 255

    return images

if __name__ == "__main__":

    file_path = "CMAPSSData/train_FD002.txt"
    num_settings = 3
    num_sensors = 21
    num_units = 100
    step = "VAE"

    filters = [8,16]
    kernel = 2
    stride = 2
    neurons = 4

    # Data prep
    data = DataPrep(file=file_path,
                    num_settings=num_settings, 
                    num_sensors=num_sensors, 
                    num_units=num_units, 
                    step=step)
    
    df = data.ReadData()
    
    image_data = Vec2Img(df=df,
                         data=data,
                         image_size=num_settings+num_sensors,
                         plot=True)
    
    images = image_data.Transform()
    images = img_norm(images)
    
    n = VAE(latent_dim=1,image_size=24)
    encoder = n.Encoder(filters, kernel, stride, neurons)
    decoder = n.Decoder(filters, kernel, stride, neurons)
    n.compile(optimizer=tf.keras.optimizers.Adam())
    n.fit(images, epochs=30, batch_size=32)

    # Save decoder to use later as RL environment
    decoder.save('saved_models/environment.h5')


