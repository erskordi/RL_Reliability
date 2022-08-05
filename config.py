# Keep all parameters here, so we don't need to change them manually everywhere

class Config(object):
    def __init__(self) -> None:
        self.file_path = "CMAPSSData/train_FD002.txt"
        self.num_settings = 3
        self.num_sensors = 21
        self.num_units = [200, 40, 20]
        self.prev_step_units = [0, self.num_units[0], self.num_units[0]+self.num_units[1]]
        self.step = ["VAE", "RL", "EVAL"]
        self.VAE_neurons = [256, 128, 64]
        self.num_frames = 50

        # Parameters for VAEs
        self.latent_dim = 2
        self.image_size = 25

        # Parameters for VAE-LSTM
        self.sequence_length = 5 # look-back window
        self.units = 64
        self.dense_neurons = 32

        # policy parameters
        self.policy_neurons = [256, 128, 64]