# Keep all parameters here, so we don't need to change them manually everywhere

class Config(object):
    def __init__(self) -> None:
        self.file_path = "CMAPSSData/train_FD002.txt"
        self.num_settings = 3
        self.num_sensors = 21
        self.num_units = [200, 40, 20]
        self.prev_step_units = [0, 200, 240]
        self.step = ["VAE", "RL", "EVAL"]
        self.VAE_neurons = [256, 128, 64]