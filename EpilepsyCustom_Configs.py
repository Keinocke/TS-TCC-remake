class Config(object):
    def __init__(self):
        # model configs
        self.input_channels = 19  # Number of EEG channels
        self.kernel_size = 8    # Kernel size for convolutional layers
        self.stride = 1         
        self.final_out_channels = 128
        self.features_len = 322
        self.dropout = 0.35
        self.num_classes = 2

        # training configs
        self.num_epoch = 300
        self.batch_size = 32
        self.drop_last = False

        # optimizer
        self.lr = 0.001
        self.beta1 = 0.9
        self.beta2 = 0.999

        # sub-configs
        self.TC = TC()
        self.Context_Cont = Context_Cont_configs()
        self.augmentation = augmentations()


class TC(object):
    def __init__(self):
        self.hidden_dim = 100
        self.timesteps = 10


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 0.001
        self.jitter_ratio = 0.001
        self.max_seg = 5
