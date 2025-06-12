import os
import numpy as np
import torch
from torch.utils.data import Dataset

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, configs, training_mode='train_linear'):
        self.data_dir = data_dir
        self.configs = configs
        self.training_mode = training_mode
        self.samples = []

        for label in ['class0', 'class1']:
            label_dir = os.path.join(data_dir, label)
            if not os.path.isdir(label_dir):
                continue
            for file in os.listdir(label_dir):
                if file.endswith(".npy"):
                    self.samples.append((os.path.join(label_dir, file), int(label[-1])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        data = np.load(file_path, allow_pickle=True).astype(np.float32)
        data = torch.tensor(data)  # shape: (channels, time)

        if self.training_mode == "self_supervised":
            aug1 = self._augment(data)
            aug2 = self._augment(data)
            return data, label, aug1, aug2

        return data, label

    def _augment(self, data):
        # simple jitter + scaling for EEG
        noise = torch.randn_like(data) * self.configs.augmentation.jitter_scale_ratio
        return data + noise
