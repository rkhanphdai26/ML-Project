import os
import torch
import numpy as np
import tifffile as tiff
from torch.utils.data import Dataset

class StochasticHydrology:
    def __init__(self, timesteps=24):
        self.t = timesteps

    def generate(self, flood_fraction):
        # 1. Rainfall: Higher probability of heavy rain if pixels are flooded
        rain_scale = 12.0 if flood_fraction > 0.1 else 2.5
        rain = np.random.gamma(shape=2, scale=rain_scale, size=self.t)


        moisture = np.clip(np.cumsum(rain) * 0.12 + 25, 0, 100)

        # 3. Water Level: Modeled as delayed discharge response
        level = np.zeros(self.t)
        for i in range(1, self.t):
            # Physics: Level rises when soil is saturated; decays slowly (0.96)
            level[i] = level[i - 1] * 0.96 + (rain[i] * (moisture[i] / 100))

        # 4. Flow Velocity: Proportional to river height + sensor noise
        flow = level * 0.45 + np.random.normal(0, 0.7, self.t)

        # Stack into [Time, Features] -> [24, 4]
        seq = np.stack([level, rain, moisture, flow], axis=1)
        return torch.tensor(seq, dtype=torch.float32)



class FloodDatasetV2(Dataset):
    def __init__(self, sar_dir, label_dir, domain_name="Unknown"):
        self.sar_files = sorted([os.path.join(sar_dir, f) for f in os.listdir(sar_dir) if f.endswith('.tif')])
        self.label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.tif')])
        self.hydrology = StochasticHydrology()
        self.domain_name = domain_name

    def __len__(self):
        return len(self.sar_files)

    def __getitem__(self, idx):
        # SAR Processing with enhanced Z-score Normalization[cite: 2]
        sar = tiff.imread(self.sar_files[idx]).astype('float32')
        sar = np.nan_to_num(sar, nan=0.0, posinf=0.0, neginf=-25.0)
        sar = np.clip(sar, -25.0, 0)
        sar = (sar + 25.0) / 25.01  # Ensuring strictly [0, 1][cite: 2]

        # Label Processing[cite: 2]
        label = tiff.imread(self.label_files[idx]).astype('float32')
        label = np.nan_to_num(label, nan=0.0)
        label = (label > 0).astype('float32')
        if len(label.shape) == 2: label = np.expand_dims(label, axis=0)

        # Hydrology Generation[cite: 2]
        flood_fraction = np.mean(label)
        sensor_data = self.hydrology.generate(flood_fraction)

        sar_tensor = torch.from_numpy(sar)
        if len(sar_tensor.shape) == 3 and sar_tensor.shape[2] == 2:
            sar_tensor = sar_tensor.permute(2, 0, 1)

        return sar_tensor, sensor_data, torch.from_numpy(label), self.domain_name