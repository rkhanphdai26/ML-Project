import os
import torch
import numpy as np
import tifffile as tiff
from torch.utils.data import Dataset, DataLoader

class StochasticHydrology:
    def __init__(self, timesteps=24):
        self.t = timesteps

    def generate(self, flood_fraction):
        # 1. Rainfall: Higher probability of heavy rain if pixels are flooded
        rain_scale = 12.0 if flood_fraction > 0.1 else 2.5
        rain = np.random.gamma(shape=2, scale=rain_scale, size=self.t)

        # 2. Soil Moisture: Cumulative absorption logic
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


class FloodDataset(Dataset):
    def __init__(self, sar_dir, label_dir):
        self.sar_files = sorted([os.path.join(sar_dir, f) for f in os.listdir(sar_dir) if f.endswith('.tif')])
        self.label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.tif')])
        self.hydrology = StochasticHydrology()

        if len(self.sar_files) != len(self.label_files):
            print(f"Warning: Found {len(self.sar_files)} SAR and {len(self.label_files)} Labels.")

    def __len__(self):
        return len(self.sar_files)

    def __getitem__(self, idx):
        # --- SAR PROCESSING ---
        sar = tiff.imread(self.sar_files[idx]).astype('float32')

        # 1. Kill NaNs/Infs: Replace NaNs with 0 and Infs with decibel boundaries
        # This is the "Safety Shield" against NaN loss.
        sar = np.nan_to_num(sar, nan=0.0, posinf=0.0, neginf=-25.0)

        # 2. Advanced SAR Normalization: Clipping decibel outliers
        # Standard S1 range is roughly -25 to 0 dB
        sar = np.clip(sar, -25.0, 0)

        # 3. Safe Scaling: Using 25.01 to avoid any possible division by zero error
        sar = (sar + 25.0) / 25.01

        # 4. Final bounds check: Ensure everything is strictly [0, 1]
        sar = np.clip(sar, 0, 1)

        # --- LABEL PROCESSING ---
        label = tiff.imread(self.label_files[idx]).astype('float32')
        label = np.nan_to_num(label, nan=0.0)  # Ensure labels are also clean

        # Force binary values. If it's > 0, it becomes 1.0.
        label = (label > 0).astype('float32')

        if len(label.shape) == 2:
            label = np.expand_dims(label, axis=0)  # [1, H, W]

        # --- SENSOR PROCESSING ---
        flood_fraction = np.mean(label)
        sensor_data = self.hydrology.generate(flood_fraction)

        # Final PyTorch Conversion
        sar_tensor = torch.from_numpy(sar)
        if len(sar_tensor.shape) == 3 and sar_tensor.shape[2] == 2:
            sar_tensor = sar_tensor.permute(2, 0, 1)

        label_tensor = torch.from_numpy(label)

        return sar_tensor, sensor_data, label_tensor


# --- TEST SCRIPT ---
if __name__ == "__main__":
    SAR_PATH = r"/data/mekong/s1"
    LABEL_PATH = r"/data/mekong/labels"

    if os.path.exists(SAR_PATH):
        dataset = FloodDataset(SAR_PATH, LABEL_PATH)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

        sar, sensors, masks = next(iter(dataloader))

        print("--- Dataset Stream Test ---")
        print(f"SAR Shape (Batch, C, H, W): {sar.shape}")
        print(f"SAR Range: {sar.min().item():.4f} to {sar.max().item():.4f}")
        print(f"Sensor Shape (Batch, T, F): {sensors.shape}")
        print(f"Mask Shape (Batch, C, H, W): {masks.shape}")
        print("---------------------------")
        print("Success: Dataset is NaN-protected and ready!")
    else:
        print(f"Directory {SAR_PATH} not found. Check your paths.")