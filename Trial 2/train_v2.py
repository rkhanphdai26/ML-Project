import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_v2 import FloodDatasetV2
from model import AMFIMultimodalModel
import segmentation_models_pytorch as smp

import sys
# Adds PythonProject root to the path so 'model' is findable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'E:\PythonProject')))

from model import AMFIMultimodalModel
# Fix for Windows OpenMP error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def train_v2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure save directory exists
    save_dir = "models"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 1. Multi-Domain Dataset Setup
    # Trial 2: Using the V2 dataset class with Domain Tagging[cite: 2, 5]
    mekong_ds = FloodDatasetV2(
        sar_dir=r"E:\PythonProject\data\mekong\s1",
        label_dir=r"E:\PythonProject\data\mekong\labels",
        domain_name="Mekong"
    )

    train_loader = DataLoader(mekong_ds, batch_size=4, shuffle=True)

    model = AMFIMultimodalModel().to(device)

    # Hybrid Loss (0.5 BCE + 0.5 Dice) to balance pixel accuracy and overlap[cite: 9]
    criterion_bce = torch.nn.BCEWithLogitsLoss()
    criterion_dice = smp.losses.DiceLoss(mode='binary')

    # Optimization: Increased weight decay (1e-1) for stronger generalization[cite: 9]
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-1)

    print("Starting AMFI Trial V2: Enhanced Generalization & Metric Tracking")

    for epoch in range(50):
        model.train()
        epoch_loss = 0
        epoch_iou = 0  # Accumulator for Training IoU

        for sar, sensors, masks, domains in train_loader:
            sar, sensors, masks = sar.to(device), sensors.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(sar, sensors)

            # Hybrid Loss Calculation[cite: 9]
            loss = 0.5 * criterion_bce(outputs, masks) + 0.5 * criterion_dice(outputs, masks)

            loss.backward()

            # Gradient Clipping (Max Norm 1.0) to stabilize the BiGRU temporal branch
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_loss += loss.item()

            # --- Calculate Training Metrics ---
            # Apply sigmoid and threshold at 0.5 for binary segmentation
            preds = (torch.sigmoid(outputs) > 0.5).float()

            # Extract statistics for IoU calculation[cite: 9]
            tp, fp, fn, tn = smp.metrics.get_stats(preds.long(), masks.long(), mode='binary')

            # Calculate batch-level IoU[cite: 9]
            batch_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item()
            epoch_iou += batch_iou

        # Calculate epoch-wide averages
        avg_loss = epoch_loss / len(train_loader)
        avg_iou = epoch_iou / len(train_loader)

        print(f"Epoch [{epoch + 1}/50] | Loss: {avg_loss:.4f} | Train IoU: {avg_iou:.4f}")

        # Checkpointing: Saving V2-specific weights every 10 epochs[cite: 9]
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(save_dir, f"amfi_v2_ep{epoch + 1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"--> Checkpoint saved: {save_path}")


if __name__ == "__main__":
    train_v2()