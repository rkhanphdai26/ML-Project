import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import FloodDataset
from model import AMFIMultimodalModel
import segmentation_models_pytorch as smp
import torchvision.transforms as T

# Fix for Windows OpenMP error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = "models"
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    # 1. DATA AUGMENTATION (The Overfitting Killer)
    # This transforms the images on-the-fly during training
    augmentations = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=15),
    ])

    full_dataset = FloodDataset(
        sar_dir=r"/data/mekong/s1",
        label_dir=r"/data/mekong/labels"
    )

    # Using the 24/6 split logic
    train_size = 24
    val_size = 6
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    model = AMFIMultimodalModel().to(device)

    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_dice = smp.losses.DiceLoss(mode='binary')

    # Increased weight_decay to further penalize overfitting
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-2)

    # 2. LEARNING RATE SCHEDULER
    # This reduces LR by half if Val IoU doesn't improve for 5 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    print(f"Starting Robust Training | Train: 24 | Val: 6")

    for epoch in range(50):
        model.train()
        train_loss = 0

        for sar, sensors, masks in train_loader:
            sar, sensors, masks = sar.to(device), sensors.to(device), masks.to(device)

            # Apply augmentations to the image and mask simultaneously
            # (Note: For SAR, we apply it to the SAR tensor)
            if torch.rand(1) > 0.5:
                sar = augmentations(sar)
                masks = augmentations(masks)

            optimizer.zero_grad()
            outputs = model(sar, sensors)
            loss = 0.5 * criterion_bce(outputs, masks) + 0.5 * criterion_dice(outputs, masks)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation Pass
        model.eval()
        val_iou = 0
        with torch.no_grad():
            for sar, sensors, masks in val_loader:
                sar, sensors, masks = sar.to(device), sensors.to(device), masks.to(device)
                outputs = model(sar, sensors)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                tp, fp, fn, tn = smp.metrics.get_stats(preds.long(), masks.long(), mode='binary')
                val_iou += smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item()

        avg_val_iou = val_iou / len(val_loader)

        # Step the scheduler based on Validation performance
        scheduler.step(avg_val_iou)

        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch [{epoch + 1}/50] | Loss: {train_loss / len(train_loader):.4f} | Val IoU: {avg_val_iou:.4f} | LR: {current_lr}")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"amfi_robust_ep{epoch + 1}.pth"))


if __name__ == "__main__":
    train()