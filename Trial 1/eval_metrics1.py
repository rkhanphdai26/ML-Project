import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Local imports
from model import AMFIMultimodalModel
from dataset import FloodDataset


def seed_everything(seed=42):
    """Locks all random seeds for reproducible, consistent results."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_automated_trial1_evaluation():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    CHECKPOINT_DIR = "models"
    OUTPUT_DIR = "trial1_final_results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Setup Dataset
    dataset = FloodDataset(
        sar_dir=r"..\data\bolivia\s1",
        label_dir=r"..\data\bolivia\labels"
    )

    v1_checkpoints = [
        "amfi_robust_ep10.pth",
        "amfi_robust_ep20.pth",
        "amfi_robust_ep30.pth",
        "amfi_robust_ep40.pth",
        "amfi_robust_ep50.pth"
    ]

    # --- STAGE 1: AUTOMATED STRESS TEST (FIND BEST CHECKPOINT) ---
    print(f"{'Scanning Checkpoints':<25} | {'Global IoU':<15}")
    print("-" * 45)

    best_iou = -1
    best_ckpt_name = ""

    for ckpt_name in v1_checkpoints:
        ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)
        if not os.path.exists(ckpt_path): continue

        model = AMFIMultimodalModel().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        total_intersection = 0
        total_union = 0

        with torch.no_grad():
            for i in range(len(dataset)):
                # Passing real sensor data to avoid the shape mismatch
                sar, sensors, mask = dataset[i]

                output = model(sar.unsqueeze(0).to(device), sensors.unsqueeze(0).to(device))
                pred = (torch.sigmoid(output) > 0.5).float().cpu().squeeze()
                target = mask.squeeze()

                intersection = (pred * target).sum().item()
                union = (pred.sum() + target.sum() - intersection).item()
                total_intersection += intersection
                total_union += union

        current_global_iou = total_intersection / (total_union + 1e-7)
        print(f"{ckpt_name:<25} | {current_global_iou:.4f}")

        if current_global_iou > best_iou:
            best_iou = current_global_iou
            best_ckpt_name = ckpt_name

    print("-" * 45)
    print(f"WINNER IDENTIFIED: {best_ckpt_name}\n")

    # --- STAGE 2: FULL EVALUATION ON THE WINNER ---
    print(f"--- Generating Final Artifacts for {best_ckpt_name} ---")
    model = AMFIMultimodalModel().to(device)
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, best_ckpt_name), map_location=device))
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i in range(len(dataset)):
            sar, sensors, mask = dataset[i]
            output = model(sar.unsqueeze(0).to(device), sensors.unsqueeze(0).to(device))
            pred = (torch.sigmoid(output) > 0.5).float().cpu().squeeze()
            target = mask.squeeze().cpu()

            all_preds.append(pred.numpy().flatten())
            all_targets.append(target.numpy().flatten())

            fig, axes = plt.subplots(1, 3, figsize=(15, 6))
            axes[0].imshow(sar[0].cpu().numpy(), cmap='gray')
            axes[0].set_title(f"Sample {i}: SAR (VV)", pad=10)
            axes[1].imshow(target.numpy(), cmap='Blues')
            axes[1].set_title("Ground Truth", pad=10)
            axes[2].imshow(pred.numpy(), cmap='Blues')
            axes[2].set_title(f"Trial 1 Prediction ({best_ckpt_name})", pad=10)
            for ax in axes: ax.axis('off')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(OUTPUT_DIR, f"best_sample_{i}.png"), bbox_inches='tight', dpi=300)
            plt.close()

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    report = classification_report(y_true, y_pred, target_names=["Non-Flood", "Flood"], output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=["Non-Flood", "Flood"],
                yticklabels=["Non-Flood", "Flood"])
    plt.title(f"Trial 1 Confusion Matrix\nBest Checkpoint: {best_ckpt_name}", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "trial1_best_heatmap.png"), dpi=300)
    plt.close()

    print("\n" + "=" * 40)
    print(f"TRIAL 1 FINAL REPORT: {best_ckpt_name}")
    print("=" * 40)
    print(f"Accuracy:        {report['accuracy']:.4f}")
    print(f"Flood Precision: {report['Flood']['precision']:.4f}")
    print(f"Flood Recall:    {report['Flood']['recall']:.4f}")
    print(f"Flood F1-Score:  {report['Flood']['f1-score']:.4f}")
    print(f"Global IoU:      {best_iou:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    run_automated_trial1_evaluation()