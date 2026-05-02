import os
import sys
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

# --- SHARED CONFIGURATION ---
PROJECT_ROOT = r'E:\PythonProject'
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from model import AMFIMultimodalModel
from dataset_v2 import FloodDatasetV2

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def set_seed(seed=42):
    """Preserves results by locking random states across all libraries[cite: 1]"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed locked to: {seed}")


def get_dataloader():
    """Consistent data loading for Bolivia test set[cite: 1, 4]"""
    return FloodDatasetV2(
        r"E:\PythonProject\data\bolivia\s1",
        r"E:\PythonProject\data\bolivia\labels",
        "Bolivia"
    )


def evaluate_checkpoints_global(checkpoint_list):
    """PHASE 1: Global IoU Sweep and detailed terminal feedback[cite: 1, 2, 5]"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = get_dataloader()
    results = {}

    print("\n--- PHASE 1: Sweeping Checkpoints using Global IoU ---")
    print(f"{'Checkpoint':<20} | {'Global IoU':<15}")
    print("-" * 40)

    for ckpt_name in checkpoint_list:
        ckpt_path = os.path.join("models", ckpt_name)
        if not os.path.exists(ckpt_path):
            continue

        model = AMFIMultimodalModel().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        total_inter = 0
        total_union = 0

        with torch.no_grad():
            for i in range(len(dataset)):
                sar, sensors, mask, _ = dataset[i]
                output = model(sar.unsqueeze(0).to(device), sensors.unsqueeze(0).to(device))
                pred = (torch.sigmoid(output) > 0.5).float().cpu().squeeze()
                target = mask.squeeze().cpu()

                # Update Global Stats[cite: 1, 2]
                total_inter += (pred * target).sum().item()
                total_union += (pred + target).clamp(0, 1).sum().item()

        global_iou = total_inter / (total_union + 1e-7)
        results[ckpt_name] = global_iou
        print(f"{ckpt_name:<20} | {global_iou:.4f}")

    best_ckpt = max(results, key=results.get)
    print(f"\nMathematical Best Weight: {best_ckpt} (IoU: {results[best_ckpt]:.4f})")
    return best_ckpt, results


def save_and_plot_artifacts(best_ckpt, history, checkpoints_list):
    """PHASE 2: detailed stats, visual evolution, and per-sample comparisons[cite: 3, 4, 5]"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = get_dataloader()

    # 1. Setup Directories[cite: 4]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_path = f"Trial2_Analysis_Report_{timestamp}"
    comp_path = os.path.join(run_path, "all_sample_comparisons")
    os.makedirs(comp_path, exist_ok=True)

    print(f"\n--- PHASE 2: Generating Artifacts in '{run_path}' ---")

    # 2. Plot Training Evolution Line Graph[cite: 5]
    epochs = [int(k.split('_ep')[-1].split('.pth')[0]) for k in history.keys()]
    ious = list(history.values())
    sorted_data = sorted(zip(epochs, ious))
    epochs, ious = zip(*sorted_data)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, ious, marker='o', color='#2980b9', linewidth=2, markersize=8)
    plt.title("AMFI Trial 2: Global IoU Training Evolution", fontsize=16)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Global IoU", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(run_path, "global_iou_evolution.png"), bbox_inches='tight', pad_inches=0.2)
    plt.close()

    # 3. Training Visual Evolution Strip (Sample 0)[cite: 5]
    sample_idx = 0
    sar_ev, sensors_ev, mask_ev, _ = dataset[sample_idx]
    num_panels = len(checkpoints_list) + 2
    fig_ev, axes_ev = plt.subplots(1, num_panels, figsize=(num_panels * 5, 5))

    axes_ev[0].imshow(sar_ev[0].cpu().numpy(), cmap='gray')
    axes_ev[0].set_title("Input SAR (VV)", fontsize=14)
    axes_ev[1].imshow(mask_ev.squeeze().cpu().numpy(), cmap='Blues')
    axes_ev[1].set_title("Ground Truth", fontsize=14)

    for i, ckpt_name in enumerate(checkpoints_list):
        ckpt_path = os.path.join("models", ckpt_name)
        model = AMFIMultimodalModel().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        with torch.no_grad():
            out = model(sar_ev.unsqueeze(0).to(device), sensors_ev.unsqueeze(0).to(device))
            pred = (torch.sigmoid(out) > 0.5).float().cpu().squeeze()
            axes_ev[i + 2].imshow(pred.numpy(), cmap='Blues')
            axes_ev[i + 2].set_title(f"Epoch {ckpt_name.split('_ep')[-1].split('.')[0]}", fontsize=14)

    for ax in axes_ev: ax.axis('off')
    plt.savefig(os.path.join(run_path, "visual_evolution_strip.png"), bbox_inches='tight', pad_inches=0.3)
    plt.close()

    # 4. Per-Sample Analysis & Global Metrics for BEST Weight[cite: 3, 4]
    model = AMFIMultimodalModel().to(device)
    model.load_state_dict(torch.load(os.path.join("models", best_ckpt), map_location=device))
    model.eval()

    all_preds, all_masks = [], []
    print(f"Generating detailed comparisons for all {len(dataset)} samples...")

    with torch.no_grad():
        for idx in range(len(dataset)):
            sar, sensors, mask, _ = dataset[idx]
            output = model(sar.unsqueeze(0).to(device), sensors.unsqueeze(0).to(device))
            pred = (torch.sigmoid(output) > 0.5).float().cpu().squeeze()
            target = mask.squeeze().cpu()

            all_preds.append(pred.numpy().flatten())
            all_masks.append(target.numpy().flatten())

            # Individual Comparison Plot[cite: 4]
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            axes[0].imshow(sar[0].cpu().numpy(), cmap='gray')
            axes[0].set_title(f"Sample {idx}: SAR (VV)")
            axes[1].imshow(target.numpy(), cmap='Blues')
            axes[1].set_title("Ground Truth")
            axes[2].imshow(pred.numpy(), cmap='Blues')
            axes[2].set_title(f"Prediction ({best_ckpt})")

            for ax in axes: ax.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(comp_path, f"sample_{idx}_comparison.png"), bbox_inches='tight', pad_inches=0.2)
            plt.close()

    # 5. Final Report & Confusion Matrix[cite: 3]
    all_preds = np.concatenate(all_preds)
    all_masks = np.concatenate(all_masks)

    acc = accuracy_score(all_masks, all_preds)
    prec = precision_score(all_masks, all_preds)
    recall = recall_score(all_masks, all_preds)
    f1 = f1_score(all_masks, all_preds)

    print(f"\n--- Final Global Metrics ({best_ckpt}) ---")
    print(f"Accuracy:  {acc:.4f}\nPrecision: {prec:.4f}\nRecall:    {recall:.4f}\nF1 Score:  {f1:.4f}")

    cm = confusion_matrix(all_masks, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Dry', 'Flood'], yticklabels=['Dry', 'Flood'])
    plt.title(f"Confusion Matrix: {best_ckpt}")
    plt.savefig(os.path.join(run_path, "final_confusion_matrix.png"), bbox_inches='tight', pad_inches=0.2)
    plt.close()


if __name__ == "__main__":
    set_seed(42)  # Locking for reproducibility[cite: 1]
    checkpoints = ["amfi_v2_ep10.pth", "amfi_v2_ep20.pth", "amfi_v2_ep30.pth", "amfi_v2_ep40.pth", "amfi_v2_ep50.pth"]

    best_weight, history = evaluate_checkpoints_global(checkpoints)
    save_and_plot_artifacts(best_weight, history, checkpoints)
    print("\nTrial 2 analysis successfully completed.")