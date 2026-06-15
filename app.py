"""
=============================================================
  COLON CANCER MODEL EVALUATION SCRIPT
  University Project - TFLite Model Evaluator
=============================================================
  Dataset Structure:
  
  C:/colon_cancer_ai/clean_dataset_split/
  ├── train/
  │   ├── normal/
  │   └── adenocarcinoma/
  ├── validation/
  │   ├── normal/
  │   └── adenocarcinoma/
  └── test/
      ├── normal/
      └── adenocarcinoma/

  Output:
  - Accuracy, Precision, Recall, F1 Score
  - Confusion Matrix Heatmap
  - ROC Curve + AUC Score
  - Per-split results (train / validation / test)
  - Saved chart: evaluation_results.png
=============================================================
"""

import os
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH  = "colon_cancer_model.tflite"
DATASET_DIR = r"C:\colon_cancer_ai\clean_dataset_split"
IMAGE_SIZE  = (224, 224)
THRESHOLD   = 0.5
SAVE_PLOTS  = True

SPLITS = ["train", "validation", "test"]
CLASS_FOLDERS = {
    "normal":          0,
    "adenocarcinoma":  1
}
CLASS_NAMES = {0: "Normal", 1: "Adenocarcinoma"}
# ============================================================


# ------------------------------------------------------------
# LOAD TFLITE MODEL
# ------------------------------------------------------------
print("\n" + "="*60)
print("  COLON CANCER MODEL EVALUATION")
print("="*60)
print(f"\n[1/6] Loading TFLite model: {MODEL_PATH}")

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"      Input shape  : {input_details[0]['shape']}")
print(f"      Output shape : {output_details[0]['shape']}")
print("      ✅ Model loaded!")


# ------------------------------------------------------------
# HELPER: LOAD IMAGES FROM ONE SPLIT FOLDER
# ------------------------------------------------------------
def load_split(split_name):
    images, labels, paths = [], [], []
    split_dir = os.path.join(DATASET_DIR, split_name)
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

    for class_folder, label in CLASS_FOLDERS.items():
        class_dir = os.path.join(split_dir, class_folder)
        if not os.path.exists(class_dir):
            print(f"      ⚠️  Folder not found: {class_dir}")
            continue

        files = [f for f in os.listdir(class_dir) if f.lower().endswith(valid_ext)]
        print(f"      [{split_name}] {class_folder}: {len(files)} images")

        for filename in files:
            filepath = os.path.join(class_dir, filename)
            try:
                img = Image.open(filepath).convert("RGB").resize(IMAGE_SIZE)
                arr = np.array(img).astype("float32") / 255.0
                images.append(arr)
                labels.append(label)
                paths.append(filepath)
            except Exception as e:
                print(f"      ⚠️  Skip {filename}: {e}")

    return images, labels, paths


# ------------------------------------------------------------
# HELPER: RUN PREDICTIONS
# ------------------------------------------------------------
def predict_all(images):
    y_pred, y_probs = [], []
    for img_array in images:
        inp = np.expand_dims(img_array, axis=0)
        interpreter.set_tensor(input_details[0]['index'], inp)
        interpreter.invoke()
        raw = float(interpreter.get_tensor(output_details[0]['index'])[0][0])
        cancer_prob = 1.0 - raw
        y_probs.append(cancer_prob)
        y_pred.append(1 if cancer_prob >= THRESHOLD else 0)
    return y_pred, y_probs


# ------------------------------------------------------------
# HELPER: PRINT METRICS
# ------------------------------------------------------------
def print_metrics(split_name, y_true, y_pred, y_probs):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_probs)
    cm   = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"\n  ── {split_name.upper()} SET ──────────────────────────")
    print(f"  Accuracy     : {acc  * 100:.2f}%")
    print(f"  Precision    : {prec * 100:.2f}%")
    print(f"  Recall       : {rec  * 100:.2f}%")
    print(f"  Specificity  : {spec * 100:.2f}%")
    print(f"  F1 Score     : {f1   * 100:.2f}%")
    print(f"  AUC-ROC      : {auc  * 100:.2f}%")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(classification_report(y_true, y_pred, target_names=["Normal","Adenocarcinoma"]))

    return {"acc":acc,"prec":prec,"rec":rec,"spec":spec,"f1":f1,"auc":auc,"cm":cm,
            "tp":tp,"tn":tn,"fp":fp,"fn":fn,"fpr":roc_curve(y_true,y_probs)}


# ------------------------------------------------------------
# MAIN: LOOP OVER ALL SPLITS
# ------------------------------------------------------------
print(f"\n[2/6] Loading dataset from:\n      {DATASET_DIR}\n")

all_results = {}
all_y_true_combined = []
all_y_pred_combined = []
all_y_probs_combined = []

for split in SPLITS:
    print(f"\n  Loading '{split}' split...")
    images, labels, paths = load_split(split)
    if len(images) == 0:
        print(f"  ⚠️  No images found for '{split}', skipping.")
        continue

    print(f"  Running predictions on {len(images)} images...")
    y_pred, y_probs = predict_all(images)

    all_results[split] = {
        "images": images,
        "labels": labels,
        "paths":  paths,
        "y_pred": y_pred,
        "y_probs": y_probs
    }
    all_y_true_combined  += labels
    all_y_pred_combined  += y_pred
    all_y_probs_combined += y_probs

print(f"\n[3/6] Calculating metrics for each split...")
metrics = {}
for split, d in all_results.items():
    metrics[split] = print_metrics(split, d["labels"], d["y_pred"], d["y_probs"])

# Overall (test set is the most important)
print("\n" + "="*60)
print("  📊 OVERALL COMBINED RESULTS (All Splits)")
print("="*60)
metrics["overall"] = print_metrics("OVERALL", all_y_true_combined, all_y_pred_combined, all_y_probs_combined)


# ------------------------------------------------------------
# CHARTS
# ------------------------------------------------------------
print(f"\n[4/6] Generating charts...")

n_splits = len(all_results)
fig = plt.figure(figsize=(20, 5 * (n_splits + 1)))
fig.suptitle("Colon Cancer Detection — Model Evaluation", fontsize=16, fontweight='bold', y=1.01)

row = 0
plot_order = list(all_results.keys()) + ["overall"]
combined_data = {**all_results, "overall": {
    "labels": all_y_true_combined,
    "y_pred": all_y_pred_combined,
    "y_probs": all_y_probs_combined
}}

for split in plot_order:
    d = combined_data[split]
    y_true  = d["labels"]
    y_pred  = d["y_pred"]
    y_probs = d["y_probs"]

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_probs)
    cm   = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr_vals, tpr_vals, _ = roc_curve(y_true, y_probs)

    # Col 1: Confusion Matrix
    ax1 = fig.add_subplot(len(plot_order), 3, row * 3 + 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=["Normal","Adenocarcinoma"],
                yticklabels=["Normal","Adenocarcinoma"],
                linewidths=1, linecolor='gray')
    ax1.set_title(f"[{split.upper()}] Confusion Matrix", fontweight='bold')
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")

    # Col 2: Metrics Bar
    ax2 = fig.add_subplot(len(plot_order), 3, row * 3 + 2)
    names  = ["Accuracy","Precision","Recall","Specificity","F1","AUC"]
    values = [acc, prec, rec, spec, f1, auc]
    colors = ['#2196F3','#4CAF50','#FF9800','#9C27B0','#F44336','#00BCD4']
    bars = ax2.bar(names, [v*100 for v in values], color=colors, edgecolor='black', linewidth=0.7)
    ax2.set_ylim(0, 115)
    ax2.set_title(f"[{split.upper()}] Performance Metrics", fontweight='bold')
    ax2.set_ylabel("Score (%)")
    ax2.set_xticklabels(names, rotation=20, ha='right', fontsize=8)
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()+1,
                 f"{val*100:.1f}%", ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Col 3: ROC Curve
    ax3 = fig.add_subplot(len(plot_order), 3, row * 3 + 3)
    ax3.plot(fpr_vals, tpr_vals, color='#F44336', lw=2, label=f'AUC = {auc:.4f}')
    ax3.plot([0,1],[0,1], color='gray', linestyle='--', lw=1)
    ax3.fill_between(fpr_vals, tpr_vals, alpha=0.1, color='#F44336')
    ax3.set_xlim([0,1]); ax3.set_ylim([0,1.05])
    ax3.set_xlabel("False Positive Rate"); ax3.set_ylabel("True Positive Rate")
    ax3.set_title(f"[{split.upper()}] ROC Curve", fontweight='bold')
    ax3.legend(loc="lower right", fontsize=9)
    ax3.grid(True, alpha=0.3)

    row += 1

plt.tight_layout()

if SAVE_PLOTS:
    out_path = os.path.join(os.path.dirname(MODEL_PATH), "evaluation_results.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"      ✅ Chart saved: {out_path}")

plt.show()


# ------------------------------------------------------------
# SAMPLE PREDICTIONS — TEST SET ONLY
# ------------------------------------------------------------
print(f"\n[5/6] Sample Predictions from TEST set (first 15):")
print("-"*70)
print(f"  {'#':<4} {'File':<38} {'True':<16} {'Pred':<16} {'Status'}")
print("-"*70)

if "test" in all_results:
    test_d = all_results["test"]
    for i in range(min(15, len(test_d["labels"]))):
        fname   = os.path.basename(test_d["paths"][i])[:36]
        true_n  = CLASS_NAMES[test_d["labels"][i]]
        pred_n  = CLASS_NAMES[test_d["y_pred"][i]]
        status  = "✅ CORRECT" if test_d["labels"][i] == test_d["y_pred"][i] else "❌ WRONG"
        print(f"  {i+1:<4} {fname:<38} {true_n:<16} {pred_n:<16} {status}")
print("-"*70)


# ------------------------------------------------------------
# FINAL SUMMARY
# ------------------------------------------------------------
print("\n" + "="*60)
print("  🏁 FINAL SUMMARY")
print("="*60)
for split in list(all_results.keys()) + ["overall"]:
    y_true  = combined_data[split]["labels"]
    y_pred  = combined_data[split]["y_pred"]
    y_probs = combined_data[split]["y_probs"]
    acc = accuracy_score(y_true, y_pred) * 100
    auc = roc_auc_score(y_true, y_probs) * 100
    n   = len(y_true)
    print(f"  {split.upper():<12} | Images: {n:<6} | Accuracy: {acc:.2f}%  | AUC: {auc:.2f}%")
print("="*60)
print("\n  ✅ Evaluation complete! Check evaluation_results.png\n")
