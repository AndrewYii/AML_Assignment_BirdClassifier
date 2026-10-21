"""Evaluation metrics, identical for every experiment.

Required by the brief:
  * Top-1 accuracy  = (1/N) * sum 1{argmax(y) == ground truth}
  * Average accuracy per class = (1/C) * sum_i T_i, with T_i the accuracy on class i

Extra (justified in the report): Top-5 accuracy (fine-grained look-alikes are
often the model's second guess), and macro precision / recall / F1, which weight
every species equally, so frequent classes cannot hide errors on rare ones.
"""
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def compute_metrics(logits: np.ndarray, targets: np.ndarray, class_names=None):
    """Return (summary dict, per-class rows, confusion matrix) from raw logits."""
    n_classes = logits.shape[1]
    preds = logits.argmax(1)
    top5 = np.argsort(-logits, axis=1)[:, :5]

    top1_acc = float((preds == targets).mean())
    top5_acc = float((top5 == targets[:, None]).any(1).mean())

    # average accuracy per class, exactly as defined in the brief
    present = np.unique(targets)
    per_class_acc = np.full(n_classes, np.nan)
    for c in present:
        per_class_acc[c] = (preds[targets == c] == c).mean()
    mean_class_acc = float(np.nanmean(per_class_acc))

    labels = np.arange(n_classes)
    p, r, f1, support = precision_recall_fscore_support(targets, preds, labels=labels, zero_division=0)
    _, _, f1_w, _ = precision_recall_fscore_support(targets, preds, labels=labels,
                                                     average="weighted", zero_division=0)
    m = support > 0  # macro averages over classes that appear in the test set
    summary = {
        "top1": top1_acc,
        "mean_class_acc": mean_class_acc,
        "top5": top5_acc,
        "macro_precision": float(p[m].mean()),
        "macro_recall": float(r[m].mean()),
        "macro_f1": float(f1[m].mean()),
        "weighted_f1": float(f1_w),
        "n_test": int(len(targets)),
    }
    rows = []
    for c in range(n_classes):
        rows.append({
            "class_id": c,
            "class_name": class_names[c] if class_names else str(c),
            "support": int(support[c]),
            "accuracy": None if np.isnan(per_class_acc[c]) else float(per_class_acc[c]),
            "precision": float(p[c]),
            "recall": float(r[c]),
            "f1": float(f1[c]),
        })
    cm = confusion_matrix(targets, preds, labels=labels)
    return summary, rows, cm
