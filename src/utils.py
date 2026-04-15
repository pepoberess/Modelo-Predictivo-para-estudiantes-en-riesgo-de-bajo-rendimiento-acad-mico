import numpy as np
import matplotlib.pyplot as plt
from .metrics import confusion_matrix, roc_curve, pr_curve

def print_confusion_matrix(y_true, y_pred):
    TP, FP, FN, TN = confusion_matrix(y_true, y_pred)
    
    print("Confusion Matrix:")
    print(f"TP: {TP} | FP: {FP}")
    print(f"FN: {FN} | TN: {TN}\n")

    
def plot_roc_curves(results, title="ROC Curves"):
    """
    Plot ROC curves for multiple splits in the same figure.
    
    Parameters
    ----------
    results : list of dict with keys: 'name', 'y_true', 'y_prob'
    title   : figure title
    """
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4), sharey=True)
    if len(results) == 1:
        axes = [axes]
 
    for ax, res in zip(axes, results):
        fprs, tprs, auc = roc_curve(res["y_true"], res["y_prob"])
        ax.plot(fprs, tprs, color="steelblue", lw=2, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(res["name"])
        ax.legend(loc="lower right")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
 
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()
 
def plot_pr_curves(results, title="Precision-Recall Curves"):
    """
    Plot PR curves for multiple splits in the same figure.
 
    Parameters
    ----------
    results : list of dict with keys: 'name', 'y_true', 'y_prob'
    title   : figure title
    """
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4), sharey=True)
    if len(results) == 1:
        axes = [axes]
 
    for ax, res in zip(axes, results):
        precisions, recalls, auc = pr_curve(res["y_true"], res["y_prob"])
        baseline = np.mean(res["y_true"])
        ax.plot(recalls, precisions, color="darkorange", lw=2, label=f"AUC = {auc:.3f}")
        ax.axhline(y=baseline, color="gray", linestyle="--", lw=1, label=f"Baseline = {baseline:.2f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(res["name"])
        ax.legend(loc="upper right")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
 
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()