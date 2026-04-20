import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .metrics import confusion_matrix, roc_curve, pr_curve, roc_curve_multiclass, pr_curve_multiclass

def print_confusion_matrix(y_true, y_pred):
    TP, FP, FN, TN = confusion_matrix(y_true, y_pred)
    
    print("Confusion Matrix:")
    print(f"TP: {TP} | FP: {FP}")
    print(f"FN: {FN} | TN: {TN}\n")

def print_confusion_matrix_multiclass(y_true, y_pred, class_names, classes):
    labels = [class_names.get(c, str(c)) for c in classes]
    
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for i, true_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            matrix[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))
    
    df = pd.DataFrame(matrix, index=labels, columns=labels)
    df.index.name = "Real \\ Pred"
    
    print("Confusion Matrix:")
    display(df)

    
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

def plot_cv_f1(lambdas, f1_kfold, f1_group, f1_temporal, title="F1 vs λ (Cross-Validation)"):
    """
    Plot F1 score vs lambda for the three splitting strategies.

    Parameters
    ----------
    lambdas     : list of lambda values evaluated
    f1_kfold    : list of F1 scores for random KFold
    f1_group    : list of F1 scores for GroupKFold by school
    f1_temporal : list of F1 scores for temporal split
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    configs = [
        (axes[0], f1_kfold,    "KFold Aleatorio",     "steelblue"),
        (axes[1], f1_group,    "GroupKFold (Escuela)", "darkorange"),
        (axes[2], f1_temporal, "Temporal Split",       "seagreen"),
    ]

    for ax, f1s, name, color in configs:
        ax.plot(lambdas, f1s, marker="o", color=color, lw=2)
        best_idx = int(np.argmax(f1s))
        ax.axvline(x=lambdas[best_idx], color="gray", linestyle="--", lw=1,
                label=f"λ óptimo = {lambdas[best_idx]}")
        ax.set_xscale("log")
        ax.set_xlabel("λ (regularización)")
        ax.set_ylabel("F1 Score")
        ax.set_title(name)
        ax.legend()

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_coefficient_stability(weights_kfold, weights_group, feature_names):
    """
    Boxplot of logistic regression coefficients per feature for KFold vs GroupKFold.
    One subplot per feature, showing both distributions side by side.
 
    Parameters
    ----------
    weights_kfold  : array of shape (n_folds_kfold, n_features)
    weights_group  : array of shape (n_folds_group, n_features)
    feature_names  : list of feature names matching columns of weights arrays
    """
    # Only plot non-one-hot features for readability (skip escuela_* dummies)
    base_features = [f for f in feature_names if not f.startswith("escuela_")]
    base_idx = [feature_names.index(f) for f in base_features]
 
    n = len(base_features)
    n_cols = 4
    n_rows = int(np.ceil(n / n_cols))
 
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.ravel()
 
    for i, (feat, idx) in enumerate(zip(base_features, base_idx)):
        ax = axes[i]
        data_to_plot = [weights_kfold[:, idx], weights_group[:, idx]]
        bp = ax.boxplot(data_to_plot, labels=["KFold", "GroupKFold"], patch_artist=True)
        bp["boxes"][0].set_facecolor("steelblue")
        bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor("darkorange")
        bp["boxes"][1].set_alpha(0.6)
        ax.set_title(feat, fontsize=9)
        ax.axhline(y=0, color="gray", linestyle="--", lw=0.8)
 
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
 
    fig.suptitle("Coefficients distributions: KFold vs GroupKFold", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def show_weights(data, features, best_lam, splits, preprocess_fn, model_class):
    all_weights = []
    for train_idx, val_idx in splits:
        train_f = data.loc[train_idx].reset_index(drop=True)
        val_f   = data.loc[val_idx].reset_index(drop=True)
 
        train_f, val_f = preprocess_fn(train_f, val_f, features)
 
        X_train = train_f.drop(columns=["rendimiento", "target_b"]).values.astype(float)
        y_train = train_f["target_b"].values
        feature_names = train_f.drop(columns=["rendimiento", "target_b"]).columns.tolist()
 
        model = model_class(X_train, y_train, L2=best_lam)
        model.fit(L2=True)
        all_weights.append(model.weights)
 
    weights = np.array(all_weights)  # shape: (n_folds, n_features)
    return weights, feature_names

def plot_roc_curves_multiclass(results, classes, class_name, title="ROC Curves (One-vs-All)"):
    """
    Parameters
    ----------
    results : list of dict con keys: 'name', 'y_true', 'y_prob'
    classes : list of class labels
    class_name : str, name of the class for which to plot curves
    title : str, title for the entire figure
    """
    fig, axes = plt.subplots(len(results), len(classes), 
                              figsize=(4 * len(classes), 4 * len(results)), sharey=True)
    
    if len(results) == 1:
        axes = [axes]

    for i, res in enumerate(results):
        curves = roc_curve_multiclass(res["y_true"], res["y_prob"], classes)
        for j, c in enumerate(classes):
            ax = axes[i][j]
            fprs, tprs, auc = curves[c]
            ax.plot(fprs, tprs, color="steelblue", lw=2, label=f"AUC = {auc:.3f}")
            ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.set_title(f"{res['name']} | clase {class_name.get(c, str(c))}")
            ax.legend(loc="lower right")
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_pr_curves_multiclass(results, classes, class_name, title="PR Curves (One-vs-All)"):
    """
    Parameters
    ----------
    results : list of dict con keys: 'name', 'y_true', 'y_prob'
    classes : list of class labels
    class_name : str, name of the class for which to plot curves
    title : str, title for the entire figure
    """
    fig, axes = plt.subplots(len(results), len(classes),
                              figsize=(4 * len(classes), 4 * len(results)), sharey=True)

    if len(results) == 1:
        axes = [axes]

    for i, res in enumerate(results):
        curves = pr_curve_multiclass(res["y_true"], res["y_prob"], classes)
        for j, c in enumerate(classes):
            ax = axes[i][j]
            precisions, recalls, auc = curves[c]
            baseline = np.mean((res["y_true"] == c).astype(int))
            ax.plot(recalls, precisions, color="darkorange", lw=2, label=f"AUC = {auc:.3f}")
            ax.axhline(y=baseline, color="gray", linestyle="--", lw=1, label=f"Baseline = {baseline:.2f}")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title(f"{res['name']} | clase {class_name.get(c, str(c))}")
            ax.legend(loc="upper right")
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()