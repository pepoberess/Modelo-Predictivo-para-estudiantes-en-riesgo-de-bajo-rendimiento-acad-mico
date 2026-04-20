import numpy as np

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    return TP / (TP + FP) if (TP + FP) > 0 else 0

def recall(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return TP / (TP + FN) if (TP + FN) > 0 else 0

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r)) if (p + r) > 0 else 0

def confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return TP, FP, FN, TN

def print_metrics(y_true, y_pred, title="Metrics"):
    acc = accuracy(y_true, y_pred)
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"{title}:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")

def print_metrics_multiclass(y_true, y_pred, classes, class_names, title="Metrics"):
    print(f"\n{title}")
    print(f"{'Clase':<15} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 55)
    for c in classes:
        y_true_b = (y_true == c).astype(int)
        y_pred_b = (y_pred == c).astype(int)
        acc = accuracy(y_true_b, y_pred_b)
        prec = precision(y_true_b, y_pred_b)
        rec = recall(y_true_b, y_pred_b)
        f1 = f1_score(y_true_b, y_pred_b)
        name = class_names.get(c, str(c))
        print(f"{name:<15} {acc:>10.4f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f}")

def roc_curve(y_true, y_prob):
    """Compute ROC curve (FPR, TPR) and AUC using numpy."""
    thresholds = np.linspace(0, 1, 300)
    fprs, tprs = [], []
 
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        TP, FP, FN, TN = confusion_matrix(y_true, y_pred)
        tprs.append(TP / (TP + FN) if (TP + FN) > 0 else 0)
        fprs.append(FP / (FP + TN) if (FP + TN) > 0 else 0)
 
    fprs, tprs = np.array(fprs), np.array(tprs)
    # sort by FPR for AUC calculation
    order = np.argsort(fprs)
    fprs, tprs = fprs[order], tprs[order]
    auc = np.trapezoid(tprs, fprs)
    return fprs, tprs, auc
 
def pr_curve(y_true, y_prob):
    """Compute Precision-Recall curve and AUC using numpy."""
    thresholds = np.linspace(0, 1, 300)
    precisions, recalls = [], []
 
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        TP, FP, FN, TN = confusion_matrix(y_true, y_pred)
        precisions.append(TP / (TP + FP) if (TP + FP) > 0 else 1)
        recalls.append(TP / (TP + FN) if (TP + FN) > 0 else 0)
 
    precisions, recalls = np.array(precisions), np.array(recalls)
    # sort by recall for AUC calculation
    order = np.argsort(recalls)
    recalls, precisions = recalls[order], precisions[order]
    auc = np.trapezoid(precisions, recalls)
    return precisions, recalls, auc

def roc_curve_multiclass(y_true, y_prob, classes):
    """One-vs-all ROC curve per class. y_prob shape: (n_samples, n_classes)"""
    curves = {}
    for i, c in enumerate(classes):
        y_true_b = (y_true == c).astype(int)
        fprs, tprs, auc = roc_curve(y_true_b, y_prob[:, i])
        curves[c] = (fprs, tprs, auc)
    return curves

def pr_curve_multiclass(y_true, y_prob, classes):
    """One-vs-all PR curve per class. y_prob shape: (n_samples, n_classes)"""
    curves = {}
    for i, c in enumerate(classes):
        y_true_b = (y_true == c).astype(int)
        precisions, recalls, auc = pr_curve(y_true_b, y_prob[:, i])
        curves[c] = (precisions, recalls, auc)
    return curves