import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_error_curve(rounds, train_errors, test_errors):
    plt.figure(figsize=(8, 6))
    plt.plot(rounds, train_errors, marker='o', linestyle='-', label='Training Error')
    plt.plot(rounds, test_errors, marker='s', linestyle='-', label='Test Error')
    plt.xlabel("Boosting Round")
    plt.ylabel("Error")
    plt.title("Training and Testing Error vs Boosting Rounds")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_comparison_weak_vs_boost(y_true, stump_scores, boost_scores):
    # convert y_true to binary format for ROC computation
    y_true = np.where(y_true == 1, 1, 0)

    # ROC for the base weak learner
    fpr_stump, tpr_stump, _ = roc_curve(y_true, stump_scores)
    auc_stump = auc(fpr_stump, tpr_stump)

    # ROC for the boosted model
    fpr_boost, tpr_boost, _ = roc_curve(y_true, boost_scores)
    auc_boost = auc(fpr_boost, tpr_boost)

    plt.figure(figsize=(8, 6))
    # base weak learner
    plt.plot(fpr_stump, tpr_stump, color='red', lw=2, label=f"Weak Learner ROC (AUC = {auc_stump:0.2f})")
    # boosted model
    plt.plot(fpr_boost, tpr_boost, color='green', lw=2, label=f"AdaBoost ROC (AUC = {auc_boost:0.2f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Comparison: Weak Learner vs AdaBoost")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()