from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

def compute_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def compute_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)

def compute_auc(y_true, y_scores):
    return roc_auc_score(y_true, y_scores)