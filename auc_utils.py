from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc, f1_score
import numpy as np
import matplotlib.pyplot as plt
from scipy import interp
from itertools import cycle
import matplotlib


def compute_AUC_scores(y_true, y_pred, labels):
    """
    Computes the Area Under the Curve (AUC) from prediction scores
    y_true.shape  = [n_samples, n_classes]
    y_preds.shape = [n_samples, n_classes]
    labels.shape  = [n_classes]
    """
    AUROC_avg = roc_auc_score(y_true, y_pred)
    print('The average AUROC is {AUROC_avg:.4f}'.format(AUROC_avg=AUROC_avg))
    for y, pred, label in zip(y_true.transpose(), y_pred.transpose(), labels):
        print('The AUROC of {0:} is {1:.4f}'.format(label, roc_auc_score(y, pred)))

def plot_ROC_curve(y_true, y_pred, labels, roc_path): 
    """
    Plots the ROC curve from prediction scores
    y_true.shape  = [n_samples, n_classes]
    y_preds.shape = [n_samples, n_classes]
    labels.shape  = [n_classes]
    """
    n_classes = len(labels)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for y, pred, label in zip(y_true.transpose(), y_pred.transpose(), labels):
        fpr[label], tpr[label], _ = roc_curve(y, pred)
        roc_auc[label] = auc(fpr[label], tpr[label])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[label] for label in labels]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for label in labels:
        mean_tpr += interp(all_fpr, fpr[label], tpr[label])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.3f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=2)

    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.3f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=2)

    if len(labels) == 4:
        colors = ['green', 'cornflowerblue', 'darkorange', 'darkred']
    else:
        colors = ['green', 'cornflowerblue', 'darkred']
    for label, color in zip(labels, cycle(colors)):
        plt.plot(fpr[label], tpr[label], color=color, lw=lw,
                label='ROC curve of {0} (area = {1:0.3f})'
                ''.format(label, roc_auc[label]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    matplotlib.rcParams.update({'font.size': 14})
    plt.savefig('%s.png' % roc_path, pad_inches = 0, bbox_inches='tight')