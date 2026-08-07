import numpy as np

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score


def clustering_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    labels_true = np.unique(y_true)
    labels_pred = np.unique(y_pred)

    cost_matrix = np.zeros((len(labels_true), len(labels_pred)), dtype=int)

    for i, true_label in enumerate(labels_true):
        for j, pred_label in enumerate(labels_pred):
            cost_matrix[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))

    row_ind, col_ind = linear_sum_assignment(-cost_matrix)

    correct = cost_matrix[row_ind, col_ind].sum()

    return correct / len(y_true)

def nmi_score(y_true, y_pred):
    return normalized_mutual_info_score(y_true, y_pred)


