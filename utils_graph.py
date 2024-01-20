import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score
import torch

def onehot_encode(labes):
    classes = set(labes)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labes_onehot = np.array(list(map(classes_dict.get, labes)), dtype=np.int32)
    # print(labes_onehot.shape)
    return labes_onehot


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def stastic_indicators(output, labels):
    TP = ((output.max(1)[1] == 1) & (labels == 1)).sum()
    TN = ((output.max(1)[1] == 0) & (labels == 0)).sum()
    FN = ((output.max(1)[1] == 0) & (labels == 1)).sum()
    FP = ((output.max(1)[1] == 1) & (labels == 0)).sum()
    ACC = (TP + TN) / (TP + TN + FP + FN)
    SEN = TP / (TP + FN)
    SPE = TN / (FP + TN)
    output = output.detach().numpy()
    labels = labels.detach().numpy()
    output[np.isnan(output)] = 0
    AUC = roc_auc_score(labels, output[:, 1])
    # ------------------
    PRE = TP / (TP + FP)
    REC = SEN
    F1 = (2 * PRE * REC) / (PRE + REC)
    print("TP", TP)
    print("TN", TN)
    print("FN", FN)
    print("FP", FP)
    return ACC, SEN, SPE, AUC, F1

