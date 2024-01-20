import numpy as np
import torch
import math

from sklearn.metrics import roc_auc_score


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    acc = correct / len(labels)
    return acc

def onehot_encode(labes):
    classes = set(labes)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labes_onehot = np.array(list(map(classes_dict.get, labes)), dtype=np.int32)
    #print(labes_onehot.shape)
    return labes_onehot

# 风险预测指标计算
def predict_indicators(output, label):
    output = output.max(1)[1].type_as(label)
    TP = ((output == 1) & (label == 1)).sum()
    TN = ((output == 0) & (label == 0)).sum()
    FP = ((output == 0) & (label == 1)).sum()
    FN = ((output == 1) & (label == 0)).sum()
    ACC = (TP + TN) / (TP + TN + FP + FN)
    SEN = TP / (TP + FN)
    SPE = TN / (FP + TN)
    TPR = TP / (TP + FN)
    FPR = FP / (TP + FP)
    output = output.detach().numpy()
    label = label.detach().numpy()
    output[np.isnan(output)] = 0
    # 预测值是概率
    AUC = roc_auc_score(label,output)
    return ACC, SEN, SPE, AUC, TPR, FPR


