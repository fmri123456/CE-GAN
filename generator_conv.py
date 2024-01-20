import math
from sklearn.metrics import mutual_info_score
import numpy as np
import torch

import load_data


class generator_convolution_layer():
    def forward(self,Pa,Ec,Wc,Ev,Pb,W_pie,feat):
        c_count = len(Pa)
        row = feat.shape[1]
        col = feat.shape[2]
        batch_del_Ec = {}
        batch_del_Ev = {}
        F = np.zeros(c_count, row, col)
        # 初始化
        for i in range(c_count):
            C_name = 'C' + str(i + 1)
            batch_del_Ec[C_name + '的社团的社团熵信息增量矩阵'] = {}
            batch_del_Ev[C_name + '的节点熵信息增量矩阵'] = {}
        for i in range(c_count):
            C_name = 'C' + str(i + 1)
            P_a_i = Pa[C_name + '的社团的社团熵信息传播系数矩阵']
            P_b_i = Pb[C_name + '的节点熵信息传播系数矩阵']
            E_c_i = Ec[C_name + '的社团的社团熵信息矩阵']
            W_c_i = Wc[C_name + '的关键边权重矩阵']
            E_v_i = Ev[C_name + '的节点熵信息矩阵']
            W_i = W_pie[i]
            E = np.eye(P_a_i.shape[0])
            del_EC = np.dot(P_a_i, np.dot(E_c_i, W_c_i)) * E
            del_EV = np.dot(P_b_i, np.dot(E_v_i, W_i)) * E
            batch_del_Ec[C_name + '的社团的社团熵信息增量矩阵'] = del_EC
            Ec[C_name + '的社团的社团熵信息矩阵'] = del_EC + E_c_i
            batch_del_Ev[C_name + '的节点熵信息增量矩阵'] = del_EV
            F[i] = 0.3 * np.dot(del_EV, feat[i]) + feat[i]

        generator_W = np.zeros(shape=(c_count, row, row))
        for i in range(0, c_count):
            for j in range(0, row):
                for k in range(0, row):
                    R = np.corrcoef(F[i, j], F[i, k])
                    generator_W[i, j, k] = R[0, 1]

        return generator_W
