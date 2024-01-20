import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter


# 生成器模型设计
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.beta = Parameter(torch.randn(48, 48), requires_grad=True)
        self.alpha = Parameter(torch.FloatTensor(164, 70), requires_grad=True)
        self.reset_parameters()


    def reset_parameters(self):
        tdv = 1. / math.sqrt(self.alpha.size(1))
        self.alpha.data.uniform_(-tdv, tdv)

    def forward(self,Pa,Ec,Wc,Ev,Pb,W_pie,feat):
        feat = feat.numpy()
        c_count = len(Pa)
        row = feat.shape[1]
        col = feat.shape[2]
        batch_del_Ec = {}
        batch_del_Ev = {}
        F = np.zeros(shape=(c_count, row, col))
        #初始化
        for i in range(c_count):
            C_name = 'C' + str(i + 1)
            batch_del_Ec[C_name+'的社团的社团熵信息增量矩阵'] = {}
            batch_del_Ev[C_name + '的节点熵信息增量矩阵'] = {}
        for i in range(c_count):
            C_name = 'C' + str(i + 1)
            P_a_i = Pa[C_name + '的社团的社团熵信息传播系数矩阵']
            P_b_i = Pb[C_name + '的节点熵信息传播系数矩阵']
            E_c_i = Ec[C_name+'的社团的社团熵信息矩阵']
            W_c_i = Wc[C_name+'的关键边权重矩阵']
            E_v_i = Ev[C_name+'的节点熵信息矩阵']
            W_i = W_pie[i]
            E_c = np.eye(P_a_i.shape[0])
            E_v = np.eye(P_b_i.shape[0])
            del_EC = np.dot(P_a_i,np.dot(E_c_i,W_c_i)) * E_c
            del_EV = np.dot(P_b_i,np.dot(E_v_i,W_i)) * E_v
            batch_del_Ec[C_name+'的社团的社团熵信息增量矩阵'] = del_EC
            Ec[C_name + '的社团的社团熵信息矩阵'] = del_EC+E_c_i
            batch_del_Ev[C_name + '的节点熵信息增量矩阵'] = del_EV
            F[i] = 0.3*np.dot(del_EV,feat[i]) + feat[i] #gama = 0.3


        generator_W = np.zeros(shape=(c_count, row, row))

        for i in range(0, c_count):
            for j in range(0, row):
                for k in range(0, row):
                    R = np.corrcoef(F[i, j], F[i, k])
                    generator_W[i, j, k] = R[0, 1]

        return generator_W
