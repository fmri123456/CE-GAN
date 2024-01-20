import math
import numpy as np
import Convolution_layer
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.parameter import Parameter

class Com_GCN(nn.Module):
    def __init__(self):
        super(Com_GCN, self).__init__()
        # 社团卷积
        self.conv1 = Convolution_layer.Community_Convolution_layer()
        self.conv2 = Convolution_layer.Community_Convolution_layer()

        # 全连接层
        self.theta1 = Parameter(torch.randn(322, 161))
        self.theta2 = Parameter(torch.randn(161, 35))
        self.theta3 = Parameter(torch.randn(35, 2))


    def reset_parameters(self):
        stdv1 = 1. / math.sqrt(self.theta1.size(1))
        self.theta1.data.uniform_(-stdv1, stdv1)
        stdv2 = 1. / math.sqrt(self.theta2.size(1))
        self.theta2.data.uniform_(-stdv2, stdv2)
        stdv3 = 1. / math.sqrt(self.theta3.size(1))
        self.theta3.data.uniform_(-stdv3, stdv3)

    def forward(self, all_C, all_C_Hc, all_C_Rcs, all_C_Rc, all_C_H, all_C_Rnp, all_C_D, all_Vertex, W, all_C_in_K, all_C_P):
        hum_num = np.size(W.numpy(), 0)
        W, H = self.conv1.forward(all_C, all_C_Hc, all_C_Rcs, all_C_Rc, all_C_H, all_C_Rnp, all_C_D, all_Vertex, W)
        # 特征向量X
        all_C_X = {}  # 所有人的特征向量X
        all_C_label = []  # 所有人的标签
        for i in range(hum_num):  # 遍历每个人的脑区基因网络的节点度，先遍历第一个人
            C_name = 'C' + str(i + 1)
            all_C_X[C_name] = np.append(all_C_in_K[C_name], all_C_P[C_name])

        for i in range(hum_num):  # 遍历每个人的脑区基因网络的节点度，先遍历第一个人
            C_name = 'C' + str(i + 1)
            fc_rl = torch.FloatTensor(all_C_X[C_name])
            print(self.theta1)
            fc1 = F.relu(torch.matmul(fc_rl, self.theta1))
            print(fc1)
            fc2 = F.relu(torch.matmul(fc1, self.theta2))
            print(fc2)
            fc3 = F.softmax(torch.matmul(fc2, self.theta3))
            print(fc3)
            all_C_label.append(fc3)
        return all_C_label, W, H, self.theta1, self.theta2, self.theta3
