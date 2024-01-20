import math
import numpy as np
import torch
from torch.nn.parameter import Parameter
from torch import nn

class Community_Convolution_layer(nn.Module):
    def __init__(self, use_bias=False):
        super(Community_Convolution_layer, self).__init__()
        self.use_bias = use_bias
        self.theta = Parameter(torch.FloatTensor(70, 70))
        self.reset_parameters()

    def save_weight(self):
        pass

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.theta.size(1))
        self.theta.data.uniform_(-stdv, stdv)

    def forward(self,all_C,all_C_Hc,all_C_Rcs,all_C_Rc,all_C_H,all_C_Rnp,all_C_D,all_Vertex,W):
        all_C_Rct = {}
        all_C_del_Hc = {}
        for i in range(10):  # 遍历每个人的脑区基因网络的节点度，先遍历第一个人
            C_index = 'C' + str(i + 1) + '的亲和力矩阵'
            all_C_Rct[C_index] = {}
            C_Del_Hc_index = 'C' + str(i + 1)
            all_C_del_Hc[C_Del_Hc_index] = {}
        hum = len(all_C)

        # 1、团间卷积的过程
        for hum in range(10):
            C_index = 'C' + str(hum + 1)
            CRc_index_name = 'C' + str(hum + 1) + '的亲和力矩阵'
            Hc = all_C_Hc[C_index]
            Hc = torch.FloatTensor(Hc)

            F = np.ones(shape=(len(Hc), len(Hc)))
            F = torch.FloatTensor(F)

            Rc = all_C_Rc[CRc_index_name]
            Rc = torch.FloatTensor(Rc)

            E = np.eye(len(Hc))
            E = torch.FloatTensor(E)
            alerpha = 0.1 #α取值范围0.1~1
            del_Hc = alerpha * torch.mm(F, torch.mm(Hc, Rc)) * E
            del_Hc = del_Hc.numpy()

            all_C_del_Hc[C_index] = del_Hc

            CRcs_index_name = 'C' + str(hum + 1) + '的亲和力对角矩阵'
            Rcs = all_C_Rcs[CRcs_index_name]
            Rcs = torch.FloatTensor(Rcs)
            Rcs = Rcs.inverse()
            del_Hc = torch.FloatTensor(del_Hc)
            del_Rc = torch.mm(torch.mm(Rc, Rcs), del_Hc)
            del_Rc_t = del_Rc.t()
            Rc_k = del_Rc + del_Rc_t + Rc
            Rc_k = Rc_k.numpy()
            all_C_Rct[CRc_index_name] = Rc_k

        # 2、团内卷积的过程
        for hum in range(10):
            C_index = 'C' + str(hum + 1)
            C_i_Cnum = len(all_C[C_index])
            for C_i_index in range(C_i_Cnum):
                C_index_name = 'C' + str(C_i_index + 1) + '社区'
                Wp = all_C[C_index][C_index_name]
                Wp = torch.FloatTensor(Wp)

                CRn_index_name = 'C' + str(C_i_index + 1) + '社区内的亲和力矩阵'
                Rn = all_C_Rnp[C_index][CRn_index_name]
                Rn = torch.FloatTensor(Rn)

                Hp = all_C_H[C_index][C_index_name+'的特征矩阵']
                Hp = torch.FloatTensor(Hp)

                CD_index_name = 'C' + str(C_i_index + 1) + '社区的节点度矩阵'
                D = all_C_D[C_index][CD_index_name]
                D_12 = np.sqrt(D)
                D_12 = torch.FloatTensor(D_12)
                D_F12 = D_12.inverse()  # D的负1/2次幂
                beta = 0.1 #β取值范围0.1~1

                Hp_k = beta*torch.mm(torch.mm(torch.mm(torch.mm(D_F12, (Wp * Rn)), D_F12), Hp), self.theta)
                Hp_k = Hp_k.detach().numpy()
                Hp_k_index = 'C' + str(hum + 1) + '社区'
                all_C_H[C_index][Hp_k_index] = Hp_k

        for hum in range(10):
            C_index = 'C' + str(hum + 1)
            C_len = len(all_Vertex[C_index])  # 看第i个人有几个社团
            for i in range(161):
                for j in range(161):
                    for C_xh in range(C_len):
                        C_name = 'C' + str(C_xh + 1) + '社区'
                        if i in all_Vertex[C_index][C_name]:
                            p = C_xh
                        if j in all_Vertex[C_index][C_name]:
                            q = C_xh
                    if p != q:
                        CRc_index_name = 'C' + str(hum + 1) + '的亲和力矩阵'
                        Rct = all_C_Rct[CRc_index_name]
                        Rc = all_C_Rc[CRc_index_name]
                        W[hum, i, j] = W[hum, i, j] * Rct[p, q] / Rc[p, q]

        return W,all_C_H

