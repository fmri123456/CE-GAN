import copy
import math

import scipy.io as sio
import numpy as np
import torch
import os
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import Construction_matrix
import utils
from discriminator import discriminator
from generator import generator
from sklearn.metrics import mutual_info_score

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 导入数据
# =========AD_NC
dataFile = 'AD risk prediction task based LMCI.mat'
data = sio.loadmat(dataFile)

feat_train = torch.tensor(np.concatenate((data['feat_AD'], data['feat_LMCI'])))[0:210]
feat_test = torch.tensor(np.concatenate((data['AD_true'], data['AD_fake'])))  # 就是AD_fake+AD_true

test_num = feat_test.shape[0]
brain_num = feat_test.shape[1]
gene_num = feat_test.shape[2]
ad_test_W = np.zeros(shape=(test_num, brain_num, brain_num))
ad_test_W_Chebyshev = np.zeros(shape=(test_num, brain_num, brain_num))
for i in range(0, test_num):
    for j in range(0, brain_num):
        for k in range(0, brain_num):
            R = np.corrcoef(feat_test[i, j], feat_test[i, k])
            ad_test_W[i, j, k] = R[0, 1]

            Chebyshev = max(abs(feat_test[i, j] - feat_test[i, k]))
            ad_test_W_Chebyshev[i, j, k] = Chebyshev

train_num = feat_train.shape[0]
ad_train_W = np.zeros(shape=(train_num, brain_num, brain_num))
ad_train_W_Chebyshev = np.zeros(shape=(train_num, brain_num, brain_num))
for i in range(0, train_num):
    for j in range(0, brain_num):
        for k in range(0, brain_num):
            R = np.corrcoef(feat_train[i, j], feat_train[i, k])
            ad_train_W[i, j, k] = R[0, 1]

            Chebyshev = max(abs(feat_train[i, j] - feat_train[i, k]))
            ad_train_W_Chebyshev[i, j, k] = Chebyshev


np.save("ad_test_W",ad_test_W)
np.save("ad_train_W",ad_train_W)

ad_train_W = np.load("ad_train_W.npy")
ad_test_W = np.load("ad_test_W.npy")
ad_train = np.zeros(shape=(210,161,161))
for i in range(210):
    if np.isnan(ad_train_W[i]).any() ==False:
        ad_train[i] = ad_train_W[i]


test_num = feat_test.shape[0]
train_num = feat_train.shape[0]



# 社团网络构建
def weight_threshold(W):
    row = W.shape[0]
    col = W.shape[1]
    result = np.zeros(shape=(row,col,col))
    for i in range(row):
        result_1 = W[i].copy()
        result_2 = W[i].copy()
        threshold = np.sort(np.abs(W[i].flatten()))[int(col * col * 0.6)]  # 阈值
        result_1[result_1 <= threshold] = 0  # 全负矩阵
        result_2[result_2 >= (-threshold)] = 0 #  全负矩阵
        result[i] = result_1 + result_2
    return result

index = np.random.permutation(test_num)
ad_train_W = weight_threshold(ad_train_W)
ad_test_W = weight_threshold(ad_test_W)

ad_test_W = ad_test_W[index]
ad_train_W = torch.tensor(ad_train_W,dtype=torch.float32) #为节省时间仅取0到29的脑区基因网络
ad_test_W = torch.tensor(ad_test_W,dtype=torch.float32) #为节省时间仅取0到29的脑区基因网络
# 定义标签
train_label = np.concatenate((np.zeros(int(train_num/2)),np.ones(int(train_num/2))))[0:210]
train_label = torch.LongTensor(train_label)
# 定义标签
test_label = np.concatenate((np.zeros(int(test_num/2)),np.ones(int(test_num/2))))
test_label = test_label[index] # 此时数组已经打乱
test_label = torch.LongTensor(test_label)


# 其他参数定义
LR = 0.0000001
EPOCH = 1
batch_size = 10
feat_train = feat_train
dataset = TensorDataset(ad_train_W, feat_train, train_label)
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
print("数据导入成功！")


# 开始训练
print("开始训练...")
ACC, SEN, SPE, AUC, TPR, FPR = 0,0,0,0,0,0
max_acc = 0
D_loss = []
acc_list = []
loss_list = []

G = generator()
g_optimizer = torch.optim.Adam(params=G.parameters(), lr=LR)
D = discriminator()
# 判定器的优化器
d_optimizer = torch.optim.Adam(D.parameters(), lr=LR)
# 损失函数：二进制交叉熵损失
criterion = nn.CrossEntropyLoss()
P_all = []
for epoch in range(EPOCH):
    for step, (W_train, feat_train, label_train) in enumerate(train_loader):
        batch_C, batch_Vertex, batch_C_H, batch_C_Hc, batch_C_Vertex_Node_degree, batch_C_rich_node, batch_C_Kernel_node, batch_E_C, batch_E_V, batch_P_a, batch_P_b, batch_C_W_c,batch_W_pie= Construction_matrix.all_matrix(
            W_train, feat_train)
        print("第{}轮第{}批数据训练...".format(epoch + 1, step + 1))

        Generator_W= G.forward(batch_P_a,batch_E_C,batch_C_W_c,batch_E_V,batch_P_b,batch_W_pie,feat_train)

        Generator_W = weight_threshold(Generator_W)

        Generator_W = torch.tensor(Generator_W,dtype=torch.float32)
        batch_C1, batch_Vertex1, batch_C_H1, batch_C_Hc1, batch_C_Vertex_Node_degree1, batch_C_rich_node1, batch_C_Kernel_node1, batch_E_C1, batch_E_V1, batch_P_a1, batch_P_b1, batch_C_W_c1, batch_W_pie1 = Construction_matrix.all_matrix(
            Generator_W, feat_train)

        # 特征提取
        def entropy(labels):
            n_labels = len(labels)

            if n_labels <= 1:
                return 0

            counts = np.bincount(labels)
            probs = counts[np.nonzero(counts)] / n_labels
            n_classes = len(probs)

            if n_classes <= 1:
                return 0
            return - np.sum(probs * np.log(probs)) / np.log(n_classes)

        T = np.zeros(shape=(batch_size,161))
        for i in range(0,batch_size):
            for j in range(0,161):
                C_in = "C" + str(i+1)
                vertex_i = batch_Vertex[C_in]
                vertex_i1 = batch_Vertex1[C_in]
                C_num = len(vertex_i)
                C_num1 = len(vertex_i1)
                a = 0
                b = 0
                for k1 in range(0,C_num):
                    C_num_i_name = "C" + str(k1 + 1)+"社区"
                    vertex_list = batch_Vertex[C_in][C_num_i_name]
                    if j in vertex_list:
                        a = k1 + 1
                for k2 in range(0,C_num1):
                    C_num_i_name1 = "C" + str(k2 + 1) + "社区"
                    vertex_list1 = batch_Vertex1[C_in][C_num_i_name1]
                    if j in vertex_list1:
                        b = k2 + 1
                a_kernel = batch_C_Kernel_node[C_in]["C" + str(a) + "社区的核心节点"]
                b_kernel = batch_C_Kernel_node1[C_in]["C" + str(b) + "社区的核心节点"]
                if a_kernel == b_kernel:
                    T[i,j] = 0
                else:
                    T[i,j] = 1
        C_G = []
        C_W = []
        for i in range(0,batch_size):
            p_wx1 = 0
            p_wy1 = 0
            p_wy2 = 0
            for j in range(0,161):
                p_wx1 = p_wx1 + torch.matmul(W_train[i][j],Generator_W[i][j])
                p_wy1 = p_wy1 + torch.matmul(W_train[i][j],W_train[i][j])
                p_wy2 = p_wy2 + torch.matmul(Generator_W[i][j],Generator_W[i][j])
            p_w = p_wx1*p_wx1/p_wy1*p_wy2
            W_num = len(batch_C_Kernel_node['C' + str(i + 1)])
            G_num = len(batch_C_Kernel_node1['C'+str(i+1)])
            for k in range(0,W_num):
                C_W.append(batch_C_Kernel_node['C' + str(i + 1)]['C' + str(k+1)+'社区的核心节点'])
            for k in range(0,G_num):
                C_G.append(batch_C_Kernel_node1['C' + str(i + 1)]['C' + str(k+1)+'社区的核心节点'])
            chazhi = len(C_W) - len(C_G)
            if chazhi >0:
                for cz_i in range(0,chazhi):
                    C_G.append(0)
            else:
                for cz_i in range(0,abs(chazhi)):
                    C_W.append(0)
            a = mutual_info_score(C_W, C_G)
            b = entropy(C_W)
            c = entropy(C_G)
            p_c = 2*a/(b+c)
            p = (p_w+p_c)/2
            P_all.append(p)


        T_sum = T.sum(axis=0)
        T_sum = T_sum/batch_size



        import pandas as pd
        import numpy as np

        # df = pd.DataFrame(list(T_sum), columns=['list'])
        # df.to_excel("重要度分数_list.xlsx")
        T_sum = pd.read_excel("重要度分数_list.xlsx")['list'].tolist()

        best_gene_idx = np.zeros(45)
        best_brain_idx = np.zeros(116)
        best_gene_imp = np.zeros(45)
        best_brain_imp = np.zeros(116)
        j = 0
        k = 0

        for i in range(0,116):
            best_brain_idx[i] = i
            best_brain_imp[i] = T_sum[i]
        for i in range(116,161):

            best_gene_idx[i-116] = i
            best_gene_imp[i-116] = T_sum[i]

        np.save('best_gene_idx.npy', best_gene_idx)
        np.save('best_brain_idx.npy', best_brain_idx)
        np.save('best_gene_imp.npy', best_gene_imp)
        np.save('best_brain_imp.npy', best_brain_imp)
        np.save('best_idx.npy', best_idx)
        np.save('best_imp.npy', best_imp)

        real_out = D.forward(feat_train,W_train)
        d_loss_real = criterion(real_out, label_train)  # 真实的图片标签即为1，与图片本身的内容无关
        fake_out= D.forward(feat_train,Generator_W)
        d_loss_fake = criterion(fake_out, label_train)
        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # 训练生成器
        g_out = D.forward(feat_train,Generator_W)
        g_label = np.ones(W_train.size(0))
        g_label = torch.LongTensor(g_label)
        g_loss = criterion(g_out, g_label)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        # 保存每批数据的损失
        D_loss.append(d_loss)
        print('D_loss={:.4f}'.format(sum(D_loss) / len(D_loss)))

    # 每10个为一个epoch
    if (epoch + 1) % 2 == 0:
        D.eval()

        output = D.forward(feat_test, ad_test_W)
        acc_val = utils.accuracy(output, test_label)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Test set results:",
              "accuracy= {:.4f}".format(acc_val.item()))
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        acc_list.append(float(acc_val.item()))
        torch.save(D.state_dict(), 'discriminator.pth')
        torch.save(G.state_dict(), 'generator.pth')
        ACC, SEN, SPE, AUC, TPR, FPR = utils.predict_indicators(output, test_label)
        print("ACC=", ACC)
        print("SEN=", SEN)
        print("SPE=", SPE)
        print("AUC=", AUC)
        print("tpr=", TPR)
        print("fpr=", FPR)
np.save("ACC.npy", ACC)
np.save("SEN.npy", SEN)
np.save("SPE.npy", SPE)
np.save("AUC.npy", AUC)
np.save("TPR.npy", TPR)
np.save("FPR.npy", FPR)
np.save("loss.npy", D_loss)
print("best accuracy={:.4f}".format(max_acc))


