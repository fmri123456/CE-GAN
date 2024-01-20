import math
import numpy as np


def all_matrix(bra_ge_net,bra_ge_feat):
    W = bra_ge_net.numpy()
    feat = bra_ge_feat.numpy()
    all_C = {}  # 所有人脑区基因网络划分为社团网络的集合
    all_Vertex = {}  # 所有人社团网络内的点的集合
    all_C_H = {}  # 所有人社团网络内点的特征矩阵
    all_C_Hc = {}  # 所有人社团的特征矩阵
    all_C_Vertex_Node_degree = {}  # 社团内节点的节点度
    all_C_rich_node = {}  # 社团内的富人节点
    all_C_Kernel_node = {}  # 社团内的核心节点
    all_C_W_c = {}  # 所有人关键边权重矩阵
    all_Kernel_node_list = {}
    hum_count = 10  # np.size(W, 0)  # 214个测试体
    row = np.size(W, 1)

    Node_degree_network = np.zeros(shape=(hum_count, row))  # 统计脑区基因网络的节点度  214*161
    for i in range(hum_count):  # 遍历每个人的脑区基因网络，先遍历第一个人的脑区基因网络
        person_W = W[i]
        for j in range(row):
            a = person_W[j]
            a_not_zero = int(np.count_nonzero(a))  # 遍历脑区基因每一行的节点度，即元素非0个数
            Node_degree_network[i, j] = a_not_zero

    # 构建所有有关社区的矩阵
    for i in range(hum_count):  # 遍历每个人的脑区基因网络的节点度，先遍历第一个人
        C_name = 'C' + str(i + 1)
        all_C[C_name] = {}
        all_Vertex[C_name] = {}
        all_C_H[C_name] = {}
        all_C_Hc[C_name] = {}
        all_C_Vertex_Node_degree[C_name] = {}
        all_C_rich_node[C_name] = {}
        all_C_Kernel_node[C_name] = {}
        all_Kernel_node_list[C_name + '的核心节点'] = {}

    # 构建 all_C，all_Vertex，all_C_H
    for i in range(hum_count):  # 遍历每个人的脑区基因网络的节点度，先遍历第一个人
        print("开始构建第" + str(i + 1) + "个人的社区网络")
        person_W = W[i]  # 第i个人的脑区基因网络
        person_H = feat[i]  # 第i个人的特征信息

        C_ed = []  # 第i个人的脑区基因的节点已经被划分为社团的点
        person_jiedian = Node_degree_network[i]  # 第i个人的脑区基因的节点度
        list_person = list(person_jiedian)  # 第i个人的脑区基因的节点度list
        # 对节点按节点度大小进行排序
        numpy_person = np.array(list_person)
        numpy_person_index = numpy_person.argsort()
        numpy_person_index = np.flipud(numpy_person_index)  # 排序结果

        Vc = list(numpy_person_index)

        k = 1
        while len(Vc) > 0:
            print(Vc)
            print("开始构建第" + str(i + 1) + "个人的" + "第" + str(k) + "个社区网络")
            Cp = []  # 存放第k个社区的所有节点
            vertex_temp = Vc[0]  # Vc[0]始终是第k个社区节点度最大的节点，也就是社区的第一个节点
            del_Q = 1
            Q2 = 0
            while del_Q > 0:
                Cp.append(vertex_temp)
                Nei_cp = []  # 社区每新增一个点，Nei_cp需要更新为0
                for vertex in Cp:  # 遍历社团内的点，找到与社团内的点所有相连的点
                    person_W_hang = list(np.nonzero(person_W[vertex])[0])  # 与vertex相邻的点
                    Nei_cp.extend(person_W_hang)  # 将与vertex相邻的点添加值Nei_cp中
                Nei_cp = list(set(Nei_cp) - set(Cp))  # 与 Cp里面的点所有有连接的点
                Nei_cp = list(set(Nei_cp) - set(C_ed))  # 去除之前已经加入社团的点，确保两个社团内的点互不重叠

                p_edge = []
                edge = []
                if len(Nei_cp) != 0:
                    # 找到社区内的点和社团外的点所有能连接的边
                    for row_vertex in Cp:
                        for col_vertex in Nei_cp:
                            point_edge = str(col_vertex)
                            p_edge.append(point_edge)
                            edge.append(person_W[row_vertex, col_vertex])  # 两点所成边的值的大小

                    value = max(edge)  # 与社团Cp连接最紧密的点所构成边的值
                    max_edge_idx = edge.index(value)  # 与社团Cp连接最紧密的点的下标
                    max_point_idx = int(p_edge[max_edge_idx])
                    vertex_temp = max_point_idx
                    e_in = 0
                    e_out = 0
                    for row_vertex in Cp:
                        for col_vertex in Cp:
                            if person_W[row_vertex, col_vertex] != 0:
                                e_in = e_in + 1
                    e_in = (e_in - len(Cp)) / 2 + len(Cp)  # 社团Cp内的边数

                    Without_Cp = [i for i in range(161)]
                    Without_Cp = list(set(Without_Cp) - set(Cp))  # 社团外所有的点

                    for row_vertex in Cp:
                        for col_vertex in Without_Cp:  # 找到社团内的点与社团外的点有多少连接的边
                            if person_W[row_vertex, col_vertex] != 0:
                                e_out = e_out + 1

                    Q1 = Q2
                    Q2 = e_in / (e_in + e_out)
                    del_Q = Q2 - Q1
                else:
                    del_Q = 0
            # 至此循环结束，确定了第i个人的第k个社区网络内的所有点
            print("while循环结束")
            C_ed.extend(Cp)  # 第i个人的第k个社区网络内的所有点
            print(Cp)
            Cp_Node_degree = []
            for d in Cp:
                Cp_d_link = list(np.nonzero(person_W[d])[0])  # 与Cp[i]相邻的点
                Cp_Node_degree.append(len(Cp_d_link))

            Cp_net = []
            rrow = 0
            Cp_feat = np.zeros(shape=(len(Cp), 70))
            for Cp_row in Cp:
                for Cp_col in Cp:
                    Cp_net.append(person_W[Cp_row, Cp_col])
            Cp_net = np.array(Cp_net)
            Cp_net.resize([len(Cp), len(Cp)])

            for Cp_feat_row in Cp:
                Cp_feat[rrow] = person_H[Cp_feat_row]
                rrow = rrow + 1

            C_name = 'C' + str(i + 1)
            Cp_name = 'C' + str(k) + '社区'
            all_C[C_name][Cp_name] = Cp_net
            all_C_Vertex_Node_degree[C_name][Cp_name] = Cp_Node_degree
            all_Vertex[C_name][Cp_name] = Cp
            Cp_feat_name = 'C' + str(k) + '社区的特征矩阵'
            all_C_H[C_name][Cp_feat_name] = Cp_feat
            Vc = list(set(Vc) - set(Cp))
            k = k + 1

    for i in range(hum_count):
        C_name = 'C' + str(i + 1)
        k = len(all_C[C_name])
        flag = 0  # 标记位

        kernel_node_list = []
        for k_i in range(k):
            Cp_name = 'C' + str(k_i + 1) + '社区'
            C_Kernel_name = 'C' + str(k_i + 1) + '社区的核心节点'
            a = np.array(all_C_Vertex_Node_degree[C_name][Cp_name])
            a_max_index = list(np.where(a == np.max(a))[0])
            rich_node = []  # 每个社区开始确定核心节点和富人节点之前都要设置为空
            if len(a_max_index) >= 2:
                flag = 1
                for i in range(len(a_max_index)):
                    rich_node.append(all_Vertex[C_name][Cp_name][a_max_index[i]])
                all_C_rich_node[C_name][Cp_name] = rich_node

            else:
                all_C_Kernel_node[C_name][C_Kernel_name] = int(all_Vertex[C_name][Cp_name][a_max_index[0]])
                kernel_node_list.append(all_Vertex[C_name][Cp_name][a_max_index[0]])
                rich_node.append(all_Vertex[C_name][Cp_name][a_max_index[0]])
                all_C_rich_node[C_name][Cp_name] = rich_node

        if flag == 1:
            for k_i in range(k):
                Cp_name = 'C' + str(k_i + 1) + '社区'
                C_Kernel_name = 'C' + str(k_i + 1) + '社区的核心节点'
                if len(all_C_rich_node[C_name][Cp_name]) > 1:
                    rich_node_list = all_C_rich_node[C_name][Cp_name]
                    rich_node_count_list = []
                    for rich_list_node in rich_node_list:
                        count = 0
                        for kernel_list_node in kernel_node_list:
                            if W[i][rich_list_node][kernel_list_node] != 0:
                                count = +1
                        rich_node_count_list.append(count)

                    max_node_count_index = rich_node_count_list.index(max(rich_node_count_list))
                    all_C_Kernel_node[C_name][C_Kernel_name] = int(rich_node_list[max_node_count_index])
                    kernel_node_list.append(rich_node_list[max_node_count_index])
        all_Kernel_node_list[C_name + '的核心节点'] = kernel_node_list
        W_c = np.zeros(shape=(k, k))
        for k_index_i in range(k):
            for k_index_j in range(k):
                W_c[k_index_i][k_index_j] = W[i][all_C_Kernel_node[C_name]['C' + str(k_index_i + 1) + '社区的核心节点']][
                    all_C_Kernel_node[C_name]['C' + str(k_index_j + 1) + '社区的核心节点']]

        all_C_W_c[C_name + '的关键边权重矩阵'] = W_c

    # 删除社团之间非核心节点之间的连接
    for i in range(hum_count):
        C_name = 'C' + str(i + 1)
        k = len(all_C[C_name])
        for k_i in range(k - 1):
            Cp_name1 = 'C' + str(k_i + 1) + '社区'
            C_Kernel_name1 = 'C' + str(k_i + 1) + '社区的核心节点'
            for k_j in range(k_i + 1, k):
                Cp_name2 = 'C' + str(k_j + 1) + '社区'
                C_Kernel_name2 = 'C' + str(k_j + 1) + '社区的核心节点'
                C_i = all_Vertex[C_name][Cp_name1]
                C_j = all_Vertex[C_name][Cp_name2]
                C_i_kernel = all_C_Kernel_node[C_name][C_Kernel_name1]
                C_j_kernel = all_C_Kernel_node[C_name][C_Kernel_name2]

                for m in C_i:
                    for n in C_j:
                        if m != C_i_kernel and n != C_j_kernel:
                            W[i][m][n] = 0
    # 计算社团的社团熵
    hum_count = len(all_C)
    all_E_C = {}
    all_E_V = {}
    all_P_a = {}
    all_P_b = {}
    for i in range(hum_count):
        C_name = 'C' + str(i + 1)
        c_count = len(all_C[C_name])
        EC = np.zeros(shape=(c_count, c_count))
        alpha = 0.3
        P_a = np.ones(shape=(c_count, c_count)) * alpha
        for k in range(c_count):
            Cp_name = 'C' + str(k + 1) + '社区'
            Com_vertex = all_Vertex[C_name][Cp_name]  # 第k个社团的节点
            Com_W = all_C[C_name][Cp_name]
            global E_Ca
            n_a = len(Com_vertex)  # 第k个社团的节点数
            EC_a, E_v = 0, []
            e = int(np.count_nonzero(Com_W))
            for j in range(n_a):
                node_j = Com_vertex[j]
                node_j_degree = int(np.count_nonzero(W[i][node_j]))
                E_Ca = EC_a + node_j_degree / (e) * math.log2((node_j_degree) / (e))
            EC[k][k] = E_Ca
        all_E_C[C_name + '的社团的社团熵信息矩阵'] = EC
        all_P_a[C_name + '的社团的社团熵信息传播系数矩阵'] = P_a

        node_count = np.size(W[i], 1)
        beta = 0.3
        P_b = np.ones(shape=(node_count, node_count)) * beta

        EV = np.zeros(shape=(node_count, node_count))
        Ki_all = np.sum(Node_degree_network[i])
        for j in range(node_count):
            Dj = Node_degree_network[i][j] / Ki_all
            kernel_node_list = all_Kernel_node_list[C_name + '的核心节点']
            if j in kernel_node_list:
                for j_in_which_C in range(c_count):
                    if j in all_Vertex[C_name]['C' + str(j_in_which_C + 1) + '社区']:
                        EV[j][j] = EC[j_in_which_C][j_in_which_C]
            else:
                EV[j][j] = -Dj * math.log(Dj)
        all_E_V[C_name + '的节点熵信息矩阵'] = EV
        all_P_b[C_name + '的节点熵信息传播系数矩阵'] = P_b

    return all_C,all_Vertex,all_C_H,all_C_Hc,all_C_Vertex_Node_degree,all_C_rich_node,all_C_Kernel_node, all_E_C,all_E_V, all_P_a, all_P_b,all_C_W_c,W


