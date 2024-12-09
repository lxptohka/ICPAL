import itertools
import copy
import math
import numpy as np
import matplotlib.pyplot as plt
import csv

min_prev = 0.3  # 最小参与度
R = 10  # 设最小邻近关系的距离为10


def drawData(E, R):
    """
    可视化函数，方便查看数据之间的关系
    :param E: 空间实例集
    :param R: 空间邻近关系
    """
    values = [x[2] for x in E]
    x = np.array(values)[:, 0]
    y = np.array(values)[:, 1]
    n = [x[0] + str(x[1]) for x in E]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.scatter(x, y)
    for i, txt in enumerate(n):
        ax.annotate(txt, (x[i], y[i]))
    for i in range(len(values) - 1):
        for j in range(i + 1, len(values)):
            if math.sqrt((values[j][0] - values[i][0]) ** 2 + (values[j][1] - values[i][1]) ** 2) < R:
                ax.plot([values[i][0], values[j][0]], [values[i][1], values[j][1]])
    plt.show()


def getCountET(E, ET):
    """
    获取每个特征类型的实例出现次数
    :param E: 空间实例集
    :param ET: 空间特征集
    :return: 各个特征类型的实例出现次数
    """
    count = {}
    for featureType in ET:
        count[featureType] = 0
        for instance in E:
            if instance[0] == featureType:
                count[featureType] += 1

    return count


def getInstance(E, featureType, instanceID):
    """
    根据特征类型和实例 id 在实例集中查询对应的实例数据
    :param featureType: 实例特征类型
    :param instanceID: 实例id
    :return: 实例数据
    """
    for item in E:
        if item[0] == featureType and item[1] == instanceID:
            return item


def isNeighbor(instance1, instance2, R):
    """
    判断两个实例之间是否相邻
    :param instance1: 第一个实例
    :param instance2: 第二个实例
    :param R: 空间邻近关系
    :return: bool值，表示这两个实例是否满足邻近关系
    """
    distance = math.sqrt(math.pow(instance1[2][0] - instance2[2][0], 2)
                         + math.pow(instance1[2][1] - instance2[2][1], 2))
    if distance <= R:
        return 1
    else:
        return 0


def calParticipationIndex(Pattern, count_ET):
    """
    # 计算候选模式的参与度
    :param Pattern: 候选模式
    :param count_ET: 各个特征类型的实例出现次数
    :return: 候选模式的参与度
    """
    participation_index = 1  # 设置一个最大参与率
    for featureType in Pattern.keys():
        participation_ratio = Pattern[featureType] / count_ET[featureType]
        if participation_ratio < participation_index:  # 返回最低参与率，即参与度
            participation_index = participation_ratio

    return participation_index


def createT2(E, R):
    """
    根据空间实例集 E 和空间邻近关系 R 生成2阶表实例
    :param E: 空间实例集
    :param R: 空间邻近关系
    :return: 2阶表实例
    """
    T2 = []
    for i in range(len(E) - 1):
        for j in range(i+1, len(E)):
            if E[i][0] != E[j][0] and isNeighbor(E[i], E[j], R):
                instance = []
                instance.append(E[i][0] + '.' + str(E[i][1]))
                instance.append(E[j][0] + '.' + str(E[j][1]))
                T2.append(instance)
    T2_dic = table2Tdic(T2)

    return T2_dic


def createC2(ET):
    """
    生成二阶候选模式
    :param ET: 空间特征集
    :return: 二阶候选模式
    """
    ET_len = len(ET)
    C2 = []
    for i in range(ET_len-1):
        for j in range(i+1, ET_len):
            C2.append([ET[i], ET[j]])

    return C2


def table2Tdic(T):
    """
    将表实例如 [A1,B2] 转换为表实例字典 { A,B : [A1,B2] }
    :param T:  表实例集
    :return: 表实例字典
    """
    table_instance = {}
    for row in T:
        co_location = ""
        row_instance = []
        row_len = len(row)
        for i in range(row_len):
            if i != row_len - 1:
                co_location += str(row[i]).split('.')[0] + ','
            else:
                co_location += str(row[i]).split('.')[0]
            row_instance.append(str(row[i]))
        if co_location not in table_instance.keys():
            table_instance[co_location] = [row_instance]
        else:
            table_instance[co_location].append(row_instance)

    return table_instance


def T2Tk(T, E):
    """
    由 k-1 阶满足参与度的表实例集生成 k 阶表实例集
    :param T: k-1 阶满足参与度的表实例集
    :param E: 数据集
    :return: k 阶表实例集
    """
    T_list = []
    keys = list(T.keys())
    keys.sort()
    keys_len = len(keys)
    for i in range(keys_len - 1):
        for j in range(i + 1, keys_len):
            if keys[j].split(',')[:-1] == keys[i].split(',')[:-1]:
                for instance_i in T[keys[i]]:
                    for instance_j in T[keys[j]]:
                        if instance_i[:-1] == instance_j[:-1]:
                            instance_k_1 = instance_i[-1]
                            instance_k_2 = instance_j[-1]

                            featureType_k_1, instanceID_k_1 = instance_k_1.split('.')
                            featureType_k_2, instanceID_k_2 = instance_k_2.split('.')

                            instance_k_1 = getInstance(E, featureType_k_1, int(instanceID_k_1))
                            instance_k_2 = getInstance(E, featureType_k_2, int(instanceID_k_2))

                            if isNeighbor(instance_k_1, instance_k_2, R):
                                temp = instance_i
                                temp.append(instance_j[-1])
                                T_list.append(temp)
            else:
                break
    T_dic = table2Tdic(T_list)

    return T_dic


def Tk2Pk(T_dic, count_ET, min_prev):
    """
    由 k 阶表实例集和 k 阶候选集以及最小参与度生成 k 阶频繁项集和满足参与度的表实例集
    :param T_dic: k 阶表实例集
    :param count_ET: 各个特征类型的实例出现次数
    :param min_prev: 最小参与度阈值
    :return: k 阶频繁项集和满足参与度的表实例集
    """
    Pk = []
    prevalent_T = {}
    keys = list(T_dic.keys())
    keys.sort()
    for key in keys:
        key_list = key.split(",")
        p = {}
        for i in range(len(key_list)):
            p[key_list[i]] = len(set([row_instance[i] for row_instance in T_dic[key]]))
        participation_index = calParticipationIndex(p, count_ET)
        if participation_index >= min_prev:
            prevalent_T[key] = T_dic[key]
            Pk.append(key_list)

    return prevalent_T, Pk


def P2Ck(P):
    """
    由k阶频繁模式 P,生成 k+1 阶候选模式 C
    :param P: k阶频繁模式
    :return: k+1 阶候选模式 C
    """
    candidate = []
    p_len = len(P)
    for i in range(p_len):
        for j in range(i + 1, p_len):
            if P[j][:-1] == P[i][:-1]:
                temp = P[i].copy()
                temp.append(P[j][-1])
                candidate.append(temp)

    return candidate


def get_prevalent_patterns(filename):
    """
    获取数据集的频繁模式
    :param filename: 数据文件
    :return: 频繁模式集
    """
    E = []
    with open(filename, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            if len(row[0]) > 1:
                continue
            temp = [row[0], int(row[1]), (float(row[2]), float(row[3]))]
            E.append(temp)
    del E[0]

    ET = []
    for data in E:
        if data[0] not in ET:
            ET.append(data[0])

    # drawData(E, R)
    countET = getCountET(E, ET)
    C2 = createC2(ET)
    # print("C2")
    # print(C2)
    T2 = createT2(E, R)
    # print("T2")
    # for key in T2.keys():
    #     print(key, T2[key])
    T, P = Tk2Pk(T2, countET, min_prev)
    # print("P2")
    # print(P)

    prevalentPatterns = []
    for item in P:
        prevalentPatterns.append(item)
    k = 2
    while True:
        C = P2Ck(P)
        if len(C) == 0:
            break
        # print("C{}".format(k+1))
        # print(C)

        T_C = T2Tk(T, E)
        # if bool(T_C):
        #     print("T{}".format(k+1))
        #     for i in T_C:
        #         print(i, T_C[i])

        T, P = Tk2Pk(T_C, countET, min_prev)
        if bool(P):
            for item in P:
                prevalentPatterns.append(item)
            # print("P{}".format(k+1))
            # print(P)

        k += 1
    # print("all co-locations")
    # print(prevalentPatterns)

    return prevalentPatterns


# if __name__ == '__main__':
#     E = []
#     with open('data.csv', 'r', encoding='utf-8') as csvfile:
#         csvreader = csv.reader(csvfile)
#         for row in csvreader:
#             if len(row[0]) > 1:
#                 continue
#             temp = [row[0], int(row[1]), (float(row[2]), float(row[3]))]
#             E.append(temp)
#     del E[0]
#
#     # drawData(E, R)
#     countET = getCountET(E, ET)
#     C2 = createC2(ET)
#     # print("C2")
#     # print(C2)
#     T2 = createT2(E, R)
#     # print("T2")
#     # for key in T2.keys():
#     #     print(key, T2[key])
#     T, P = Tk2Pk(T2, countET, min_prev)
#     print("P2")
#     print(P)
#
#     prevalentPatterns = []
#     for item in P:
#         prevalentPatterns.append(item)
#     k = 2
#     while True:
#         C = P2Ck(P)
#         if len(C) == 0:
#             break
#         # print("C{}".format(k+1))
#         # print(C)
#
#         T_C = T2Tk(T, E)
#         # if bool(T_C):
#         #     print("T{}".format(k+1))
#         #     for i in T_C:
#         #         print(i, T_C[i])
#
#         T, P = Tk2Pk(T_C, countET, min_prev)
#         if bool(P):
#             for item in P:
#                 prevalentPatterns.append(item)
#             print("P{}".format(k+1))
#             print(P)
#
#         k += 1
#     print("all co-locations")
#     print(prevalentPatterns)

