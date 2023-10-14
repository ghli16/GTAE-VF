import argparse
import pandas as pd
import networkx as nx
import numpy as np
import random
import numpy as np
import torch
from torch_geometric.data import Data,Batch
from torch_geometric.loader import DataLoader
import os
import csv
import time


def get_metrics(real_score, predict_score):
    real_score, predict_score = real_score.flatten(), predict_score.flatten()
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num*np.arange(1, 1000)/1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1)-TP
    FN = real_score.sum()-TP
    TN = len(real_score.T)-TP-FP-FN

    fpr = FP/(FP+TN)
    tpr = TP/(TP+FN)

    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]

    # np.savetxt(roc_path.format(i), ROC_dot_matrix)

    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5*(x_ROC[1:]-x_ROC[:-1]).T*(y_ROC[:-1]+y_ROC[1:])

    recall_list = tpr
    precision_list = TP/(TP+FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]

    # np.savetxt(pr_path.format(i), PR_dot_matrix)

    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5*(x_PR[1:]-x_PR[:-1]).T*(y_PR[:-1]+y_PR[1:])

    f1_score_list = 2*TP/(len(real_score.T)+TP-TN)
    accuracy_list = (TP+TN)/len(real_score.T)
    specificity_list = TN/(TN+FP)
    # plt.plot(x_ROC, y_ROC)
    # plt.plot(x_PR,y_PR)
    # plt.show()
    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    #print( ' auc:{:.4f} ,aupr:{:.4f},f1_score:{:.4f}, accuracy:{:.4f}, recall:{:.4f}, specificity:{:.4f}, precision:{:.4f}'.format( auc[0, 0],aupr[0, 0], f1_score, accuracy, recall, specificity, precision))
    return [auc[0, 0], aupr[0, 0], f1_score, accuracy, recall, specificity, precision]
    # mm = [auc[0, 0], aupr[0, 0], f1_score, accuracy, recall, specificity, precision]
    # print(f"auc = {mm[0]}")
    # return mm, auc[0, 0]


def set_seed(args):
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed((args.seed))
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def get_data(path):
    matrix_list = []

    with open(path, 'r') as file:
        lines = file.readlines()
        num_rows = None
        matrix = []
        for line in lines:
            line = line.strip()
            if line:
                if num_rows is None:
                    num_rows = int(line)
                else:
                    row = list(map(float, line.split(',')))
                    matrix.append(row)
                    if len(matrix) == num_rows:
                        matrix_list.append(np.array(matrix))
                        matrix = []
                        num_rows = None
            if len(matrix_list) == 1000:
                break
    return matrix_list

def feature300():
    path1 = "../dataSet/afterExtractFeature300/testN.csv"
    path2 = "../dataSet/afterExtractFeature300/testP.csv"
    path3 = "../dataSet/afterExtractFeature(train)/trainN_features.csv"
    path4 = "../dataSet/afterExtractFeature(train)/trainP_features.csv"

    #listN1 = get_data(path1)
    #listP1 = get_data(path2)
    listN2 = get_data(path3)
    listP2 = get_data(path4)
    listP2 = listP2[:1000]
    listN2 = listN2[:1000]

    merged_list = listP2 + listN2
    return merged_list

def graph489():
    path5 = "../dataSet/Graph/graph_N.csv"
    path6 = "../dataSet/Graph/graph_P.csv"
    path7 = "../dataSet/Graph(train)/graph_N.csv"
    path8 = "../dataSet/Graph(train)/graph_P.csv"

    #listN1 = get_data(path5)
    #listP1 = get_data(path6)
    listN2= get_data(path7)
    listP2 = get_data(path8)

    #listN = listN1[:300]
    #listP = listP1[:300]
    listN2 = listN2[:1000]
    listP2 = listP2[:1000]
    merged_list = listP2 + listN2
    return merged_list

def data_chuli():
    start_time = time.time()

    G_list = graph489()


    node_features_list = feature300()

    graph_labels = [1] * 1000 + [0] * 1000

    data_list = [Data(x=torch.tensor(node_features_list[i], dtype=torch.float),
                      edge_index=torch.tensor(np.column_stack(np.where(G_list[i])), dtype=torch.long).t().contiguous(),
                      y=torch.tensor(graph_labels[i], dtype=torch.long))
                 for i in range(len(G_list))]
    # set_seed()
    random.shuffle(data_list)
    original_indices = list(range(len(data_list)))


    random.shuffle(original_indices)



    with open('./index_notsort.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(original_indices)

    end_time = time.time()  # 记录结束时间
    elapsed_time =(end_time - start_time)/60
    print(f"Data creation took {elapsed_time:.2f} minutes")
    return data_list






