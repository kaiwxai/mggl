import os
import zipfile
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from numpy import zeros, array
from math import sqrt, log
from scipy.spatial import distance
from scipy.stats import entropy

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset1, dataset2, dataset3):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.dataset3 = dataset3

    def __len__(self):
        return len(self.dataset1) # 数据集的长度

    def __getitem__(self, idx):
        data1 = self.dataset1[idx] # 从第一个数据集中获取数据
        data2 = self.dataset2[idx] # 从第二个数据集中获取数据
        data3 = self.dataset3[idx] # 从第二个数据集中获取数据
        return data1, data2, data3 # 返回两个数据集的元素

def mmd_loss(x, y, sigma):
    # MMD
    k_xx = torch.exp(-torch.pow(torch.cdist(x, x), 2) / (2 * sigma**2))
    k_yy = torch.exp(-torch.pow(torch.cdist(y, y), 2) / (2 * sigma**2))
    k_xy = torch.exp(-torch.pow(torch.cdist(x, y), 2) / (2 * sigma**2))
    # Compute the MMD loss
    loss = torch.mean(k_xx) + torch.mean(k_yy) - 2 * torch.mean(k_xy)
    return loss

    
# def JSD(P, Q):
#     """Compute the Jensen-Shannon divergence between two probability distributions.

#     Input
#     -----
#     P, Q : array-like
#         Probability distributions of equal length that sum to 1
#     """

#     def _kldiv(A, B):
#         return np.sum([v for v in A * np.log2(A/B) if not np.isnan(v)])
#     M = 0.5 * (P + Q)
#     return 0.5 * (_kldiv(P, M) +_kldiv(Q, M))

import scipy.stats
def get_js_divergence(p1, p2):
    """
    calculate the Jensen-Shanon Divergence of two probability distributions
    :param p1:
    :param p2:
    :return:
    """
    # normalize
    p1 = p1 / (p1.sum()+1e-9)
    p2 = p2 / (p2.sum()+1e-9)
    m = (p1 + p2) / 2
    js = 0.5 * scipy.stats.entropy(p1, m) + 0.5 * scipy.stats.entropy(p2, m)
    return js

def KL_divergence(p, q):
    """ Compute KL divergence of two vectors, K(p || q)."""
    return sum(p[x] * log((p[x]) / (q[x])) for x in range(len(p)) if p[x] != 0.0 or p[x] != 0)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def metric_func(pred, y, times):
    result = {}

    result['MSE'], result['RMSE'], result['MAE'], result['MAPE'], result['JSD'] = np.zeros(times), np.zeros(times), np.zeros(times), np.zeros(times), np.zeros(times)

    # print("metric | pred shape:", pred.shape, " y shape:", y.shape)
    
#     def cal_MAPE(pred, y):
#         diff = np.abs(np.array(y) - np.array(pred))
#         return np.mean(diff / y)
    def masked_smape(predicted, actual, null_val=np.nan):
        return np.mean(2.0 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted)))
                       
    for i in range(times):
        y_i = y[:,i,:]
        pred_i = pred[:,i,:]
        MSE = mean_squared_error(pred_i, y_i)
        RMSE = mean_squared_error(pred_i, y_i) ** 0.5
        MAE = mean_absolute_error(pred_i, y_i)
        MAPE = masked_smape(pred_i, y_i)
#         print(pred_i.shape)
#         print(y_i.shape)
        JSD = get_js_divergence(pred_i.reshape(-1, 1)[:, 0], y_i.reshape(-1, 1)[:, 0])
        result['MSE'][i] += MSE
        result['RMSE'][i] += RMSE
        result['MAE'][i] += MAE
        result['MAPE'][i] += MAPE
        result['JSD'][i] += JSD
    return result

def result_print(result, info_name='Evaluate'):
#     total_MSE, total_RMSE, total_MAE, total_MAPE = result['MSE'], result['RMSE'], result['MAE'], result['MAPE']
#     print("========== {} results ==========".format(info_name))
#     print(" MAE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_MAE[0], total_MAE[1], total_MAE[2], total_MAE[3], total_MAE[4], total_MAE[5]))
#     print("MAPE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_MAPE[0] * 100, total_MAPE[1] * 100, total_MAPE[2] * 100, total_MAPE[3] * 100, total_MAPE[4] * 100, total_MAPE[5] * 100))
#     print("RMSE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_RMSE[0], total_RMSE[1], total_RMSE[2], total_RMSE[3], total_RMSE[4], total_RMSE[5]))
#     print("---------------------------------------")
    try:
        total_MSE, total_RMSE, total_MAE, total_MAPE, total_JSD = result['MSE'], result['RMSE'], result['MAE'], result['MAPE'], result['JSD']
        print("========== {} results ==========".format(info_name))
        print(" MAE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_MAE[0], total_MAE[1], total_MAE[2], total_MAE[3], total_MAE[4], total_MAE[5]))
        print("MAPE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_MAPE[0] * 100, total_MAPE[1] * 100, total_MAPE[2] * 100, total_MAPE[3] * 100, total_MAPE[4] * 100, total_MAPE[5] * 100))
        print("RMSE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_RMSE[0], total_RMSE[1], total_RMSE[2], total_RMSE[3], total_RMSE[4], total_RMSE[5]))
        print("JSD: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_JSD[0], total_JSD[1], total_JSD[2], total_JSD[3], total_JSD[4], total_JSD[5]))
        print("---------------------------------------")
    except:
        total_MSE, total_RMSE, total_MAE, total_MAPE, total_JSD = result['MSE'], result['RMSE'], result['MAE'], result['MAPE'], result['JSD']
        print("========== {} results ==========".format(info_name))
        print(" MAE: %.4f"%(total_MAE[0]))
        print("MAPE: %.4f"%(total_MAPE[0] * 100))
        print("RMSE: %.4f"%(total_RMSE[0]))
        print("JSD: %.4f"%(total_JSD[0]))
        print("---------------------------------------")

    

    if info_name == 'Best':
        print("========== Best results ==========")
        print(" MAE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_MAE[0], total_MAE[1], total_MAE[2], total_MAE[3], total_MAE[4], total_MAE[5]))
        print("MAPE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_MAPE[0] * 100, total_MAPE[1] * 100, total_MAPE[2] * 100, total_MAPE[3] * 100, total_MAPE[4] * 100, total_MAPE[5] * 100))
        print("RMSE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_RMSE[0], total_RMSE[1], total_RMSE[2], total_RMSE[3], total_RMSE[4], total_RMSE[5]))
        print("---------------------------------------")


def load_data(dataset_name, stage):
    print("INFO: load {} data @ {} stage".format(dataset_name, stage))

    A = np.load("data/" + dataset_name + "/matrix.npy")
    A = get_normalized_adj(A)
    A = torch.from_numpy(A)
    X = np.load("data/" + dataset_name + "/dataset.npy")
    X = X.transpose((1, 2, 0))
    X = X.astype(np.float32)

    # Normalization using Z-score method
    means = np.mean(X, axis=(0, 2))
    X = X - means.reshape(1, -1, 1)
    stds = np.std(X, axis=(0, 2))
    X = X / stds.reshape(1, -1, 1)

    # train: 70%, validation: 10%, test: 20%
    # source: 100%, target_1day: 288, target_3day: 288*3, target_1week: 288*7
    if stage == 'train':
        X = X[:, :, :int(X.shape[2]*0.7)]
    elif stage == 'validation':
        X = X[:, :, int(X.shape[2]*0.7):int(X.shape[2]*0.8)]
    elif stage == 'test':
        X = X[:, :, int(X.shape[2]*0.8):]
    elif stage == 'source':
        X = X
    elif stage == 'target_1day':
        X = X[:, :, :288]
    elif stage == 'target_3day':
        X = X[:, :, :288*3]
    elif stage == 'target_1week':
        X = X[:, :, :288*7]
    else:
        print("Error: unsupported data stage")

    print("INFO: A shape is {}, X shape is {}, means = {}, stds = {}".format(A.shape, X.shape, means, stds))

    return A, X, means, stds


def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave


def generate_dataset(X, num_timesteps_input, num_timesteps_output, means, stds):
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features,
    num_timesteps)
    :return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_input).
        - Node targets for the samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_output).
    """
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1)]

    # Save samples
    features, target = [], []
    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose(
                (0, 2, 1)))
#         target.append(X[:, 0, i + num_timesteps_input: j]*stds[0]+means[0])
        target.append(X[:, 0, i + num_timesteps_input: j])
    return torch.from_numpy(np.array(features)), \
           torch.from_numpy(np.array(target))


def aggregate_dynamic_graphs(dynamic_graphs, decay_lambda, windows_size):
    num_graphs = len(dynamic_graphs)
    # Initialize aggregated graph
    nx_graph = dgl.to_networkx(dynamic_graphs[:-1])
    aggregated_graph = nx.to_numpy_array(nx_graph)
    print(aggregated_graph.shape)

    for t in range(len(dynamic_graphs)-windows_size, len(dynamic_graphs)):
        nx_graph = dgl.to_networkx(dynamic_graphs[t])
        adj_matrix = nx.to_numpy_array(nx_graph)
        decay = np.exp([decay_lambda * (len(dynamic_graphs)-t-1)])   # Calculate decay value and unsqueeze to match dimensions
        print(adj_matrix.shape)
        aggregated_graph += decay * adj_matrix  # Aggregate dynamic graph
    graph = dgl.from_scipy(sp.csr_matrix(aggregated_graph))
    return graph