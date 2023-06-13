import numpy as np
import torch
from typing import Optional
from scipy.optimize import linear_sum_assignment
import random


def cluster_accuracy(y_true, y_predicted, cluster_number: Optional[int] = None):
    """
    Calculate clustering accuracy after using the linear_sum_assignment function in SciPy to
    determine reassignments.

    :param y_true: list of true cluster numbers, an integer array 0-indexed
    :param y_predicted: list  of predicted cluster numbers, an integer array 0-indexed
    :param cluster_number: number of clusters, if None then calculated from input
    :return: reassignment dictionary, clustering accuracy
    """
    if cluster_number is None:
        cluster_number = (
            max(y_predicted.max(), y_true.max()) + 1
        )  # assume labels are 0-indexed
    count_matrix = np.zeros((cluster_number, cluster_number), dtype=np.int64)
    for i in range(y_predicted.size):
        count_matrix[y_predicted[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(count_matrix.max() - count_matrix)
    reassignment = dict(zip(row_ind, col_ind))
    accuracy = count_matrix[row_ind, col_ind].sum() / y_predicted.size
    return reassignment, accuracy


def target_distribution(batch: torch.Tensor) -> torch.Tensor:
    """
    Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
    Xie/Girshick/Farhadi; this is used the KL-divergence loss function.

    :param batch: [batch size, number of clusters] Tensor of dtype float
    :return: [batch size, number of clusters] Tensor of dtype float
    """
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()
    
def get_pairwise_constraints(Y: torch.Tensor, idxes: list)->torch.Tensor:
    """
    Get the pairwise constraints for semi-supervised learning. This function originates from
    https://github.com/yongzx/SDEC-Keras/issues, where the user sunbicheng gave the answer below.
    """
    n = Y.shape[0]
    a = torch.zeros((n, n))
    for index1, index2 in idxes:
        if (Y[index1] == Y[index2]):
            a[index1, index2] = 1
        else:
            a[index1, index2] = -1
    a = torch.tril(a,-1) + torch.triu(a, 1)
    return a

def semi_sup_loss(Z:torch.Tensor, Y:torch.Tensor, idxes: list, lambd=1e-5)->torch.Tensor:
        n = Z.shape[0]
        a = get_pairwise_constraints(Y,idxes)
        diff = Z[np.newaxis, :, :] - Z[:, np.newaxis, :]
        res = torch.sum(torch.matmul(a.cpu(), torch.sum(torch.square(diff.cpu()),axis=2).T))
        return torch.mul(res,lambd / n)

