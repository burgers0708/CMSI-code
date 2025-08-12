import torch
import torch.nn as nn
from typing import Optional
import numpy as np
import pandas as pd
import statsmodels.api as sm

def to_numpy(x):
    """convert Pytorch tensor to numpy array
    """
    return x.clone().detach().cpu().numpy()

def stein_hess(X: torch.Tensor, eta_G: float, eta_H: float, s: Optional[float] = None) -> torch.Tensor:
    r"""
    Estimates the diagonal of the Hessian of :math:`\log p(x)` at the provided samples points :math:`X`, 
    using first and second-order Stein identities.

    Parameters
    ----------
    X : torch.Tensor
        dataset X
    eta_G : float
        Coefficient of the L2 regularizer for estimation of the score.
    eta_H : float
        Coefficient of the L2 regularizer for estimation of the score's Jacobian diagonal.
    s : float, optional
        Scale for the Kernel. If ``None``, the scale is estimated from data, by default ``None``.

    Returns
    -------
    torch.Tensor
        Estimation of the score's Jacobian diagonal.
    """
    n, d = X.shape
    X_diff = X.unsqueeze(1) - X
    if s is None:
        D = torch.norm(X_diff, dim=2, p=2)
        s = D.flatten().median()
    K = torch.exp(-torch.norm(X_diff, dim=2, p=2) ** 2 / (2 * s ** 2)) / s

    nablaK = -torch.einsum('kij,ik->kj', X_diff, K) / s ** 2
    G = torch.matmul(torch.inverse(K + eta_G * torch.eye(n)), nablaK)

    nabla2K = torch.einsum('kij,ik->kj', -1 / s ** 2 + X_diff ** 2 / s ** 4, K)
    return -G ** 2 + torch.matmul(torch.inverse(K + eta_H * torch.eye(n)), nabla2K)

def Stein_score(X, eta_G, s = None):
    n, d = X.shape
    X_diff = X.unsqueeze(1)-X
    if s is None:
        D = torch.norm(X_diff, dim=2, p=2)
        # s = D.flatten().median()
        s = max(D.flatten().median(), 1e-8)
    K = torch.exp(-torch.norm(X_diff, dim=2, p=2)**2 / (2 * s**2)) / s
    nablaK = -torch.einsum('kij,ik->kj', X_diff, K) / s**2
    # A = torch.inverse(K + eta_G * torch.eye(n))
    A = torch.pinverse(K + eta_G * torch.eye(n))
    G = torch.matmul(A, nablaK)
    return G


def Stein_hess_diag(X, eta_G, eta_H, s = None):
    """
    Estimates the diagonal of the Hessian of log p_X at the provided samples points
    """
    n, d = X.shape
    
    X_diff = X.unsqueeze(1)-X
    if s is None:
        D = torch.norm(X_diff, dim=2, p=2)
        s = D.flatten().median()
    K = torch.exp(-torch.norm(X_diff, dim=2, p=2)**2 / (2 * s**2)) / s
    
    nablaK = -torch.einsum('kij,ik->kj', X_diff, K) / s**2
    G = torch.matmul(torch.inverse(K + eta_G * torch.eye(n)), nablaK)
    
    nabla2K = torch.einsum('kij,ik->kj', -1/s**2 + X_diff**2/s**4, K)
    return -G**2 + torch.matmul(torch.inverse(K + eta_H * torch.eye(n)), nabla2K)


def Stein_hess_col(X_diff, G, K, v, s, eta, n):
    """
    See https://arxiv.org/pdf/2203.04413.pdf Section 2.2 and Section 3.2 (SCORE paper)
        Args:
            X_diff (tensor): X.unsqueeze(1)-X difference in the NxD matrix of the data X
            G (tensor): G stein estimator 
            K (tensor): evaluated gaussian kernel
            s (float): kernel width estimator
            eta (float): regularization coefficients
            n (int): number of input samples

        Return:
            Hess_v: estimator of the v-th column of the Hessian of log(p(X))
    """
    Gv = torch.einsum('i,ij->ij', G[:,v], G)
    nabla2vK = torch.einsum('ik,ikj,ik->ij', X_diff[:,:,v], X_diff, K) / s**4
    nabla2vK[:,v] -= torch.einsum("ik->i", K) / s**2
    Hess_v = -Gv + torch.matmul(torch.inverse(K + eta * torch.eye(n)), nabla2vK)

    return Hess_v


def Stein_hess_matrix(X, s, eta):
    """
    Compute the Stein Hessian estimator matrix for each sample in the dataset

    Args:
        X: N x D matrix of the data
        s: kernel width estimate
        eta: regularization coefficient

    Return:
        Hess: N x D x D hessian estimator of log(p(X))
    """
    n, d = X.shape
    
    X_diff = X.unsqueeze(1)-X
    K = torch.exp(-torch.norm(X_diff, dim=2, p=2)**2 / (2 * s**2)) / s
    
    nablaK = -torch.einsum('ikj,ik->ij', X_diff, K) / s**2
    G = torch.matmul(torch.inverse(K + eta * torch.eye(n)), nablaK)
    
    # Compute the Hessian by column stacked together
    Hess = Stein_hess_col(X_diff, G, K, 0, s, eta, n) # Hessian of col 0
    Hess = Hess[:, None, :]
    for v in range(1, d):
        Hess = torch.hstack([Hess, Stein_hess_col(X_diff, G, K, v, s, eta, n)[:, None, :]])
    
    return Hess


def heuristic_kernel_width(X):
    """
    Estimator of width parameter for gaussian kernel

    Args:
        X (tensor): N x D matrix of the data

    Return: 
        s(float): estimate of the variance in the kernel
    """
    X_diff = X.unsqueeze(1)-X
    D = torch.norm(X_diff, dim=2, p=2)
    s = D.flatten().median()
    return s
def estimate_residuals(X, alpha, gamma, n_cv):
    """
    Estimate the residuals.
    For each variable X_j, regress X_j on all the remainig varibales of X, and estimate the residuals
    Return: 
        n x d matrix of the residuals estimates
    """
    R = []
    for i in range(X.shape[1]):
        explainatory = np.hstack([X[:, 0:i], X[:, i+1:]])
        response = X[:, i].numpy()
        from rffridge import RFFRidgeRegression
        clf = RFFRidgeRegression(rff_dim=20)
        clf.fit(explainatory, response)
        pred  = clf.predict(explainatory)
        R_i = response - pred
        R.append(R_i)
    return np.vstack(R).transpose()
def pred_err(X, Y, alpha, gamma, n_cv):
    err = []
    _, d = Y.shape
    for col in range(d):
        response = Y[:, col]
        explainatory = X[:, col].reshape(-1, 1)
        from rffridge import RFFRidgeRegression
        clf = RFFRidgeRegression(rff_dim=20)
        clf.fit(explainatory, response)
        pred = clf.predict(explainatory)
        res = response-pred
        mse = (res**2).mean().item()
        err.append(mse)
    return err

def regression_pvalues(X, y):
    X = sm.add_constant(X)
    glm = sm.GLM(y, X, family=sm.families.Gaussian())
    glm_results = glm.fit()
    return glm_results.pvalues[1:]

def top_order(X, eta_G, alpha, gamma, n_cv):
    _, d = X.shape
    top_order = []

    remaining_nodes = list(range(d))
    np.random.shuffle(remaining_nodes) # account for trivial top order
    for _ in range(d-1):
        S = Stein_score(X[:, remaining_nodes], eta_G=eta_G)
        R = estimate_residuals(X[:, remaining_nodes], alpha, gamma, n_cv)
        err = pred_err(R, S, alpha, gamma, n_cv)
        leaf = np.argmin(err)
        l_index = remaining_nodes[leaf]
        top_order.append(l_index)
        remaining_nodes = remaining_nodes[:leaf] + remaining_nodes[leaf+1:]
        print(_)

    top_order.append(remaining_nodes[0])
    return top_order[::-1]
    