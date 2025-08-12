import pandas as pd
from kneed import KneeLocator
import numpy as np
import torch
from score_estimator import Stein_score, estimate_residuals, pred_err
from typing import Union, Tuple
import math
import numpy as np

def _find_elbow(d: int, diff_dict: dict, hard_thres: float = 30, online: bool = True) -> np.ndarray:
    r"""
    Return selected shifted nodes by finding elbow point on sorted variance

    Parameters
    ----------
    diff_dict : dict
        A dictionary where ``key`` is the index of variables/nodes, and ``value`` is its variance ratio.
    hard_thres : float, optional
        | Variance ratios larger than hard_thres will be directly regarded as shifted. 
        | Selected nodes in this step will not participate in elbow method, by default 30.
    online : bool, optional
        If ``True``, the heuristic will find a more aggressive elbow point, by default ``True``.

    Returns
    -------
    np.ndarrayeerrre
        A dict with selected nodes and corresponding variance
    """
    diff = pd.DataFrame()
    diff.index = diff_dict.keys()
    diff["ratio"] = [x for x in diff_dict.values()]
    shift_node_part1 = diff[diff["ratio"] >= hard_thres].index
    undecide_diff = diff[diff["ratio"] < hard_thres]
    kn = KneeLocator(range(undecide_diff.shape[0]), undecide_diff["ratio"].values, S=0.8, 
                     curve='convex', direction='decreasing',online=online,interp_method="interp1d")
    
    allKnee = sorted(kn.all_knees)
    Knee_1_3 = d//3
    target_knee = 0  
    minestKnee = [i for i in range(1, math.ceil(d//20)+1)] 
    for knee in allKnee:
        if knee < Knee_1_3:
            target_knee = knee
            continue
        if knee >= Knee_1_3 and knee < Knee_1_3*2:
            if (target_knee in minestKnee) or target_knee == 0:
                target_knee = knee
            elif target_knee >= Knee_1_3 and knee < Knee_1_3*2:
                target_knee = knee
            else:
                continue
        if knee >= Knee_1_3*2:
            if target_knee == 0:
                target_knee = knee
            elif (target_knee in minestKnee) and undecide_diff.iloc[knee-2, undecide_diff.columns.get_loc("ratio")] > 1.3:
                target_knee = knee
            else:
                continue
    shift_node_part2 = undecide_diff.index[:target_knee]
    shift_node = np.concatenate((shift_node_part1, shift_node_part2))
    return shift_node
def _get_min_rank_sum(HX: torch.Tensor, HY: torch.Tensor) -> int:
    order_X = torch.argsort(torch.from_numpy(np.array(HX)))
    rank_X = torch.argsort(order_X)
    order_Y = torch.argsort(torch.from_numpy(np.array(HY)))
    rank_Y = torch.argsort(order_Y)
    
    l = int((rank_X + rank_Y).argmin())
    return l
def est_node_shifts_TopK(
        k, 
        dataset,
        eta_G: float = 0.005,
        elbow: bool = True,
        elbow_thres: float = 30.,
        elbow_online: bool = True,
        verbose: bool = False,
    ) -> Tuple[list, list, dict]:
    vprint = print if verbose else lambda *a, **k: None
    dataTorch = []
    for data in dataset:
        if type(data) is np.ndarray:
            dataTorch.append(torch.from_numpy(np.float32(data)))
    n, d = dataTorch[0].shape
    order = [] # estimates a valid topological sort
    active_nodes = list(range(d))
    shifted_nodes = [] # list of estimated shifted nodes
    dict_stats = dict() # dictionary of variance ratios for all nodes
    A = torch.cat(dataTorch)
    for _ in range(d-1):
        actDataTorch = []
        for dataT in dataTorch:
            actDataTorch.append(dataT[:, active_nodes])
        actA = A[:, active_nodes]
        S = []
        R = []
        err = []
        for actDataT in actDataTorch:
            S_X = Stein_score(actDataT, eta_G=eta_G) # (500, d)-tensor
            R_X = estimate_residuals(actDataT, alpha=0.1, gamma=0.1, n_cv=5)
            err_X = pred_err(R_X, S_X, alpha=0.1, gamma=0.1, n_cv=5)  # pred_err_GPU
            S.append(S_X)
            R.append(R_X)
            err.append(err_X)
        S_A = Stein_score(actA, eta_G=eta_G)
        R_A = estimate_residuals(actA, alpha=0.1, gamma=0.1, n_cv=5)
        err_A = pred_err(R_A, S_A, alpha=0.1, gamma=0.1, n_cv=5)
        err_sum = [0 for _ in range(len(err[0]))]
        for i in range(len(err[0])):
            for errl in err:
                err_sum[i] += errl[i]
        l = err_sum.index(min(err_sum))
        minE = 9999999
        for errl in err:
            if errl[l] < minE:
                minE = errl[l]
        if minE == 0:
            dict_stats[active_nodes[l]] = err_A[l] / 1e-10
        else :
            dict_stats[active_nodes[l]] = err_A[l] / minE
        order.append(active_nodes[l])
        active_nodes.pop(l)
    
    order.append(active_nodes[0])
    order.reverse()
    dict_stats = dict(sorted(dict_stats.items(), key=lambda item: -item[1]))
    TopK = list(dict_stats.keys())[:k]
    if elbow:
        shifted_nodes = _find_elbow(d, dict_stats, elbow_thres, elbow_online)

    return shifted_nodes, dict_stats, TopK, order

def est_node_shifts(
        X: Union[np.ndarray, torch.Tensor], 
        Y: Union[np.ndarray, torch.Tensor], 
        eta_G: float = 0.005,
        normalize_var: bool = False,
        shifted_node_thres: float = 1.0,
        elbow: bool = True,
        elbow_thres: float = 30.,
        elbow_online: bool = True,
        use_both_rank: bool = False,
        verbose: bool = False,
    ) -> Tuple[list, list, dict]:
    vprint = print if verbose else lambda *a, **k: None
    if type(X) is np.ndarray:
        X = torch.from_numpy(np.float32(X))
    if type(Y) is np.ndarray:
        Y = torch.from_numpy(np.float32(Y))
    n, d = X.shape
    order = [] # estimates a valid topological sort
    active_nodes = list(range(d))
    shifted_nodes = [] # list of estimated shifted nodes
    dict_stats = dict() # dictionary of variance ratios for all nodes
    A = torch.cat((X, Y))
    for _ in range(d-1):
        actX = X[:, active_nodes]
        actY = Y[:, active_nodes]
        actA = A[:, active_nodes]
        
        S_X = Stein_score(actX, eta_G=eta_G)
        R_X = estimate_residuals(actX, alpha=0.1, gamma=0.1, n_cv=5)
        err_X = pred_err(R_X, S_X, alpha=0.1, gamma=0.1, n_cv=5)
        S_Y = Stein_score(actY, eta_G=eta_G)
        R_Y = estimate_residuals(actY, alpha=0.1, gamma=0.1, n_cv=5)
        err_Y = pred_err(R_Y, S_Y, alpha=0.1, gamma=0.1, n_cv=5)
            
        if normalize_var:
            err_X = err_X / pd.Series(err_X).mean()
            err_Y = err_Y / pd.Series(err_Y).mean()
        
        l = _get_min_rank_sum(err_X, err_Y) if use_both_rank else int(np.array(err_Y).argmin())
        
        S_A = Stein_score(actA, eta_G=eta_G)
        R_A = estimate_residuals(actA, alpha=0.1, gamma=0.1, n_cv=5)
        err_A = pred_err(R_A, S_A, alpha=0.1, gamma=0.1, n_cv=5)
        
        HX_l = torch.from_numpy(np.array(err_X[l]))
        HY_l = torch.from_numpy(np.array(err_Y[l]))
        HA_l = torch.from_numpy(np.array(err_A[l]))
        vprint(f"l: {active_nodes[l]} Var_X = {HX_l} Var_Y = {HY_l} Var_Pool = {HA_l}")
        if torch.min(HX_l, HY_l) * shifted_node_thres < HA_l:
            shifted_nodes.append(active_nodes[l])
        dict_stats[active_nodes[l]] = (HA_l / torch.min(HX_l, HY_l)).numpy()
        order.append(active_nodes[l])
        active_nodes.pop(l)
    
    order.append(active_nodes[0])
    order.reverse()
    dict_stats = dict(sorted(dict_stats.items(), key=lambda item: -item[1]))
    
    if elbow:
        shifted_nodes = _find_elbow(d, dict_stats, elbow_thres, elbow_online)

    return shifted_nodes, order, dict_stats