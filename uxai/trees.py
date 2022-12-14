import sklearn.ensemble as se
import numpy as np
import subprocess
from .partial_orders import RashomonPartialOrders
import time
import os
from functools import partial

from shap.explainers import Tree
from shap.maskers import Independent

import ctypes
import numpy as np
import glob
import pandas as pd


def min_Hm(phis, M_min, M):
    """
    Minimize a linear functional phi over 
    H_{m:} for m=M_min,M_min+1,..., M
    """
    m = np.arange(M_min, M+1)
    cherry_picked_min = np.partition(phis, kth=M_min)
    phis_min = np.hstack( (np.sum(cherry_picked_min[:, :M_min], axis=1, keepdims=True), 
                            np.sort(cherry_picked_min[:, M_min:], axis=1)) )
    phis_min = np.cumsum(phis_min, axis=1) / m
    return phis_min



def min_max_Hm(phis, M_min, M):
    """
    Minimize and Maximize a linear functional phi over 
    H_{m:} for m=M_min,M_min+1,..., M
    """
    phis_min = min_Hm(phis, M_min, M)
    phis_max = -min_Hm(-phis, M_min, M)
    return np.stack((phis_min, phis_max), axis=2)



def all_tree_preds(X, ensemble, task='regression'):
    if type(ensemble) not in [se._forest.RandomForestRegressor,
                              se._forest.RandomForestClassifier,
                              se._forest.ExtraTreesRegressor,
                              se._forest.ExtraTreesClassifier]:
        raise TypeError("The tree ensemble provided is not valid !!!")
        
    # This line is done in parallel in sklearn
    all_leafs = ensemble.apply(X) # (n_samples, n_estimators)
    all_preds = np.zeros((X.shape[0], len(ensemble)))

    # This loop could be parallelized
    for j, estimator in enumerate(ensemble.estimators_):
        if task == "regression":
            values = estimator.tree_.value.ravel()
            all_preds[:, j] = values[all_leafs[:, j]]
        elif task=="classification":
            values = estimator.tree_.value[:, 0]
            values = values / np.sum(values, axis=1, keepdims=True)
            all_preds[:, j] = values[all_leafs[:, j], 1]
        else:
            raise NotImplementedError()
    return all_preds



def epsilon_upper_bound(all_preds, y, task="regression", M_min=None):
    M = all_preds.shape[1]
    if M_min is None:
        M_min = M // 2
    
    # Min-Max predictions
    all_preds = min_max_Hm(all_preds, M_min, M)

    if task == "regression":
        errors_upper = np.sqrt( np.mean(np.max( (all_preds - y.reshape((-1, 1, 1))) ** 2, axis=-1), axis=0))
    else:
        errors_upper = np.mean( np.max( (all_preds>0.5) != y.reshape((-1, 1, 1)), axis=-1), axis=0)
    return np.arange(M_min, M+1), errors_upper






def custom_treeshap(model, foreground, background, features=None, ohe=None):
    
    # Find out which ohe columns correspond to which high-level feature
    if features is not None and ohe is not None:
        n_num_features = len(features.non_nominal)
        # We assume that all numerical features come first
        categorical_to_features = list(range(n_num_features))
        counter = n_num_features
        for idx in features.nominal:
            # Associate the feature to its encoding columns
            for _ in features.maps[idx].cats:
                categorical_to_features.append(counter)
            counter += 1

    # Otherwise we dont worry about ohe features
    else:
        categorical_to_features = list(range(foreground.shape[1]))
    categorical_to_features = np.array(categorical_to_features, dtype=np.int32)
    n_features = categorical_to_features[-1] + 1

    # Use the SHAP API to extract the tree structure
    mask = Independent(background, max_samples=len(background))
    ensemble = Tree(model, data=mask).model
    
    # All numpy arrays must be C_CONTIGUOUS
    assert ensemble.thresholds.flags['C_CONTIGUOUS']
    assert ensemble.features.flags['C_CONTIGUOUS']
    assert ensemble.children_left.flags['C_CONTIGUOUS']
    assert ensemble.children_right.flags['C_CONTIGUOUS']

    if type(foreground) == pd.DataFrame:
        foreground = np.ascontiguousarray(foreground)
    if type(background) == pd.DataFrame:
        background = np.ascontiguousarray(background)

    # Shape properties
    Nx = foreground.shape[0]
    Nz = background.shape[0]
    Nt = ensemble.features.shape[0]
    d = foreground.shape[1]
    depth = ensemble.features.shape[1]

    # Recale values since TreeSHAP divides values by n_tree
    values = np.ascontiguousarray(ensemble.values[...,1]) * Nt

    # Where to store the output
    results = np.zeros((Nx, Nt, n_features))

    ####### Wrap C / Python #######

    # Find the shared library, the path depends on the platform and Python version
    project_root = os.path.dirname(__file__).split('uxai')[0]
    libfile = glob.glob(os.path.join(project_root, 'build', '*', 'treeshap*.so'))[0]

    # Open the shared library
    mylib = ctypes.CDLL(libfile)

    # Tell Python the argument and result types of function main_treeshap
    mylib.main_treeshap.restype = ctypes.c_int
    mylib.main_treeshap.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, 
                                    ctypes.c_int, ctypes.c_int,
                                    np.ctypeslib.ndpointer(dtype=np.float64),
                                    np.ctypeslib.ndpointer(dtype=np.float64),
                                    np.ctypeslib.ndpointer(dtype=np.int32),
                                    np.ctypeslib.ndpointer(dtype=np.float64),
                                    np.ctypeslib.ndpointer(dtype=np.float64),
                                    np.ctypeslib.ndpointer(dtype=np.int32),
                                    np.ctypeslib.ndpointer(dtype=np.int32),
                                    np.ctypeslib.ndpointer(dtype=np.int32),
                                    np.ctypeslib.ndpointer(dtype=np.float64)]

    # 3. call function mysum
    mylib.main_treeshap(Nx, Nz, Nt, d, depth, foreground, background, categorical_to_features,
                        ensemble.thresholds, values, ensemble.features, ensemble.children_left, 
                        ensemble.children_right, results)

    return results  # (N, d)



def get_minmax_attribs(epsilon, epsilon_upper, all_phis):
        m_idx_epsilon = np.argmax(epsilon_upper <= epsilon)
        return all_phis[:, m_idx_epsilon, ...]



def tree_attributions(model, foreground, background, epsilon_upper, features=None, ohe=None):

    M = len(model.estimators_)
    M_min = M // 2
    N = foreground.shape[0]
    if features is None:
        d = foreground.shape[1]
    else:
        d = len(features.names)
    foreground_preds = all_tree_preds(foreground, model, task="classification")# (N, T)
    background_preds = all_tree_preds(background, model, task="classification")# (N, T)
    gaps = foreground_preds - background_preds.mean(0)
    all_gaps = min_max_Hm(gaps, M_min, M)
    # Identify the smallest m where the gap is still well-defined
    gap_eps = np.zeros(N)
    gap_eps = epsilon_upper[np.argmax(all_gaps[..., 0] * all_gaps[...,1] > 0, axis=1)]

    ################ TreeSHAP ###############
    phis = custom_treeshap(model, foreground, background, features, ohe)# Shape (N, T, d)

    ### Range of the Attributions across the Rashomon Set ###
    phis_ref = phis.mean(1) #(N, d)
    all_phis = np.zeros((N, M-M_min+1, d, 2))
    for i in range(d):
        all_phis[:, :, i, :] = min_max_Hm(phis[..., i], M_min, M)
    
    minmax_attribs_lambda = partial(get_minmax_attribs, epsilon_upper=epsilon_upper, all_phis=all_phis)


    tau = 0.01
    # Determine critical epsilon values for positive, 
    # negative attribution statements
    pos_eps = np.zeros((N, d))
    neg_eps = np.zeros((N, d))
    for j in range(d):
        pos_idx = phis_ref [:, j] > tau
        pos_eps[pos_idx, j] = epsilon_upper[np.argmax(all_phis[pos_idx, :, j, 0] > tau, axis=1)]
        neg_idx = phis_ref [:, j] < -tau
        neg_eps[neg_idx, j] = epsilon_upper[np.argmax(all_phis[neg_idx, :, j, 1] < -tau, axis=1)]


    # Compare features by features to make partial order
    adjacency_eps = np.zeros((N, d, d))
    for i in range(d):
        for j in range(d):
            if i < j:
                # Reference difference in importance
                ref_ij_diff = np.abs(phis_ref[:, i]) - np.abs(phis_ref[:, j])
                s_i = np.sign(phis_ref[:, [i]])
                s_j = np.sign(phis_ref[:, [j]])
                
                # Min-max difference in importance
                min_max_ij_diff = min_max_Hm(s_i * phis[..., i] - s_j * phis[..., j], M_min, M)

                # Critical epsilon for comparing relative importance
                ij_idx = ref_ij_diff < 0
                adjacency_eps[ij_idx, i, j] = epsilon_upper[np.argmax(min_max_ij_diff[ij_idx, :, 1] < 0, axis=1)]
                ji_idx = ref_ij_diff > 0
                adjacency_eps[ji_idx, j, i] = epsilon_upper[np.argmax(min_max_ij_diff[ji_idx, :, 0] > 0, axis=1)]

    PO = RashomonPartialOrders(phis_ref, minmax_attribs_lambda, gap_eps, 
                                neg_eps, pos_eps, adjacency_eps, True)
    return PO
