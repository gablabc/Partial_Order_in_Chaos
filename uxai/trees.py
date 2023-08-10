import ctypes
import glob
import pandas as pd
import sklearn.ensemble as se
import numpy as np
from functools import partial
from tqdm import tqdm
import os
from shap.explainers import Tree

from .partial_orders import PartialOrder, RashomonPartialOrders
from .utils_optim import solve_bilinear, solve_lp



def interventional_treeshap(model, foreground, background, I_map=None):
    """ 
    Compute the Interventional Shapley Values with the TreeSHAP algorithm

    Parameters
    ----------
    model : model_object
        The tree based machine learning model that we want to explain. XGBoost, LightGBM, CatBoost, Pyspark
        and most tree-based scikit-learn models are supported.

    foreground : numpy.array or pandas.DataFrame
        The foreground dataset is the set of all points whose prediction we wish to explain.

    background : numpy.array or pandas.DataFrame
        The background dataset to use for integrating out missing features in the coallitional game.

    I_map : List(int), default=None
        A mapping from column to high-level feature. This is useful when feature are one-hot-encoded
        but you really want a single Shapley value for each group of columns. For example,
        `I_map = [0, 1, 2, 2, 2]` treats the last three columns as an encoding of the same feature. 
        Therefore we would return 3 Shapley values. Setting `I_map`
        to None will yield one Shapley value per column.
    """

    # Extract tree structure with the SHAP API
    ensemble = Tree(model, data=background).model
    
    # All numpy arrays must be C_CONTIGUOUS
    assert ensemble.thresholds.flags['C_CONTIGUOUS']
    assert ensemble.features.flags['C_CONTIGUOUS']
    assert ensemble.children_left.flags['C_CONTIGUOUS']
    assert ensemble.children_right.flags['C_CONTIGUOUS']

    # All arrays must be C-Contiguous and DataFrames are not.
    if type(foreground) == pd.DataFrame:
        foreground = np.ascontiguousarray(foreground)
    if type(background) == pd.DataFrame:
        background = np.ascontiguousarray(background)

    # Mapping from column to partition index
    if I_map is None:
        I_map = np.arange(foreground.shape[1]).astype(np.int32)
    else:
        I_map = I_map.astype(np.int32)
    
    # Shapes
    Nt = ensemble.features.shape[0]
    n_features = np.max(I_map) + 1
    depth = ensemble.features.shape[1]
    Nx = foreground.shape[0]
    Nz = background.shape[0]

    # Values at each leaf
    values = np.ascontiguousarray(ensemble.values[...,-1])
    # For Random Forests SHAP automatically scales the tree
    # by 1 / Nt. We multiply by Nt so that averaging SHAP
    # values across all trees will lead to the right result
    if type(model) in  [se._forest.RandomForestRegressor,
                        se._forest.RandomForestClassifier,
                        se._forest.ExtraTreesRegressor,
                        se._forest.ExtraTreesClassifier]:
        values *= Nt

    # Where to store the output
    results = np.zeros((Nx, n_features, Nt))

    ####### Wrap C / Python #######

    # Find the shared library, the path depends on the platform and Python version    
    project_root = os.path.dirname(__file__).split('uxai')[0]
    libfile = glob.glob(os.path.join(project_root, 'build', '*', 'treeshap*.so'))[0]

    # Open the shared library
    mylib = ctypes.CDLL(libfile)

    # Tell Python the argument and result types of function main_treeshap
    mylib.main_int_treeshap.restype = ctypes.c_int
    mylib.main_int_treeshap.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, 
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
    mylib.main_int_treeshap(Nx, Nz, Nt, foreground.shape[1], depth, foreground, background, 
                            I_map, ensemble.thresholds, values,
                            ensemble.features, ensemble.children_left, ensemble.children_right, results)

    return results, ensemble



def min_Hm(phis, M_min):
    """
    Minimize a linear functional phi over 
    H_{m:} for m=M_min,M_min+1,..., M

    Parameters
    ----------
    phis: (..., M) `np.array`
        Linear functionals to optimize
    M_min: `int`
        Positive Integer lower than `M`

    Returns
    -------
    optim_values: (..., M-M_min+1) `np.array`
        For each functional return the min values for each value of m
    """
    m = np.arange(M_min, phis.shape[-1]+1)
    m = m.reshape([1]*(phis.ndim - 1) + [len(m)])
    cherry_picked_min = np.partition(phis, kth=M_min)
    phis_min = np.concatenate( (np.sum(cherry_picked_min[..., :M_min], axis=-1, keepdims=True), 
                                np.sort(cherry_picked_min[..., M_min:], axis=-1)), axis=-1)
    phis_min = np.cumsum(phis_min, axis=-1) / m
    return phis_min



def min_max_Hm(phis, M_min):
    """
    Minimize and Maximize a linear functional phi over 
    H_{m:} for m=M_min,M_min+1,..., M

    Parameters
    ----------
    phis: (..., M) `np.array`
        Linear functionals to optimize
    M_min: `int`
        Positive Integer lower than `M`

    Returns
    -------
    optim_values: (..., M-M_min+1, 2) `np.array`
        For each functional return the min/max values for each m
    """
    phis_min =  min_Hm( phis, M_min)
    phis_max = -min_Hm(-phis, M_min)
    return np.stack((phis_min, phis_max), axis=-1)



def get_all_tree_preds(X, ensemble, task='regression'):
    """ 
    Return predictions of all trees in the tree ensemble

    Parameters
    ----------
    X: (n_samples, n_features) `np.array`
        Samples on which to predict
    ensemble: `sklearn class`
        A class from `sklearn.ensemble._forest` that represents
        a forest of trees.
    task: `str`, default='regression'
        The type of task: regression, clasification etc.

    Returns
    -------
    all_preds: (n_samples, len(ensemble)) `np.array`
        Predicted values on each sample for all trees.
    """
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
    """ 
    Compute the upper-bound `epsilon^+(m)` for `m=M_min, ..., M`.

    Parameters
    ----------
    all_preds: (n_samples, len(ensemble)) `np.array`
        Predicted values on each sample for all trees.
    y: (n_samples,) `np.array`
        Labels for each data point.
    task: `str`, default='regression'
        The type of task: regression, clasification etc.
    M_min: `int`, default=None
        Miminum value considered for m. If `M_min=None`, it is
        set to `M // 2`.

    Returns
    -------
    m_range: `np.array`
        List of values for m.
    epsilon_upper: `np.array`
        Value of the upper bound for each m.
    """
    M = all_preds.shape[1]
    if M_min is None:
        M_min = M // 2
    
    # Min-Max predictions
    all_preds = min_max_Hm(all_preds, M_min) #(n_instances, M-M_min, 2)

    if task == "regression":
        errors_upper = np.sqrt( np.mean(np.max( (all_preds - y.reshape((-1, 1, 1))) ** 2, axis=-1), axis=0))
    else:
        errors_upper = np.mean( np.max( (all_preds>0.5) != y.reshape((-1, 1, 1)), axis=-1), axis=0)
    return np.arange(M_min, M+1), errors_upper



def get_minmax_attribs(epsilon, epsilon_upper, all_phis):
    m_idx_epsilon = np.argmax(epsilon_upper <= epsilon)
    return all_phis[..., m_idx_epsilon, :]




class RandomForestRashomon(object):
    """
    Rashomon Set for Random Forests.

    RandomForestRashomon takes an already fitted RandomForest from `sklearn` 
    and compute its Rashomon Set by only keeping a subset of trees.
    Conceptualizing the Rashomon Set that way allows to compute exact
    consensus on local/global feature importance statements.


    Attributes
    ----------
    m : (M-M_min+1,) `np.array`
        Possible number of trees in the RashomonSet

    epsilon_upper : (M-M_min+1,) `np.array`
        For each m, the associated upper bound `epsilon^+(m)`

    """
    def __init__(self, model, task="regression"):
        # We wrap the class around the sklearn one
        self.model = model
        self.M = model.n_estimators
        self.task = task


    def fit(self, X_test, y_test, M_min=50):
        """
        Fit the Rashomon Set on top of the RF model.

        Parameters
        ----------
        X_test : (n_samples, n_features) `np.array`
            Input values.
        y_test : (n_samples,) `np.array`
            Target values.

        Returns
        -------
        self : object
            Fitted Rashomon Set Estimator.
        
        """
        self.M_min = M_min
        preds = get_all_tree_preds(X_test, self.model, self.task)
        self.m, self.epsilon_upper = \
            epsilon_upper_bound(preds, y_test, M_min=M_min, task=self.task)

        return self


    def get_m_epsilon(self, epsilon):
        m_epsilon_idx = np.argmax(self.epsilon_upper < epsilon)
        m_epsilon = self.m[m_epsilon_idx]
        return m_epsilon
    

    def predict(self, X, epsilon=None):
        """ 
        Return point prediction of whole forest and [Min, Max] preds 
        of Rashomon Set

        Parameters
        ----------
        X: (n_samples, n_features) `np.array`
            Samples on which to predict
        epsilon: `float`, default=None
            Rashomon parameter. If it is provided, then the function also
            returns the min/max predictions on each sample.

        Returns
        -------
        y: (n_samples,) `np.array`
            Predicted values.
        minmax_preds: (n_samples, 2) `np.array`
            Minimum and Maximum predictions over the Rashomon Set.
        """
        

        if epsilon is None:
            if self.task == "regression":
                return self.model.predict(X)
            else:
                return self.model.predict_proba(X)[:, 1]
        tree_preds = get_all_tree_preds(X, self.model, self.task)
        mean_pred = tree_preds.mean(1)
        m_epsilon = self.get_m_epsilon(epsilon)

        # Cherry pick m_epsilon trees with lowest/largest pred
        cherry_picked_min =  np.partition( tree_preds, kth=m_epsilon)[:, :m_epsilon].mean(1)
        cherry_picked_max = -np.partition(-tree_preds, kth=m_epsilon)[:, :m_epsilon].mean(1)
        return mean_pred, np.column_stack((cherry_picked_min, cherry_picked_max))


    # def partial_dependence(self, x, idx, epsilon):
    #     x_ = (x - self.X_mean[idx]) / self.X_std[idx]

    #     # Compute min/max preds
    #     a = np.zeros((self.n_features + 1, len(x)))
    #     a[np.array(idx)+1] = x_.T
    #     A_half_inv = self.A_half_inv * np.sqrt(epsilon)
    #     minmax_preds = opt_lin_ellipsoid(a, A_half_inv, self.w_hat, return_input_sol=False)
    #     return self.y_std * minmax_preds + self.y_mean



    def feature_importance(self, phis, epsilon, feature_names, expand=False, threshold=0):
    
        N, d, _ = phis.shape
        m_epsilon = self.get_m_epsilon(epsilon)

        # Attribution of the total forest
        GFI_mean = np.mean(np.abs(phis.mean(-1)), 0)
        # Min/max of each feature attribution
        attrib_min_idx = np.argpartition( phis, kth=m_epsilon)[..., :m_epsilon] # (N, d, m_epsilon)
        attrib_max_idx = np.argpartition(-phis, kth=m_epsilon)[..., :m_epsilon] # (N, d, m_epsilon)
        attrib_min = phis[np.arange(N).reshape((-1, 1, 1)), 
                          np.arange(d).reshape((1, -1, 1)), 
                          attrib_min_idx].mean(-1)
        attrib_max = phis[np.arange(N).reshape((-1, 1, 1)), 
                          np.arange(d).reshape((1, -1, 1)), 
                          attrib_max_idx].mean(-1)

        assert (attrib_min <= attrib_max).all()

        # # Create a collection of models with extreme local attribs
        # collection_idx = np.vstack((attrib_min_idx.reshape((N*d, -1)),
        #                             attrib_max_idx.reshape((N*d, -1))))
        # collection_idx = np.unique(np.sort(collection_idx, axis=1), axis=0)
        # if collection_idx.shape[0] > 1000:
        #     collection_idx = collection_idx[np.random.choice(range(len(collection_idx)), 1000, replace=False)]
        
        # # Compute all LFA on all models in the collection (N_collect, d)
        # GFI_collect = np.zeros((len(collection_idx), d))
        # for i, idxs in tqdm(enumerate(collection_idx), desc="Optimizing LFA"):
        #     GFI_collect[i] = np.mean(np.abs(phis[..., idxs].mean(-1)), 0)
        

        # Expand the collection of models by approx-optimizing the GFIs
        GFI_to_add = []
        min_max_GFI = np.zeros((d, 2))
        for i in tqdm(range(d), desc="Optimizing GFI"):

            ### Linear Problem (LP) to minimize importance ###
            _, z_sol = solve_lp(phis[:, i, :], m_epsilon)
            min_trees_idxs = np.where(z_sol==1)[0]
            k = m_epsilon - len(min_trees_idxs)
            # print(f"Choosing {k} trees out of fractional values")
            if k > 0:
                # There are fractionary solutions so we heuristically choose the 
                # remaining trees in descending order of z
                fraction_tree = np.where((z_sol > 0) & (z_sol < 1))[0]
                # print(f"{100*(1-len(fraction_tree)/np.sum(z_sol > 0)):.2f}% of non-null z_s are one")
                to_add = fraction_tree[np.argpartition(-z_sol[fraction_tree], kth=k)[:k]]
                min_trees_idxs = np.concatenate((min_trees_idxs, to_add))
            # Add argmin to collection
            GFI_to_add.append(np.mean(np.abs(phis[..., min_trees_idxs].mean(-1)), 0))

            ### Bilinear Problem (BP) to maximize importance ###
            # To fix an initial solution, we look at all trees that appear in the most
            # extreme attributions min/max phi_i(h, x^(j)). Then we sample with higher 
            # probability the trees which appeared most often
            pos_greater_neg = np.where(np.abs(attrib_max[:, i]) > np.abs(attrib_min[:, i]))[0]
            candidates = attrib_min_idx[:, i, :]
            candidates[pos_greater_neg] = attrib_max_idx[pos_greater_neg, i, :]
            candidates, counts = np.unique(candidates.ravel(), return_counts=True)
            for _ in range(5):
                z_0 = np.random.choice(candidates, m_epsilon, p=counts/counts.sum(), replace=False)
                # BP Solver
                _, max_trees_idxs = solve_bilinear(phis[:, i, :], z_0)
                # Add argmax to collection
                GFI_to_add.append(np.mean(np.abs(phis[..., max_trees_idxs].mean(-1)), 0))


            # When solving for relative importance, we simply restrict ourselves to data
            # instances whose attribution has consistent sign for both i and j
            for j in range(d):
                if i < j:
                    consistent_sign_i = np.sign(attrib_min[:, i]) == np.sign(attrib_max[:, i])
                    consistent_sign_j = np.sign(attrib_min[:, j]) == np.sign(attrib_max[:, j])
                    consistent_sign_ij = consistent_sign_i & consistent_sign_j
                    s_i = np.sign(attrib_min[consistent_sign_ij, i])[:, np.newaxis]
                    s_j = np.sign(attrib_min[consistent_sign_ij, j])[:, np.newaxis]
                    relative_imp = np.mean(s_i * phis[consistent_sign_ij, i] -\
                                           s_j * phis[consistent_sign_ij, j], axis=0) # (n_trees,)
                    relative_imp_min_idx = np.argpartition( relative_imp, kth=m_epsilon)[:m_epsilon]
                    relative_imp_max_idx = np.argpartition(-relative_imp, kth=m_epsilon)[:m_epsilon]
                    GFI_to_add.append(np.mean(np.abs(phis[..., relative_imp_min_idx].mean(-1)), 0))
                    GFI_to_add.append(np.mean(np.abs(phis[..., relative_imp_max_idx].mean(-1)), 0))

        GFI_collect = np.vstack(GFI_to_add)
        
        min_max_GFI = np.zeros((d, 2))
        min_max_GFI[:, 0] = GFI_collect.min(0)
        min_max_GFI[:, 1] = GFI_collect.max(0)

        # If there exists model that dont rely on this feature, we do not
        # plot it in the PO. We only plot features that are "necessary"
        # for good performance
        ambiguous = set()
        for i in range(d):
            if min_max_GFI[i, 0] < threshold:
                ambiguous.add(i)
        
        # Compare features by features to make partial order
        select_features_idx = [i for i in range(d) if i not in ambiguous]
        adjacency = np.zeros((d, d))
        for i in select_features_idx:
            for j in select_features_idx:
                if i < j:
                    # Features are comparable
                    if (GFI_collect[:, i] < GFI_collect[:, j]).all():
                        adjacency[i, j] = 1
                    elif (GFI_collect[:, j] < GFI_collect[:, i]).all():
                        adjacency[j, i] = 1

        
        po = PartialOrder(GFI_mean, adjacency, ambiguous=ambiguous, 
                            features_names=feature_names, top_bottom=True)
        return min_max_GFI, po



    def feature_attributions(self, phis, tau=0.01):
        """ 
        Compute a RashomonPartialOrders from the Local Feature Attributions (LFA)
        returned by the TreeSHAP methods.

        Parameters
        ----------
        phis: (n_samples, n_features, n_trees) `np.array`
            Precomputed Shap values for each instance, feature and tree.
        
        Returns
        -------
        PO : RashomonPartialOrders
            Object encoding all partial orders for all instances at any tolerance level epsilon

        """
        N, d, _ = phis.shape
        # Uncertainty in the gaps
        gaps = phis.sum(1)
        all_gaps = min_max_Hm(gaps, self.M_min)
        # Identify the smallest m where the gap is still well-defined
        gap_eps = self.epsilon_upper[np.argmax(all_gaps[..., 0] * all_gaps[...,1] > 0, axis=1)]

        ### Range of the Attributions across the Rashomon Set ###
        phis_ref = phis.mean(2) #(N, d)
        all_phis = min_max_Hm(phis, self.M_min) #(N, d, M-M_min, 2)
        
        minmax_attribs_lambda = partial(get_minmax_attribs, epsilon_upper=self.epsilon_upper, 
                                                            all_phis=all_phis)

        # Determine critical epsilon values for positive, 
        # negative attribution statements
        pos_eps = np.zeros((N, d))
        neg_eps = np.zeros((N, d))
        for j in range(d):
            pos_idx = phis_ref[:, j] > tau
            pos_eps[pos_idx, j] = self.epsilon_upper[np.argmax(all_phis[pos_idx, j, :, 0] >  tau, axis=1)]
            neg_idx = phis_ref[:, j] < -tau
            neg_eps[neg_idx, j] = self.epsilon_upper[np.argmax(all_phis[neg_idx, j, :, 1] < -tau, axis=1)]


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
                    min_max_ij_diff = min_max_Hm(s_i * phis[:, i, :] - 
                                                 s_j * phis[:, j, :], self.M_min)

                    # Critical epsilon for comparing relative importance
                    ij_idx = ref_ij_diff < 0
                    adjacency_eps[ij_idx, i, j] = self.epsilon_upper[np.argmax(min_max_ij_diff[ij_idx, :, 1] < 0, axis=1)]
                    ji_idx = ref_ij_diff > 0
                    adjacency_eps[ji_idx, j, i] = self.epsilon_upper[np.argmax(min_max_ij_diff[ji_idx, :, 0] > 0, axis=1)]

        PO = RashomonPartialOrders(phis_ref, minmax_attribs_lambda, gap_eps, 
                                    neg_eps, pos_eps, adjacency_eps, True)
        return PO
