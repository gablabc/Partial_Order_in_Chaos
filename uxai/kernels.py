""" Rashomon Sets of Kernel Ridge models """

import numpy as np
from scipy import linalg
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics import mean_squared_error

from .utils import Ellipsoid
from .utils_optim import opt_qpqc_standard_exact, opt_qpqc_standard_approx
from .partial_orders import PartialOrder, RashomonPartialOrders


def solve_cholesky(K, y, lambd):
    # w = inv(K + lambd*R*I) * y
    R = K.shape[0]
    A = K + lambd * R * np.eye(R)
    return linalg.solve(A, y, sym_pos=True, overwrite_a=False), K.dot(A/R)


def grad_rbf(X, Dict, K, gamma):
    """ 
    Compute the gradient of the RBF kernels

    Parameters
    ----------
    X: (N, d) `np.array`
        Where to compute the gradients
    Dict: (R, d) `np.array`
        Reference points
    K: (N, R) `np.array`
        Matrix whose ith row and jth column if k(x^(i), r^(j))
    gamma: `float`
        Scale parameter of the RBF kernel

    Returns
    -------
    sol_values: (N, d, R) `np.array`
        3D tensor whose element ijk is the jth component of the
        gradient of k(. , r^(k)) evaluated at x^(i)
    """
    N, d = X.shape
    R = Dict.shape[0]
    pairwise_diff = X.reshape((N, d, 1)) - Dict.T.reshape((1, d, R))
    return -2*gamma * pairwise_diff * K.reshape((N, 1, R))



def grad_poly(X, Dict, gamma, degree):
    """ 
    Compute the gradient of the Poly kernels

    Parameters
    ----------
    X: (N, d) `np.array`
        Where to compute the gradients
    Dict: (R, d) `np.array`
        Reference points
    gamma: `float`
        Scale parameter of the Poly kernel
    degree: `int`
        Polynomial degree

    Returns
    -------
    sol_values: (N, d, R) `np.array`
        3D tensor whose element ijk is the jth component of the
        gradient of k(. , r^(k)) evaluated at x^(i)
    """
    N, d = X.shape
    R = Dict.shape[0]
    params = {"gamma" : gamma, "degree" : degree-1}
    K = pairwise_kernels(X, Dict, metric="poly", filter_params=True, **params)
    ref = Dict.T.reshape((1, d, R))
    C = degree * gamma
    return C * ref * K.reshape((N, 1, R))



class KernelRashomon(RegressorMixin, BaseEstimator):
    """
    Rashomon Set for Kernel Ridge Regression.

    KernelRashomon fits a model in the RKHS associated with a symmetric
    positive definite Kernel K. The final function takes the form 
    
    `h(x) = \sum_{k=1}^R alpha_k k(x, r^(k))`
    
    and the parameters alpha are learned via regularized least square. 
    The corresponding Rashomon Set is an ellipsoid of the form

    `(alpha - alpha_S)^T A (alpha - alpha_S) <= epsilon`

    over which we can compute a consensus on local/global feature importance
    statements.


    Attributes
    ----------
    alpha_s : (R, 1) `np.array`
        Estimated alpha coefficients

    MSE : `float`
        Mean-Squared-Error of w_hat

    RMSE : `float`
        Root-Mean-Squared-Error of w_hat

    train_loss : `float`
        Loss minimized during training `MSE + lambd ||h||^2`

    """
    def __init__(self, lambd=1, kernel="rbf", gamma=None, degree=3, coef0=1,
                                                kernel_params=None, n_jobs=None
    ):
        self.lambd = lambd
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.n_jobs = n_jobs


    def get_kernel(self, X, Y=None):
        if Y is None:
            # Add small noise to Kernel to ensure 
            # it is positive definite
            eps = 1e-8 * np.eye(self.R)
        else:
            eps = 0
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma, "degree": self.degree, "coef0": self.coef0}
        # Compute the kernel using the sklearn implementation
        K = pairwise_kernels(X, Y, metric=self.kernel, filter_params=True, n_jobs=self.n_jobs, **params) 
        return K + eps


    def fit(self, X, y, fit_rashomon=False):
        """
        Fit linear model.

        Parameters
        ----------
        X : (n_samples, n_features) `np.array`
            Input values.
        y : (n_samples,) `np.array`
            Target values.
        fit_rashomon : `bool`, default=False
            whether or not to compute the Rashomon Set. This is useful
            ehen fine-tuning the hyperparameters and you do not care
            about the Rashomon Set yet.

        Returns
        -------
        self : object
            Fitted Estimator.
        
        Examples
        --------
        >>> from uxai.kernels import KernelRashomon
        >>> kern_rashomon = KernelRashomon()
        >>> kern_rashomon.fit(X, y, fit_rashomon=True)
        >>> # Get the training objective
        >>> kern_rashomon.train_loss
        0.1346
        >>> # Get the first three coefficients
        >>> kern_rashomon.alpha_s[:3, 0]
        array([16.76765973, -2.9384931 , -3.6456996 ])
        """
        self.N, self.n_features = X.shape
        self.fit_rashomon = False

        # Use training data as the dictionnary
        self.Dict = X
        self.R = self.N
        # The target must be centerted
        self.mu = y.mean()
        y = y - self.mu
        lambd = np.atleast_1d(self.lambd)

        # We need y to be (N, 1)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        # Solve for the alpha_s
        self.K = self.get_kernel(X)
        self.alpha_s, A = solve_cholesky(self.K, y, lambd)
        preds = np.dot(self.K, self.alpha_s)
        
        # Train errors
        self.MSE = mean_squared_error(y, preds)
        self.train_loss = self.MSE + self.lambd * self.h_norm()
        self.RMSE = np.sqrt(self.MSE)
        
        # Add Rashomon-related parameters, this is not necessary when 
        # Fine-tuning the hyperparameters.
        if fit_rashomon:
            self.ellipsoid = Ellipsoid(A, self.alpha_s)

        return self


    def h_norm(self):
        """ RKHS norm ||h||^2_{H_K} """
        return self.alpha_s.T.dot(self.K.dot(self.alpha_s)).item()


    def get_epsilon(self, relative_eps):
        """ Get the epsilon required to reach a given relative loss increase """
        return relative_eps * self.train_loss


    def predict(self, X, epsilon=None):
        """ 
        Return point prediction of alpha_S and [Min, Max] preds of Rashomon Set

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
            Only returned if epsilon is not None.
        """
        K = self.get_kernel(X, self.Dict)
        preds = np.dot(K, self.alpha_s)

        if epsilon is None:
            return preds + self.mu
        
        # Compute min/max preds
        minmax_preds = self.ellipsoid.opt_linear(K.T, epsilon)
        return preds + self.mu, minmax_preds + self.mu


    def gradients(self, X, K=None):
        if self.kernel == "rbf":
            # With RBF we might have already computed the kernel
            if K is None:
                K = self.get_kernel(X, self.Dict)
            return grad_rbf(X, self.Dict, K, self.gamma)
        elif self.kernel == "poly":
            return grad_poly(X, self.Dict, self.gamma, self.degree)


    def IG_per_kernel(self, X, z, n):
        """
        Compute the Integrated Gradient feature attribution

        Parameters
        ----------
        X: (N, d) array
            Points to explain
        z: (1, d) array
            Reference input for the attribution
        n: int
            Number of steps in quadrature

        Returns
        -------
        IG: (N, d, R) array
            IG feature attribution on the mth kernel phi_j(k(., r^(m)), x^(i))
        """
        N, d = X.shape
        delta_t = 1 / (n - 1)
        t = np.linspace(0, 1, n).reshape((1, -1, 1))
        lines = t * X.reshape((N, 1, d)) + (1 - t) * z.reshape((1, 1, d))
        lines = lines.reshape((N*n, d))
        grad = self.gradients(lines).reshape((N, n, d, self.R))
        grad[:, 0, ...] /= 2
        grad[:, -1, ...] /= 2
        IG = (X - z).reshape((N, d, 1)) * delta_t * grad.sum(1)
        return IG



    def feature_attributions(self, X, z, n=100, threshold=0, top_bottom=True):
        """ 
        Compute the Local Feature Attributions (LFA) for a set of samples `x_instances`.
        These LFAs are encoded as a RashomonPartialOrders object.

        Parameters
        ----------
        X: (n_samples, n_features) `np.array`
            Instances on which to compute the LFA
        z: (1, features) `np.array`
            Baseline Instance
        n: `int`
            Number of steps in the quadrature
        
        Returns
        -------
        PO : RashomonPartialOrders
            Object encoding all partial orders for al instances at any tolerance level epsilon

        Examples
        --------
        >>> from uxai.kernels import KernelRashomon
        >>> kern_rashomon = KernelRashomon()
        >>> kern_rashomon.fit(X, y, fit_rashomon=True)
        >>> # Get epsilon for a tolerance 1.05 * train_loss
        >>> epsilon = kern_rashomon.get_epsilon(0.05)
        >>> # Compute the LFA
        >>> rashomon_po = kern_rashomon.feature_attributions(X)
        >>> # Get the partial order on instance i with tolerance epsilon
        >>> PO = rashomon_po.get_poset(i, epsilon, feature_names=["x1", "x2"])
        """

        if X.ndim == 1:
            X = X.reshape((1, -1))
        if z.ndim == 1:
            z = z.reshape((1, -1))
        N, d = X.shape

        ### Range of the Gap across the Rashomon Set ###
        Kz = self.get_kernel(z, self.Dict)
        KX = self.get_kernel(X, self.Dict)
        K_diff = KX - Kz
        gap_lstsq = np.dot(K_diff, self.alpha_s).ravel()
        minmax_gap = self.ellipsoid.opt_linear(K_diff.T)
        gap_eps = (gap_lstsq / (minmax_gap[:, 1] - gap_lstsq)) ** 2

        ### Range of the Attributions across the Rashomon Set ###
        IG = self.IG_per_kernel(X, z, n) # (N, d, R)
        self.IG = IG
        lstsq_attrib = np.zeros((N, d))
        minmax_attribs = np.zeros((N, d, 2))
        for j in range(d):
            # Least-Square
            lstsq_attrib[:, [j]] = np.dot(IG[:, j, :], self.alpha_s)
            # Min-Max
            a = IG[:, j, :].T # (R, N)
            minmax_attribs[:, j, :] = self.ellipsoid.opt_linear(a)
        minmax_attribs_lambda = lambda eps: np.sqrt(eps)*(minmax_attribs - lstsq_attrib[:, :, np.newaxis]) \
                                                + lstsq_attrib[:, :, np.newaxis]


        # Determine critical epsilon values for positive, 
        # negative attribution statements
        pos_eps = np.zeros((N, d))
        neg_eps = np.zeros((N, d))
        for j in range(d):
            step = minmax_attribs[:, j, 1] - lstsq_attrib[:, j]
            critical_eps = ( (np.abs(lstsq_attrib[:, j])-threshold) / step ) ** 2
            pos_idx = lstsq_attrib[:, j] > threshold
            pos_eps[pos_idx, j] = critical_eps[pos_idx]
            neg_idx = lstsq_attrib[:, j] < -threshold
            neg_eps[neg_idx, j] = critical_eps[neg_idx]
        

        # Compare features by features to make partial order
        adjacency_eps = np.zeros((N, d, d))
        for i in range(d):
            for j in range(d):
                if i < j:
                    # Least-Square difference in importance
                    lstsq_ij_diff = np.abs(lstsq_attrib[:, i]) - np.abs(lstsq_attrib[:, j])
                    a = np.sign(lstsq_attrib[:, [i]]) * IG[:, i, :] - \
                        np.sign(lstsq_attrib[:, [j]]) * IG[:, j, :]
                    
                    # Min-max difference in importance
                    min_max_ij_diff = self.ellipsoid.opt_linear(a.T)
                    step = min_max_ij_diff[:, 1] - lstsq_ij_diff
                    # Critical epsilon for comparing relative importance
                    ij_idx = lstsq_ij_diff < 0
                    adjacency_eps[ij_idx, i, j] = (lstsq_ij_diff[ij_idx] / step[ij_idx]) ** 2
                    ji_idx = lstsq_ij_diff > 0
                    adjacency_eps[ji_idx, j, i] = (lstsq_ij_diff[ji_idx] / step[ji_idx]) ** 2
                
        PO = RashomonPartialOrders(lstsq_attrib, minmax_attribs_lambda, gap_eps, 
                                   neg_eps, pos_eps, adjacency_eps, top_bottom)
        return PO



    def get_K_switch(self, X, y, idxs, samples_per_chunk=2):
        """ 
        For general model classes, returns an augmented "Switched" dataset, 
        containing the synthetic instances employed to approximate e_switch.

        Parameters
        ----------
        X: (N, d) array
            The input features.
        y: (N, 1) array
            The target values.
        idxs: List(List(int))
            Partition of features into groups
        samples_per_chunk: int
            Split the data in chunks of `samples_per_chunk` and compute all permutations
            on these chunks. Setting `samples_per_chunk=N` will result in all `N(N-1)`
            permutations.

        Returns
        -------
        X_switch: List((N', d) array)
            List of X_switch for each group of feature.
        y_switch: (N', 1) array
            Vector of switched target values.

        """
        N, d = X.shape

        if samples_per_chunk <= 1:
            raise Exception('If samples_per_chunk=1, then no permuted values are created.')
        if samples_per_chunk % 1 != 0 or samples_per_chunk>N:
            raise Exception('samples_per_chunk must be an integer <= N.')

        # Generate a list of combinations of instances
        combn_inds = [] # Index for permuting data.
        n_chunks = int(N / samples_per_chunk)
        starts = samples_per_chunk * np.arange(n_chunks+1) # Beginning of each chunk
        for j in range(n_chunks):
            # Store all combinations within the chunk
            chunk_inds = np.arange(starts[j], starts[j+1])
            p1_idx, p2_idx = np.meshgrid(chunk_inds, chunk_inds)
            combn_inds += [np.column_stack((p1_idx.ravel(), p2_idx.ravel()))]
        combn_inds = np.vstack(combn_inds).astype(int)
        is_perm = combn_inds[:, 0] != combn_inds[:, 1]
        combn_inds = combn_inds[is_perm]
        full_n = combn_inds.shape[0]

        # Evaluate the Kernel on the switched dataset
        K_switch = []
        for grouped_idx in idxs:
            not_grouped_idx = [i for i in range(d) if not i in grouped_idx]
            X_switch = np.zeros( (full_n, d) )
            X_switch[:, grouped_idx] = X[combn_inds[:, [0]], grouped_idx]
            X_switch[:, not_grouped_idx] = X[combn_inds[:, [1]], not_grouped_idx]
            K_switch.append(self.get_kernel(X_switch, self.Dict))

        y_switch = y[combn_inds[:, 1]]

        return K_switch, y_switch



    def feature_importance(self, X, y, epsilon, feature_names, idxs=None, 
                                threshold=0, top_bottom=True, samples_per_chunk=10):
        
        N = X.shape[0]
        if idxs is None:
            idxs = [[i] for i in range(self.n_features)]
        
        # Shift the features
        K = self.get_kernel(X, self.Dict)
        K_switch, y_switch = self.get_K_switch(X, y, idxs, samples_per_chunk=samples_per_chunk)
        n_perms = len(y_switch)

        # Compute Feature Importance
        alpha_s_importance = np.zeros(len(idxs))
        min_max_importance = np.zeros((len(idxs), 2))
        ytK = y.T.dot(K)
        KtK = K.T.dot(K)
        C = self.A / epsilon

        # Reduce the dimensionality of the Rashomon Set since it can be flat
        # in some directions
        W, V = eigh(C)
        keep_dims = np.where(np.abs(W) > 1e-7)[0].reshape((1, -1))
        print("Rashomon Set of dim :", keep_dims.size)
        # Express C in its truncated eigen space
        C_half_inv = np.diag(np.sqrt(1/W[keep_dims.ravel()]))

        for i in range(len(idxs)):
            # Compute quadratic form
            A = K_switch[i].T.dot(K_switch[i]) / n_perms - KtK / N
            b = -2 * (y_switch.T.dot(K_switch[i]) / n_perms - ytK / N).T

            # Reduce dimensionality of Objective function
            A, b, alpha_s = reduce_dim(A, b, self.alpha_s, keep_dims, V)

            # Model Reliance of the regularized least square
            alpha_s_importance[i] = alpha_s.T.dot(A.dot(alpha_s)) + b.T.dot(alpha_s)

            # Transform ellipsoid into unit circle
            A_prime = C_half_inv.dot(A.dot(C_half_inv))
            b_prime = C_half_inv.dot(b + 2 * A.dot(alpha_s))

            # Solve QPQC to get MCR- and MCR+
            try:
                min_val, _, max_val, _ = opt_qpqc_standard_exact(A_prime, b_prime)
                # print("Solved exactly")
            except:
                # If we have issues, fall back to an approximate method
                min_val, max_val= opt_qpqc_standard_approx(A_prime, b_prime)
                # print("Solved approximatelly")
            min_val += alpha_s_importance[i]
            max_val += alpha_s_importance[i]
            min_max_importance[i] = [min_val, max_val]
        
        
        # If there exists model that dont rely on this feature, we do not
        # plot it in the PO. We only plot features that are "necessary"
        # for good performance
        ambiguous = set()
        for i in range(min_max_importance.shape[0]):
            if min_max_importance[i, 0] < threshold:
                ambiguous.add(i)
        # print(ambiguous)
        
        # Compare features by features to make partial order
        select_features_idx = [i for i in range(len(idxs)) if i not in ambiguous]
        adjacency = np.zeros((len(idxs), len(idxs)))
        for i in select_features_idx:
            for j in select_features_idx:
                if i < j:
                    # Sometimes the adjacency only requires compare the min/max
                    # importance of each individual feature
                    if min_max_importance[i, 1] < min_max_importance[j, 0]:
                        adjacency[i, j] = 1
                    elif min_max_importance[j, 1] < min_max_importance[i, 0]:
                        adjacency[j, i] = 1
                    # In the more general case, we must solve a QPQC  max I_i - I_j
                    elif np.sign(min_max_importance[i, 0] - min_max_importance[j, 0]) ==\
                         np.sign(min_max_importance[i, 1] - min_max_importance[j, 1]):
                        
                        # Compute quadratic form
                        A = K_switch[i].T.dot(K_switch[i]) / n_perms
                        A -= K_switch[j].T.dot(K_switch[j]) / n_perms
                        b = -2 * y_switch.T.dot(K_switch[i]) / n_perms
                        b += 2 * y_switch.T.dot(K_switch[j]) / n_perms
                        b = b.T

                        # Solve linear objective instead
                        if np.isclose(0, A).all():
                            min_max_val = self.ellipsoid.opt_linear(b, return_input_sol=False)
                            min_val = min_max_val[0, 0]
                            max_val = min_max_val[0, 1]
                        # Solve QPQC
                        else:
                            # Reduce dimensionality of Objective function
                            A, b, alpha_s = reduce_dim(A, b, self.alpha_s, keep_dims, V)

                            # Transform ellipsoid into unit circle
                            gap = float(alpha_s.T.dot(A.dot(alpha_s)) + b.T.dot(alpha_s))
                            A_prime = C_half_inv.dot(A.dot(C_half_inv.T))
                            b_prime = C_half_inv.dot(b + 2 * A.dot(alpha_s))

                            # Solve QPQC to get MCR- and MCR+
                            try:
                                min_val, _, max_val, _ = opt_qpqc_standard_exact(A_prime, b_prime)
                                # print("Solved exactly")
                            except:
                                # If A is singular, we must resort to using an approximate method
                                min_val, max_val = opt_qpqc_standard_approx(A_prime, b_prime)
                                # print("Solved approximatelly")
                            min_val += gap
                            max_val += gap
                        
                        if np.sign(min_val) == np.sign(max_val):
                            # Features are comparable
                            if max_val < 0:
                                adjacency[i, j] = 1
                            else:
                                adjacency[j, i] = 1
                    
        po = PartialOrder(alpha_s_importance, adjacency, ambiguous=ambiguous, 
                            features_names=feature_names, top_bottom=top_bottom)
        return min_max_importance, po



def reduce_dim(A, b, alpha_s, keep_dims, V):
    # Reduce dimensionality
    A = V.T.dot(A.dot(V))[keep_dims.T, keep_dims]
    b = V.T.dot(b)[keep_dims.ravel()]
    alpha_s = V.T.dot(alpha_s)[keep_dims.ravel()]
    return A, b, alpha_s