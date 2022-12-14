""" Rashomon Sets of Kernel Ridge models """

import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics import mean_squared_error
from .utils_optim import opt_lin_ellipsoid
from .partial_orders import RashomonPartialOrders


def solve_cholesky(K, y, lambd):
    # w = inv(K + lambd*R*I) * y
    R = K.shape[0]
    A = K + lambd * R * np.eye(R)
    return linalg.solve(A, y, sym_pos=True, overwrite_a=False), K.dot(A/R)+1e-4*np.eye(R)


def grad_rbf(X, Dict, K, gamma):
    """ 
    Compute the gradient of the RBF kernels

    Parameters
    ----------
    X: (N, d) array
        Where to compute the gradients
    Dict: (R, d) array
        Reference points
    K: (N, R) array
        Matrix whose ith row and ith column if k(x^(i), r^(j))
    gamma: float
        Scale parameter of the RBF kernel

    Returns
    -------
    sol_values: (N, d, R) array
        3D tensor whose ith row jth column is the gradient of 
        k(., r^(j)) evaluated at x^(i)
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
    X: (N, d) array
        Where to compute the gradients
    Dict: (R, d) array
        Reference points
    gamma: float
        Scale parameter of the Poly kernel
    degree: int
        Polynomial degree

    Returns
    -------
    sol_values: (N, d, R) array
        3D tensor whose ith row jth column is the gradient of 
        k(., r^(j)) evaluated at x^(i)
    """
    N, d = X.shape
    R = Dict.shape[0]
    params = {"gamma" : gamma, "degree" : degree-1}
    K = pairwise_kernels(X, Dict, metric="poly", filter_params=True, **params)
    ref = Dict.T.reshape((1, d, R))
    C = degree * gamma
    return C * ref * K.reshape((N, 1, R))



class KernelRashomon(RegressorMixin, BaseEstimator):

    def __init__(
        self,
        lambd=1,
        *,
        kernel="rbf",
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
        n_jobs=None
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
            eps = 1e-8 * np.eye(self.R)
        else:
            eps = 0
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma, "degree": self.degree, "coef0": self.coef0}
        K = pairwise_kernels(X, Y, metric=self.kernel, filter_params=True, n_jobs=self.n_jobs, **params) 
        return K + eps


    def fit(self, X, y):
        self.N, self.n_features = X.shape
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
        
        # Define Rashomon Set
        self.A = A
        self.A_half = np.linalg.cholesky(self.A)
        self.A_half_inv = np.linalg.inv(self.A_half)
        self.A_inv = self.A_half_inv.T.dot(self.A_half_inv)

        return self


    def h_norm(self):
        """ RKHS norm ||h||^2_{H_K} """
        return self.alpha_s.T.dot(self.K.dot(self.alpha_s)).item()


    # def get_RMSE(self, X, y):
    #     X_ = (X - self.X_mean) / self.X_std
    #     y_ = (y - self.y_mean) / self.y_std

    #     # Fit the optimal model on training data
    #     preds = self.regr.predict(X_)

    #     # Unscaled MSE
    #     uMSE = mean_squared_error(y_, preds)
    #     RMSE = self.y_std * np.sqrt(uMSE)
    #     return RMSE


    def get_epsilon(self, relative_eps):
        """ Get the epsilon required to reach a given relative loss increase """
        return relative_eps * self.train_loss


    def predict(self, X, epsilon=None):
        """ Return point prediction of alpha_s and [Min, Max] preds of Rashomon Set"""
        K = self.get_kernel(X, self.Dict)
        preds = np.dot(K, self.alpha_s)

        if epsilon is None:
            return preds + self.mu
        
        # Compute min/max preds
        A_half_inv = self.A_half_inv * np.sqrt(epsilon)
        minmax_preds = opt_lin_ellipsoid(K.T, A_half_inv, self.alpha_s, return_input_sol=False)
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



    def attributions(self, X, z, n=100, top_bottom=True):
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
        minmax_gap = opt_lin_ellipsoid(K_diff.T, self.A_half_inv, self.alpha_s)
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
            minmax_attribs[:, j, :] = opt_lin_ellipsoid(a, self.A_half_inv, self.alpha_s)
        minmax_attribs_lambda = lambda eps: np.sqrt(eps)*(minmax_attribs - lstsq_attrib[:, :, np.newaxis]) \
                                                + lstsq_attrib[:, :, np.newaxis]


        # Determine critical epsilon values for positive, 
        # negative attribution statements
        pos_eps = np.zeros((N, d))
        neg_eps = np.zeros((N, d))
        for j in range(d):
            step = minmax_attribs[:, j, 1] - lstsq_attrib[:, j]
            critical_eps = (lstsq_attrib[:, j] / step) ** 2
            pos_idx = lstsq_attrib[:, j] > 0
            pos_eps[pos_idx, j] = critical_eps[pos_idx]
            neg_idx = lstsq_attrib[:, j] < 0
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
                    min_max_ij_diff = opt_lin_ellipsoid(a.T, self.A_half_inv, self.alpha_s)
                    step = min_max_ij_diff[:, 1] - lstsq_ij_diff
                    # Critical epsilon for comparing relative importance
                    ij_idx = lstsq_ij_diff < 0
                    adjacency_eps[ij_idx, i, j] = (lstsq_ij_diff[ij_idx] / step[ij_idx]) ** 2
                    ji_idx = lstsq_ij_diff > 0
                    adjacency_eps[ji_idx, j, i] = (lstsq_ij_diff[ji_idx] / step[ji_idx]) ** 2
                
        PO = RashomonPartialOrders(lstsq_attrib, minmax_attribs_lambda, gap_eps, 
                                   neg_eps, pos_eps, adjacency_eps, top_bottom)
        return PO
