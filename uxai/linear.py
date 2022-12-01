""" Rashomon Sets of Linear/Additive models """

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from .partial_orders import PartialOrder
from .utils_optim import opt_lin_ellipsoid, opt_qpqc
from .partial_orders import RashomonPartialOrders


def get_ellipse_border(A_half_inv, x_hat):
    """ Plot the border of ellipse (x - x_hat)^T A (x - x_hat) <= 1 ? """
    theta = np.linspace(0, 2 * np.pi, 100)
    zz = np.vstack([np.cos(theta), np.sin(theta)])
    zz = A_half_inv.T.dot(zz) + x_hat
    return zz


def verif_cross(a, b, A_half_inv, x_hat):
    """ Does a^t x = b  cross (x - x_hat)^T A (x - x_hat) <= 1 ? """
    a_prime = A_half_inv.dot(a)
    b_prime = b - a.T.dot(x_hat)
    return bool(np.abs(b_prime) < np.linalg.norm(a_prime))


def shur_complement(A, idx):
    # Project the ellipdoid
    N = A.shape[0]
    select = np.array([idx])
    non_select = np.array([[f for f in range(N) if f not in select]])
    J = A[select.T, select]
    L = A[select.T, non_select]
    K = A[non_select.T, non_select]
    # Take the schur complement
    A_shur = J - L.dot(np.linalg.inv(K).dot(L.T))
    return A_shur



class LinearRashomon(object):

    def __init__(self, **kwargs):
        
        self.regr = LinearRegression(**kwargs)


    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.N = X.shape[0]

        # Get Scaling statistics
        self.X_mean = X.mean(0)
        self.X_std = X.std(0)
        self.y_mean = y.mean(0)
        self.y_std = y.std(0)

        # Rescale Input and Target
        X_ = (X - self.X_mean) / self.X_std
        y_ = (y - self.y_mean) / self.y_std

        # Fit the optimal model on training data
        self.regr.fit(X_, y_)
        preds = self.regr.predict(X_)
        
        # Unscaled MSE
        self.uMSE = mean_squared_error(y_, preds)
        self.RMSE = self.y_std * np.sqrt(self.uMSE)
        self.MSE = self.RMSE ** 2
        
        # Define Rashomon Set
        self.w_hat = np.concatenate((self.regr.intercept_, 
                                     self.regr.coef_.ravel())).reshape((-1, 1))
        X_tilde = np.column_stack( (np.ones(len(X_)), X_) )
        self.X_tilde = X_tilde
        self.A = X_tilde.T.dot(X_tilde) / self.N
        self.A_half = np.linalg.cholesky(self.A)
        self.A_half_inv = np.linalg.inv(self.A_half)
        self.A_inv = self.A_half_inv.T.dot(self.A_half_inv)

        return self


    def get_RMSE(self, X, y):
        X_ = (X - self.X_mean) / self.X_std
        y_ = (y - self.y_mean) / self.y_std

        # Fit the optimal model on training data
        preds = self.regr.predict(X_)

        # Unscaled MSE
        uMSE = mean_squared_error(y_, preds)
        RMSE = self.y_std * np.sqrt(uMSE)
        return RMSE


    def get_epsilon(self, target_RSME):
        """ Get the epsilon required to reach a given RSME """
        return (target_RSME / self.y_std)**2 - self.uMSE


    def predict(self, X, epsilon=None):
        """ Return point prediction of Least Square and [Min, Max] preds of Rashomon Set"""
        X_ = (X - self.X_mean) / self.X_std
        y_ = self.regr.predict( X_ )

        if epsilon is None:
            return self.y_std * y_ + self.y_mean
        
        # Compute min/max preds
        X_tilde = np.column_stack( (np.ones(len(X_)), X_) )
        A_half_inv = self.A_half_inv * np.sqrt(epsilon)
        minmax_preds = opt_lin_ellipsoid(X_tilde.T, A_half_inv, self.w_hat, return_input_sol=False)
        return self.y_std * y_ + self.y_mean, self.y_std * minmax_preds + self.y_mean


    def min_max_coeffs(self, epsilon):
        """ Compute the minimal value of a specific coefficient in the linear model """
        # Highest/Smallest slopes
        extreme_slopes = np.array([-1, 1]) * np.sqrt(epsilon * self.A_inv.diagonal()).reshape((-1, 1))
        extreme_slopes += self.w_hat
        return extreme_slopes[1:] * self.y_std / self.X_std.reshape((-1, 1))


    def partial_dependence(self, x, idx, epsilon):
        x_ = (x - self.X_mean[idx]) / self.X_std[idx]

        # Compute min/max preds
        a = np.zeros((self.n_features + 1, len(x)))
        a[np.array(idx)+1] = x_.T
        A_half_inv = self.A_half_inv * np.sqrt(epsilon)
        minmax_preds = opt_lin_ellipsoid(a, A_half_inv, self.w_hat, return_input_sol=False)
        return self.y_std * minmax_preds + self.y_mean


    def feature_importance(self, epsilon, feature_names, idxs=None, threshold=0, top_bottom=True):
        
        # Linear model : the min/max importances are the min/max weight magnitude
        if idxs is None:
            # Importance of the least square
            w_hat_importance = np.abs(self.y_std*self.w_hat[1:].ravel())

            # Range of the FI across the Rashomon Set
            min_max_importance = self.min_max_coeffs(epsilon) * self.X_std.reshape((-1, 1))
        
        # General additive models : the min/max component variance requires solving a QPQC
        else:
            # Importance of the least square
            w_hat_importance = np.zeros(len(idxs))
            for i, grouped_idx in enumerate(idxs):
                grouped_idx = np.array(grouped_idx)+1
                if len(grouped_idx) == 1:
                    w_hat_importance[i] = np.abs(self.y_std*self.w_hat[grouped_idx])
                else:
                    w_hat_importance[i] = self.y_std * np.std(self.X_tilde[:, grouped_idx].dot(self.w_hat[grouped_idx]))
            
            # Range of the FI across the Rashomon Set
            uni_min_max_importance = self.min_max_coeffs(epsilon) * self.X_std.reshape((-1, 1))
            N = self.X_tilde.shape[0]
            min_max_importance = np.zeros((len(idxs), 2))
            for i, grouped_idx in enumerate(idxs):
                # For features with one component
                if len(grouped_idx) == 1:
                    min_max_importance[i] = uni_min_max_importance[i]
                # For features with two or more component
                else:
                    grouped_idx = np.array(grouped_idx)+1
                    # Project the ellipdoid
                    A = shur_complement(self.A, grouped_idx)
                    A_half = np.linalg.cholesky(A) / np.sqrt(epsilon)
                    A_half_inv = np.linalg.inv(A_half)
                    # Range of the FI across the Rashomon Set
                    B = 1 / N * self.X_tilde[:, grouped_idx].T.dot(self.X_tilde[:, grouped_idx])
                    B_prime = A_half_inv.dot(B.dot(A_half_inv.T))
                    z_s = -1 * A_half.T.dot(self.w_hat[grouped_idx])
                    min_val, _, max_val, _ = opt_qpqc(B_prime, z_s)
                    min_max_importance[i, 0] = self.y_std * np.sqrt(min_val)
                    min_max_importance[i, 1] = self.y_std * np.sqrt(max_val)
            

        # Coeffs with small importance for all models are negligible
        negligible = set()
        for i in range(min_max_importance.shape[0]):
            if min_max_importance[i, 0] > -threshold and min_max_importance[i, 1] <  threshold:
                negligible.add(i)
        print(negligible)
        
        # For linear models : incomparable features occur when a specific plane crosse the ellipsoid
        if idxs is None:
            A_half_inv = self.A_half_inv * np.sqrt(epsilon)

            # Compare features by features to make partial order
            select_features_idx = [i for i in range(self.n_features) if i not in negligible]
            adjacency = np.zeros((self.n_features, self.n_features))
            for i in select_features_idx:
                for j in select_features_idx:
                    if i < j:
                        a = np.zeros((self.n_features+1, 1))
                        # Cross line 1?
                        a[i+1, 0] = 1
                        a[j+1, 0] = 1
                        cross_line_one = verif_cross(a, 0, A_half_inv, self.w_hat)
                        # Cross line 2?
                        a[j+1, 0] *= -1
                        cross_line_two = verif_cross(a, 0, A_half_inv, self.w_hat)
                        
                        if not cross_line_one and not cross_line_two:
                            # Features are comparable
                            if np.abs(self.w_hat[i+1]) < np.abs(self.w_hat[j+1]):
                                adjacency[i, j] = 1
                            else:
                                adjacency[j, i] = 1

        # General additive models : relative feature importance requires solving a QPQC
        else:
            # Compare features by features to make partial order
            select_features_idx = [i for i in range(len(idxs)) if i not in negligible]
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
                            # Identify the basis functions for i and j
                            grouped_idx_1 = np.array(idxs[i])+1
                            grouped_idx_2 = np.array(idxs[j])+1
                            grouped_idx = np.concatenate((grouped_idx_1, grouped_idx_2))
                            
                            # Project the ellipsoid on the two feature basis functions
                            A = shur_complement(self.A, grouped_idx)
                            A_half = np.linalg.cholesky(A) / np.sqrt(epsilon)
                            A_half_inv = np.linalg.inv(A_half)
                            
                            # Generate the quadratic form
                            B = np.zeros((len(grouped_idx), len(grouped_idx)))
                            B[:len(grouped_idx_1), :len(grouped_idx_1)] = self.X_tilde[:, grouped_idx_1].T.dot(self.X_tilde[:, grouped_idx_1])
                            B[-len(grouped_idx_2):, -len(grouped_idx_2):] = -self.X_tilde[:, grouped_idx_2].T.dot(self.X_tilde[:, grouped_idx_2])
                            B /= N
                            B_prime = A_half_inv.dot(B.dot(A_half_inv.T))
                            z_s = -1 * A_half.T.dot(self.w_hat[grouped_idx])

                            # Solve the non-convex QPQC
                            min_val, _, max_val, _ = opt_qpqc(B_prime, z_s)
                            
                            if np.sign(min_val) == np.sign(max_val):
                                # Features are comparable
                                if max_val < 0:
                                    adjacency[i, j] = 1
                                else:
                                    adjacency[j, i] = 1
                    
        po = PartialOrder(w_hat_importance, adjacency, negligible=negligible,
                        ambiguous=set(), features_names=feature_names, top_bottom=top_bottom)
        return min_max_importance, po


    def attributions(self, x_instances, idxs=None, threshold=0, top_bottom=True):
        if x_instances.ndim == 1:
            x_instances = x_instances.reshape((1, -1))
        assert x_instances.shape[1] == self.n_features, "Number of features is not the same as in training"
        N = x_instances.shape[0]

        # Rescale the instance
        x_instances = (x_instances - self.X_mean) / self.X_std
        

        ### Range of the Gap across the Rashomon Set ###
        x_instance_tilde = np.vstack( (np.zeros((1, N)), x_instances.T) ) * self.y_std
        gap_lstsq = x_instance_tilde.T.dot(self.w_hat)[:, 0]
        minmax_gap = opt_lin_ellipsoid(x_instance_tilde, self.A_half_inv, self.w_hat)
        gap_eps = (gap_lstsq / (minmax_gap[:, 1] - gap_lstsq)) ** 2

        # Features to which to attribute a change in model output
        if idxs is None:
            idxs = [[i] for i in range(self.n_features)]
        n_attribs = len(idxs)


        ### Range of the Attributions across the Rashomon Set ###
        lstsq_attrib = np.zeros((N, n_attribs))
        minmax_attribs = np.zeros((N, n_attribs, 2))
        for j in range(n_attribs):
            idxs_j = np.array(idxs[j])
            # Least-Square
            lstsq_attrib[:, j] = np.sum(x_instances[:, idxs_j] * self.w_hat[idxs_j+1].T, axis=1)
            # Min-Max
            a = np.zeros((self.n_features+1, N))
            a[np.array(idxs[j]) + 1, :] = self.y_std * x_instances[:, idxs_j].T
            minmax_attribs[:, j, :] = opt_lin_ellipsoid(a, self.A_half_inv, self.w_hat)
        lstsq_attrib = self.y_std * lstsq_attrib
        minmax_attribs_lambda = lambda eps: np.sqrt(eps)*(minmax_attribs - lstsq_attrib[:, :, np.newaxis]) \
                                                + lstsq_attrib[:, :, np.newaxis]


        # Determine critical epsilon values for positive, 
        # negative attribution statements
        pos_eps = np.zeros((N, n_attribs))
        neg_eps = np.zeros((N, n_attribs))
        for j in range(n_attribs):
            step = minmax_attribs[:, j, 1] - lstsq_attrib[:, j]
            critical_eps = (lstsq_attrib[:, j] / step) ** 2
            pos_idx = lstsq_attrib[:, j] > 0
            pos_eps[pos_idx, j] = critical_eps[pos_idx]
            neg_idx = lstsq_attrib[:, j] < 0
            neg_eps[neg_idx, j] = critical_eps[neg_idx]
        

        # Compare features by features to make partial order
        adjacency_eps = np.zeros((N, n_attribs, n_attribs))
        for i in range(n_attribs):
            for j in range(n_attribs):
                if i < j:
                    idxs_i = np.array(idxs[i])
                    idxs_j = np.array(idxs[j])
                    # Least-Square difference in importance
                    lstsq_ij_diff = np.abs(lstsq_attrib[:, i]) - np.abs(lstsq_attrib[:, j])
                    a = np.zeros((N, self.n_features+1))
                    a[:, idxs_i+1] =  np.sign(lstsq_attrib[:, [i]]) * x_instances[:, idxs_i]
                    a[:, idxs_j+1] = -np.sign(lstsq_attrib[:, [j]]) * x_instances[:, idxs_j]
                    
                    # Min-max difference in importance
                    min_max_ij_diff = opt_lin_ellipsoid(a.T, self.A_half_inv, self.w_hat)
                    min_max_ij_diff *= self.y_std
                    step = min_max_ij_diff[:, 1] - lstsq_ij_diff
                    # Critical epsilon for comparing relative importance
                    ij_idx = lstsq_ij_diff < 0
                    adjacency_eps[ij_idx, i, j] = (lstsq_ij_diff[ij_idx] / step[ij_idx]) ** 2
                    ji_idx = lstsq_ij_diff > 0
                    adjacency_eps[ji_idx, j, i] = (lstsq_ij_diff[ji_idx] / step[ji_idx]) ** 2
                
        PO = RashomonPartialOrders(lstsq_attrib, minmax_attribs_lambda, gap_eps, 
                                   neg_eps, pos_eps, adjacency_eps, top_bottom)
        return PO



    def plot_rashomon_set(self, feature1, feature2, epsilon, x_instance=None, importance=False):
        # Project the ellipsoid on two components
        A = shur_complement(self.A, [feature1+1, feature2+1])
        A_half_inv = np.linalg.inv(np.linalg.cholesky(A)) * np.sqrt(epsilon)
        zz = get_ellipse_border(A_half_inv, self.w_hat[[feature1+1, feature2+1]])
        # Plot feature attribution
        if x_instance is not None:
            scale = (x_instance - self.X_mean) / self.X_std * self.y_std
            plt.plot(scale[feature1]*zz[0, :], scale[feature2]*zz[1, :], 'k-', alpha=0.5)
            plt.plot(scale[feature1]*self.w_hat[feature1+1], scale[feature2]*self.w_hat[feature2+1], 'kx')
        # Plot feature importance
        elif importance:
            scale = self.y_std
            plt.plot(scale**zz[0, :], scale*zz[1, :], 'k-', alpha=0.5)
            plt.plot(scale*self.w_hat[feature1+1], scale*self.w_hat[feature2+1], 'kx')
        # Plot the coefficients
        else:
            scale = self.y_std / self.X_std[[feature1, feature2]]
            plt.plot(scale[0] * zz[0, :], scale[1] * zz[1, :], 'k-', alpha=0.5)
            plt.plot(scale[0] * self.w_hat[feature1+1], scale[1] * self.w_hat[feature2+1], 'kx')