""" 
Rashomon Sets of Linear/Additive models. Piece-Wise Linear Monotic Additive models are also supported
"""

import numpy as np
from scipy.linalg import lstsq
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import SplineTransformer
from .utils_optim import opt_qpqc_centered_exact, opt_lin_ellipsoid_quadrant
from .utils import Ellipsoid, abs_interval, intervaled_cumsum
from .partial_orders import PartialOrder, RashomonPartialOrders


class LinearRashomon(BaseEstimator, RegressorMixin):
    """
    Rashomon Set for ordinary least squares Linear Regression.

    LinearRashomon fits a linear model with coefficients `w_S = (w1, ..., wd)`
    to minimize the mean squared error on the training data S.
    The resulting Rashomon Set is an ellipsoid of the form

    `(w - w_S)^T A (w - w_S) <= epsilon`

    over which we can compute a consensus on local/global 
    feature importance statements.


    Attributes
    ----------
    w_hat : (n_features+1, 1) `np.array`
        Estimated bias and coefficients for the linear regression problem.

    MSE : `float`
        Mean-Squared-Error of `w_hat`

    RMSE : `float`
        Root-Mean-Squared-Error of `w_hat`

    """
    def __init__(self):
        # We wrap the class around the sklearn one
        self.regr = LinearRegression()


    def fit(self, X, y):
        """
        Fit linear model.

        Parameters
        ----------
        X : (n_samples, n_features) `np.array`
            Input values.
        y : (n_samples,) `np.array`
            Target values.

        Returns
        -------
        self : object
            Fitted Estimator.
        
        Examples
        --------
        >>> from uxai.linear import LinearRashomon
        >>> linear_rashomon = LinearRashomon()
        >>> linear_rashomon.fit(X, y)
        >>> # Get the train performance
        >>> RMSE = linear_rashomon.RMSE
        >>> # Get the model coefficients
        >>> coefs = linear_rashomon.w_hat[1:, 0]
        array([0.5517, 0.4964])
        """
        self.N, self.n_features = X.shape

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
        self.RMSE = float(self.y_std * np.sqrt(self.uMSE))
        self.MSE = self.RMSE ** 2
        
        # Define Rashomon Set
        self.w_hat = np.concatenate((self.regr.intercept_, 
                                     self.regr.coef_.ravel())).reshape((-1, 1))
        X_tilde = np.column_stack( (np.ones(len(X_)), X_) )
        A = X_tilde.T.dot(X_tilde) / self.N
        self.X_tilde = X_tilde
        self.ellipsoid = Ellipsoid(A, self.w_hat)
        self.A_inv = np.linalg.inv(A)

        return self


    def get_RMSE(self, X, y):
        """
        Get the RMSE on new data

        Parameters
        ----------
        X : (n_samples, n_features) `np.array`
            Test data.
        y : (n_samples,) `np.array`
            Target values.

        Returns
        -------
        RMSE : `float`
            Performance of w_hat on the test dataset
        
        Examples
        --------
        >>> from uxai.linear import LinearRashomon
        >>> linear_rashomon = LinearRashomon()
        >>> linear_rashomon.fit(X_train, y_train)
        >>> # Get the train performance
        >>> linear_rashomon.RMSE
        0.1253
        >>> # Get the test performance
        >>> linear_rashomon.get_RMSE(X_test, y_test)
        0.1488
        """
        # Normalize the input and target
        X_ = (X - self.X_mean) / self.X_std
        y_ = (y - self.y_mean) / self.y_std

        # Fit the optimal model on training data
        preds = self.regr.predict(X_)

        # Unscaled MSE
        uMSE = mean_squared_error(y_, preds)
        RMSE = float(self.y_std * np.sqrt(uMSE))
        return RMSE


    def get_epsilon(self, target_RSME):
        """ Get the epsilon required to reach a given RSME """
        return (target_RSME / self.y_std)**2 - self.uMSE


    def predict(self, X, epsilon=None):
        """ 
        Return point prediction of Least Square and [Min, Max] preds of Rashomon Set

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
        
        X_ = (X - self.X_mean) / self.X_std
        y_ = self.regr.predict( X_ )

        if epsilon is None:
            return self.y_std * y_ + self.y_mean
        
        # Compute min/max preds
        X_tilde = np.column_stack( (np.ones(len(X_)), X_) )
        minmax_preds = self.ellipsoid.opt_linear(X_tilde.T, epsilon)
        return self.y_std * y_ + self.y_mean, self.y_std * minmax_preds + self.y_mean


    def min_max_coeffs(self, epsilon):
        """ Compute the minimal value of the coefficient in the linear model """
        extreme_slopes = np.array([-1, 1]) * np.sqrt(epsilon * self.A_inv.diagonal()).reshape((-1, 1))
        extreme_slopes += self.w_hat
        return extreme_slopes[1:] * self.y_std / self.X_std.reshape((-1, 1))


    def min_max_importance(self, idxs, epsilon):
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
                    # The importance is the norm of the weights
                    min_max_importance[i] = abs_interval(uni_min_max_importance[grouped_idx[0]])
                # For features with two or more component
                else:
                    grouped_idx = np.array(grouped_idx)+1
                    # Project the ellipdoid
                    proj_ellipsoid = self.ellipsoid.projection(grouped_idx)
                    # Range of the FI across the Rashomon Set
                    Q = 1 / N * self.X_tilde[:, grouped_idx].T.dot(self.X_tilde[:, grouped_idx])
                    Q_prime = epsilon * proj_ellipsoid.C_inv.dot(Q.dot(proj_ellipsoid.C_inv.T))
                    z_s = -1 * proj_ellipsoid.C.T.dot(self.w_hat[grouped_idx]) / np.sqrt(epsilon)
                    # Solve the non-convex QPQC in centered form
                    min_val, _, max_val, _ = opt_qpqc_centered_exact(Q_prime, z_s)
                    min_max_importance[i, 0] = self.y_std * np.sqrt(min_val)
                    min_max_importance[i, 1] = self.y_std * np.sqrt(max_val)
        
        return w_hat_importance, min_max_importance


    def partial_dependence(self, x, idx, epsilon):
        x_ = (x - self.X_mean[idx]) / self.X_std[idx]

        # Compute min/max preds
        a = np.zeros((self.n_features + 1, len(x)))
        a[np.array(idx)+1] = x_.T
        minmax_preds = self.ellipsoid.opt_linear(a, epsilon)
        return self.y_std * minmax_preds + self.y_mean


    def feature_importance(self, epsilon, feature_names, idxs=None, threshold=0, top_bottom=True):
        
        # Compute min max of feature importance
        w_hat_importance, min_max_importance = self.min_max_importance(idxs, epsilon)        

        # If there exists model that dont rely on this feature, we do not
        # plot it in the PO. We only plot features that are "necessary"
        # for good performance
        ambiguous = set()
        for i in range(min_max_importance.shape[0]):
            if min_max_importance[i, 0] < threshold:
                ambiguous.add(i)
        print(ambiguous)
        
        # For linear models : incomparable features occur when a specific plane crosse the ellipsoid
        if idxs is None:

            # Compare features by features to make partial order
            select_features_idx = [i for i in range(self.n_features) if i not in ambiguous]
            adjacency = np.zeros((self.n_features, self.n_features))
            for i in select_features_idx:
                for j in select_features_idx:
                    if i < j:
                        a = np.zeros((self.n_features+1, 1))
                        # Cross line 1?
                        a[i+1, 0] = 1
                        a[j+1, 0] = 1
                        cross_line_one = self.ellipsoid.verif_cross(a, 0, epsilon)
                        # Cross line 2?
                        a[j+1, 0] *= -1
                        cross_line_two = self.ellipsoid.verif_cross(a, 0, epsilon)
                        
                        if not cross_line_one and not cross_line_two:
                            # Features are comparable
                            if np.abs(self.w_hat[i+1]) < np.abs(self.w_hat[j+1]):
                                adjacency[i, j] = 1
                            else:
                                adjacency[j, i] = 1

        # General additive models : relative feature importance requires solving a QPQC
        else:
            N = self.X_tilde.shape[0]

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
                            # Identify the basis functions for i and j
                            grouped_idx_1 = np.array(idxs[i])+1
                            grouped_idx_2 = np.array(idxs[j])+1
                            grouped_idx = np.concatenate((grouped_idx_1, grouped_idx_2))
                            
                            # Project the ellipsoid on the two feature basis functions
                            proj_ellipsoid = self.ellipsoid.projection(grouped_idx)
                            
                            # Generate the quadratic form
                            Q = np.zeros((len(grouped_idx), len(grouped_idx)))
                            Q[:len(grouped_idx_1), :len(grouped_idx_1)] = self.X_tilde[:, grouped_idx_1].T.dot(self.X_tilde[:, grouped_idx_1])
                            Q[-len(grouped_idx_2):, -len(grouped_idx_2):] = -self.X_tilde[:, grouped_idx_2].T.dot(self.X_tilde[:, grouped_idx_2])
                            Q /= N
                            Q_prime = epsilon * proj_ellipsoid.C_inv.dot(Q.dot(proj_ellipsoid.C_inv.T))
                            z_s = -1 * proj_ellipsoid.C.T.dot(self.w_hat[grouped_idx]) / np.sqrt(epsilon)

                            # Solve the non-convex QPQC in centered form
                            min_val, _, max_val, _ = opt_qpqc_centered_exact(Q_prime, z_s)
                            
                            if np.sign(min_val) == np.sign(max_val):
                                # Features are comparable
                                if max_val < 0:
                                    adjacency[i, j] = 1
                                else:
                                    adjacency[j, i] = 1
                    
        po = PartialOrder(w_hat_importance, adjacency, ambiguous=ambiguous, 
                            features_names=feature_names, top_bottom=top_bottom)
        return min_max_importance, po



    def sample_feature_attributions(self, x_instances, epsilon, background=None, 
                                                        idxs=None, n_samples=1000):
        N = x_instances.shape[0]

        # Sample from the Rashomon set boundary just in case
        w_boundary = self.ellipsoid.sample_boundary(n_samples, epsilon)
        # Rescale the instance
        x_instances = (x_instances - self.X_mean) / self.X_std
        # Work with a specific contrastive question
        if background is not None:
            x_instances -= (background.mean(0) - self.X_mean) / self.X_std

        # Features to which to attribute a change in model output
        if idxs is None:
            idxs = [[i] for i in range(self.n_features)]
        d = len(idxs)

        ### Range of the Attributions across the Rashomon Set ###
        lstsq_attrib = np.zeros((N, d))
        all_attribs = np.zeros((N, d, n_samples))
        for j in range(d):
            idxs_j = np.array(idxs[j])
            # Least-Square
            lstsq_attrib[:, j] = np.sum(x_instances[:, idxs_j] * self.w_hat[idxs_j+1].T, axis=1)
            # Boundary
            all_attribs[:, j, :] = x_instances[:, idxs_j].dot(w_boundary[:, idxs_j+1].T)
        return self.y_std * lstsq_attrib, self.y_std * all_attribs



    def feature_attributions(self, x_instances, background=None, idxs=None, top_bottom=True):
        """ 
        Compute the Local Feature Attributions (LFA) for a set of samples `x_instances`.
        These LFAs are encoded as a `RashomonPartialOrders` object.

        Parameters
        ----------
        x_instances: (n_samples, n_features) `np.array`
            Instances on which to compute the LFA
        idxs: List(List(int)), default=None
            Features organised as groups when we want the LFA for a coallition of features.
            For instance, to group the first three and last three features togheter, provide
            `idxs=[[0, 1, 2], [3, 4, 5]]`. When `idxs=None`, each feature is its own coallition
            and `idxs=[[0], [1], [2], [3], [4], [5]]` will be set automatically.

        Returns
        -------
        PO : RashomonPartialOrders
            Object encoding all partial orders for al instances at any tolerance level epsilon

        Examples
        --------
        >>> from uxai.linear import LinearRashomon
        >>> from uxai.plots import bar
        >>> linear_rashomon = LinearRashomon()
        >>> linear_rashomon.fit(X, y)
        >>> # Get the train performance
        >>> RMSE = linear_rashomon.RMSE
        >>> # Get epsilon for an extra 0.05 RMSE tolerance
        >>> epsilon = linear_rashomon.get_epsilon(RMSE + 0.05)
        >>> # Compute the LFA
        >>> rashomon_po = linear_rashomon.feature_attributions(X)
        >>> # Get the partial order on instance i with tolerance epsilon
        >>> PO = rashomon_po.get_poset(i, epsilon, feature_names=["x1", "x2"])
        """

        if x_instances.ndim == 1:
            x_instances = x_instances.reshape((1, -1))
        assert x_instances.shape[1] == self.n_features, "Number of features is not the same as in training"
        N = x_instances.shape[0]

        # Rescale the instance
        x_instances = (x_instances - self.X_mean) / self.X_std
        # Work with a specific contrastive question
        if background is not None:
            x_instances -= (background.mean(0) - self.X_mean) / self.X_std
        
        ### Range of the Gap across the Rashomon Set ###
        x_instance_tilde = np.vstack( (np.zeros((1, N)), x_instances.T) ) * self.y_std
        gap_lstsq = x_instance_tilde.T.dot(self.w_hat)[:, 0]
        minmax_gap = self.ellipsoid.opt_linear(x_instance_tilde)
        gap_eps = (gap_lstsq / (minmax_gap[:, 1] - gap_lstsq)) ** 2

        # Features to which to attribute a change in model output
        if idxs is None:
            idxs = [[i] for i in range(self.n_features)]
        d = len(idxs)


        ### Range of the Attributions across the Rashomon Set ###
        lstsq_attrib = np.zeros((N, d))
        minmax_attribs = np.zeros((N, d, 2))
        for j in range(d):
            idxs_j = np.array(idxs[j])
            # Least-Square
            lstsq_attrib[:, j] = np.sum(x_instances[:, idxs_j] * self.w_hat[idxs_j+1].T, axis=1)
            # Min-Max
            a = np.zeros((self.n_features+1, N))
            a[np.array(idxs[j]) + 1, :] = self.y_std * x_instances[:, idxs_j].T
            minmax_attribs[:, j, :] = self.ellipsoid.opt_linear(a)
        lstsq_attrib = self.y_std * lstsq_attrib
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
                    idxs_i = np.array(idxs[i])
                    idxs_j = np.array(idxs[j])
                    # Least-Square difference in importance
                    lstsq_ij_diff = np.abs(lstsq_attrib[:, i]) - np.abs(lstsq_attrib[:, j])
                    a = np.zeros((N, self.n_features+1))
                    a[:, idxs_i+1] =  np.sign(lstsq_attrib[:, [i]]) * x_instances[:, idxs_i]
                    a[:, idxs_j+1] = -np.sign(lstsq_attrib[:, [j]]) * x_instances[:, idxs_j]
                    
                    # Min-max difference in importance
                    min_max_ij_diff = self.ellipsoid.opt_linear(a.T)
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
        proj_ellipsoid = self.ellipsoid.projection([feature1+1, feature2+1])
        zz = proj_ellipsoid.get_ellipse_border(epsilon)
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




class MonotonicRashomon(object):
    """
    Rashomon Set for Piece-Wise Linear Monitonic Regression.

    MonotonicRashomon fits a Piece-Wise linear model with with monotocity constraints.
    The resulting Rashomon Set is the intersection of an ellipsoid
    `(alpha - alpha_S)^T A (alpha - alpha_S) <= epsilon`
    and the quadrant {alpha : alpha_i >= 0 for all i}.


    Attributes
    ----------
    alpha_hat : (n_features+1, 1) `np.array`
        Estimated bias and coefficients for the piece-wiselinear regression problem.

    MSE : `float`
        Mean-Squared-Error of w_hat

    RMSE : `float`
        Root-Mean-Squared-Error of w_hat

    """
    def __init__(self, knots, monotocity):
        self.n_features = len(knots)
        self.n_basis_per_feature = np.array([len(i)-1 for i in knots])
        self.n_basis = np.sum(self.n_basis_per_feature)+1
        self.knots = knots
        self.monotocity = np.concatenate([[0]] + 
                                         [monotocity[i] * np.ones(self.n_basis_per_feature[i]) \
                                          for i in range(self.n_features)])
        self.mappers = [SplineTransformer(degree=1, knots=knot[:, np.newaxis],
                                          include_bias=False) for knot in knots]
        self.ellipsoid_dict = {}


    def map(self, X):
        """ Map raw features to a form consistent with prediction with alpha coeffs"""
        X_map = np.zeros((X.shape[0], self.n_basis-1))
        curr = 0
        # Compute piece-wise linear basis
        for i in range(X.shape[1]):
            X_map[:, curr:curr+self.n_basis_per_feature[i]] = \
                                self.mappers[i].transform(X[:, [i]])
            curr += self.n_basis_per_feature[i]
        # Compute cumsum to express predictions as a linear
        # function of the positive step coefficients alpha
        X_map = intervaled_cumsum(X_map, self.n_basis_per_feature)
        return X_map


    def fit(self, X, y):
        """
        Fit piece-wise linear model.

        Parameters
        ----------
        X : (n_samples, n_features) `np.array`
            Input values.
        y : (n_samples,) `np.array`
            Target values.

        Returns
        -------
        self : object
            Fitted Estimator.
        
        Examples
        --------
        >>> from uxai.linear import LinearRashomon
        >>> linear_rashomon = LinearRashomon()
        >>> linear_rashomon.fit(X, y)
        >>> # Get the train performance
        >>> RMSE = linear_rashomon.RMSE
        >>> # Get the model coefficients
        >>> coefs = linear_rashomon.w_hat[1:, 0]
        array([0.5517, 0.4964])
        """
        self.N = X.shape[0]
        assert X.shape[1] == self.n_features
        for i in range(self.n_features):
            self.mappers[i].fit(X[:, [i]])
        X = self.map(X)

        # Rescale Target
        self.X_mean = X.mean(0)
        self.y_mean = y.mean()
        self.y_std = y.std()
        y_ = (y - self.y_mean) / self.y_std

        # Fit the optimal model on training data
        X_tilde = np.column_stack((np.ones(self.N), X))
        alpha_S = lstsq(X_tilde, y_)[0]
        # If some monotocity constraints are broken then
        # refit with hard equality constraints
        if ((alpha_S * self.monotocity) > 0).any():
            broken_constraint = np.where((alpha_S * self.monitocity) > 0)[0]
            respect_constraint = [i for i in range(X.shape[0]) if i not in broken_constraint]
            alpha_S[broken_constraint] = 0
            alpha_S[respect_constraint] = lstsq(X_tilde[:, respect_constraint], y_)[0]
        self.alpha_S = alpha_S.reshape((-1, 1))
        preds = X_tilde.dot(self.alpha_S)
        
        # Unscaled MSE
        self.uMSE = mean_squared_error(y_, preds)
        self.RMSE = float(self.y_std * np.sqrt(self.uMSE))
        self.MSE = self.RMSE ** 2
        
        # Define Rashomon Set
        X_tilde = np.column_stack( (np.ones(len(X)), X) )
        A = X_tilde.T.dot(X_tilde) / self.N
        ellipsoid = Ellipsoid(A, self.alpha_S)
        self.ellipsoid_dict[()] = ellipsoid

        return self


    def get_epsilon(self, target_RSME):
        """ Get the epsilon required to reach a given RSME """
        return (target_RSME / self.y_std)**2 - self.uMSE


    def predict(self, X, epsilon=None, constraint=True):
        """ 
        Return point prediction of Least Square and [Min, Max] preds of Rashomon Set

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
        X = self.map(X)
        y_ = X.dot(self.alpha_S)

        if epsilon is None:
            return self.y_std * y_ + self.y_mean
        
        minmax_preds = opt_lin_ellipsoid_quadrant(X.T, self.ellipsoid_dict, self.monotocity, 
                                                  epsilon, constraint=constraint)


    def partial_dependence(self, x, idx, epsilon, constraint=True):
        basis_start = np.sum(self.n_basis_per_feature[:idx])
        basis_stop = basis_start + self.n_basis_per_feature[idx]
        x = np.cumsum(self.mappers[idx].transform(x.reshape((-1, 1))), 1)
        x = x - self.X_mean[basis_start:basis_stop]
        preds = x.dot(self.alpha_S[basis_start+1:basis_stop+1, 0])
        a = np.zeros((self.n_basis, len(x)))
        a[basis_start+1:basis_stop+1] = x.T
        if constraint:
            minmax_preds = opt_lin_ellipsoid_quadrant(a, self.ellipsoid_dict, self.monotocity, epsilon)
        else:
            minmax_preds = self.ellipsoid_dict[()].opt_linear(a, epsilon)
        return self.y_std * preds, self.y_std * minmax_preds


    # def feature_attributions(self, x_instances, top_bottom=True):
    #     """ 
    #     Compute the Local Feature Attributions (LFA) for a set of samples `x_instances`.
    #     These LFAs are encoded as a RashomonPartialOrders object.

    #     Parameters
    #     ----------
    #     x_instances: (n_samples, n_features) `np.array`
    #         Instances on which to compute the LFA
    #     idxs: List(List(int)), default=None
    #         Features organised as groups when we want the LFA for a coallition of features.
    #         For instance, to group the first three and last three features togheter, provide
    #         `idxs=[[0, 1, 2], [3, 4, 5]]`. When `idxs=None`, each feature is its own coallition
    #         and `idxs=[[0], [1], [2], [3], [4], [5]]` will be set automatically.

    #     Returns
    #     -------
    #     PO : RashomonPartialOrders
    #         Object encoding all partial orders for al instances at any tolerance level epsilon

    #     Examples
    #     --------
    #     >>> from uxai.linear import LinearRashomon
    #     >>> from uxai.plots import bar
    #     >>> linear_rashomon = LinearRashomon()
    #     >>> linear_rashomon.fit(X, y)
    #     >>> # Get the train performance
    #     >>> RMSE = linear_rashomon.RMSE
    #     >>> # Get epsilon for an extra 0.05 RMSE tolerance
    #     >>> epsilon = linear_rashomon.get_epsilon(RMSE + 0.05)
    #     >>> # Compute the LFA
    #     >>> rashomon_po = linear_rashomon.feature_attributions(X)
    #     >>> # Get the partial order on instance i with tolerance epsilon
    #     >>> PO = rashomon_po.get_poset(i, epsilon, feature_names=["x1", "x2"])
    #     """

    #     if x_instances.ndim == 1:
    #         x_instances = x_instances.reshape((1, -1))
    #     assert x_instances.shape[1] == self.n_features, "Number of features is not the same as in training"
    #     N = x_instances.shape[0]

    #     # Rescale the instance
    #     x_instances = (x_instances - self.X_mean) / self.X_std
        
    #     ### Range of the Gap across the Rashomon Set ###
    #     x_instance_tilde = np.vstack( (np.zeros((1, N)), x_instances.T) ) * self.y_std
    #     gap_lstsq = x_instance_tilde.T.dot(self.w_hat)[:, 0]
    #     minmax_gap = opt_lin_ellipsoid(x_instance_tilde, self.A_half_inv, self.w_hat)
    #     gap_eps = (gap_lstsq / (minmax_gap[:, 1] - gap_lstsq)) ** 2

    #     # Features to which to attribute a change in model output
    #     if idxs is None:
    #         idxs = [[i] for i in range(self.n_features)]
    #     d = len(idxs)


    #     ### Range of the Attributions across the Rashomon Set ###
    #     lstsq_attrib = np.zeros((N, d))
    #     minmax_attribs = np.zeros((N, d, 2))
    #     for j in range(d):
    #         idxs_j = np.array(idxs[j])
    #         # Least-Square
    #         lstsq_attrib[:, j] = np.sum(x_instances[:, idxs_j] * self.w_hat[idxs_j+1].T, axis=1)
    #         # Min-Max
    #         a = np.zeros((self.n_features+1, N))
    #         a[np.array(idxs[j]) + 1, :] = self.y_std * x_instances[:, idxs_j].T
    #         minmax_attribs[:, j, :] = opt_lin_ellipsoid(a, self.A_half_inv, self.w_hat)
    #     lstsq_attrib = self.y_std * lstsq_attrib
    #     minmax_attribs_lambda = lambda eps: np.sqrt(eps)*(minmax_attribs - lstsq_attrib[:, :, np.newaxis]) \
    #                                             + lstsq_attrib[:, :, np.newaxis]


    #     # Determine critical epsilon values for positive, 
    #     # negative attribution statements
    #     pos_eps = np.zeros((N, d))
    #     neg_eps = np.zeros((N, d))
    #     for j in range(d):
    #         step = minmax_attribs[:, j, 1] - lstsq_attrib[:, j]
    #         critical_eps = (lstsq_attrib[:, j] / step) ** 2
    #         pos_idx = lstsq_attrib[:, j] > 0
    #         pos_eps[pos_idx, j] = critical_eps[pos_idx]
    #         neg_idx = lstsq_attrib[:, j] < 0
    #         neg_eps[neg_idx, j] = critical_eps[neg_idx]


    #     # Compare features by features to make partial order
    #     adjacency_eps = np.zeros((N, d, d))
    #     for i in range(d):
    #         for j in range(d):
    #             if i < j:
    #                 idxs_i = np.array(idxs[i])
    #                 idxs_j = np.array(idxs[j])
    #                 # Least-Square difference in importance
    #                 lstsq_ij_diff = np.abs(lstsq_attrib[:, i]) - np.abs(lstsq_attrib[:, j])
    #                 a = np.zeros((N, self.n_features+1))
    #                 a[:, idxs_i+1] =  np.sign(lstsq_attrib[:, [i]]) * x_instances[:, idxs_i]
    #                 a[:, idxs_j+1] = -np.sign(lstsq_attrib[:, [j]]) * x_instances[:, idxs_j]
                    
    #                 # Min-max difference in importance
    #                 min_max_ij_diff = opt_lin_ellipsoid(a.T, self.A_half_inv, self.w_hat)
    #                 min_max_ij_diff *= self.y_std
    #                 step = min_max_ij_diff[:, 1] - lstsq_ij_diff
    #                 # Critical epsilon for comparing relative importance
    #                 ij_idx = lstsq_ij_diff < 0
    #                 adjacency_eps[ij_idx, i, j] = (lstsq_ij_diff[ij_idx] / step[ij_idx]) ** 2
    #                 ji_idx = lstsq_ij_diff > 0
    #                 adjacency_eps[ji_idx, j, i] = (lstsq_ij_diff[ji_idx] / step[ji_idx]) ** 2
                
    #     PO = RashomonPartialOrders(lstsq_attrib, minmax_attribs_lambda, gap_eps, 
    #                                neg_eps, pos_eps, adjacency_eps, top_bottom)
    #     return PO
