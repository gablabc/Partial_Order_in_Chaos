import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from functools import partial

from .features import embed


def shur_complement(A, idx):
    """ 
    Take the Schur Complement of A[idx, idx]
    
    Parameters
    ----------
    A: (d, d) `np.array`
    idxs: `List(int)`
        Indices of the columns w.r.t which we compute the complement

    Returns
    -------
    A_schur: (len(idx), len(idx)) `np.array`
        The Schur complement
    """
    assert type(idx) in [list, np.ndarray], "idxs must be a list or np.array"
    N = A.shape[0]
    select = np.array([idx])
    # select must be a (1, N) np.array
    assert select.shape == (1, len(idx)), "idxs must be a list of indices"
    non_select = np.array([[f for f in range(N) if f not in select]])
    J = A[select.T, select]
    L = A[select.T, non_select]
    K = A[non_select.T, non_select]
    # Take the schur complement
    A_shur = J - L.dot(np.linalg.inv(K).dot(L.T))
    return A_shur


class Ellipsoid(object):
    """ 
    Representation of an ellipsoid
    `(x - mu)^T A (x-mu) <= size_fun(epsilon)`
    """
    def __init__(self, A, mu, size_fun=None):
        self.d = A.shape[0]
        self.A = A
        self.mu = mu
        self.A_half = np.linalg.cholesky(self.A)
        self.A_half_inv = np.linalg.inv(self.A_half)
        if size_fun is None:
            size_fun = lambda eps: eps
        else:
            assert callable(size_fun), "Size_fun must be a callable"
        self.size_fun = size_fun
    

    def projection(self, idxs):
        """ 
        Project the Ellipsoid on the components in idxs
        
        Parameters
        ----------
        idxs: `List(int)`
            Indices of the components on which to project

        Returns
        -------
        proj_ellipsoid: `Ellipsoid`
            The projected ellipsoid
        """
        proj_ellipsoid = Ellipsoid(shur_complement(self.A, idxs), self.mu[idxs], self.size_fun)
        return proj_ellipsoid
    

    def slice(self, idxs_bar):
        """ 
        Slice the Ellipsoid along a linear subspace x_i=0 i\in idxs
        
        Parameters
        ----------
        idxs: `List(int)`
            Indices of the components we fix to zero

        Returns
        -------
        sliced_ellipsoid: `Ellipsoid`
            The sliced ellipsoid
        """
        # Index to keep and remove
        idxs = [i for i in range(self.d) if not i in idxs_bar]
        idxs = np.array(idxs).reshape((-1, 1))
        idxs_bar = np.array(idxs_bar).reshape((-1, 1))

        mu = self.mu
        half_step = self.A[idxs, idxs_bar.T].dot(mu[idxs_bar, 0])
        step = np.linalg.solve(self.A[idxs, idxs.T], half_step)
        # New mean
        new_mu = mu[idxs, 0] + step
        # New size
        S = shur_complement(self.A, idxs_bar.ravel())
        delta_size = float(mu[idxs_bar, 0].T.dot(S.dot(mu[idxs_bar, 0])))
        new_size = lambda eps : eps - delta_size
        return Ellipsoid(self.A[idxs, idxs.T], new_mu, new_size)


    def opt_linear(self, a, epsilon=1, return_input_sol=False):
        """ 
        Compute the min/argmin and max/argmax of `g(x) = a^t x`  with the
        constraint that 
        
        `(x - x_hat)^T A (x - x_hat) <= size(epsilon)`

        Parameters
        ----------
        a: (d, N) `np.array`
            N different linear functions to optimize in a vectorized fashion
        epsilon: `float`
            Size of the ellipsoid
        return_input_sol: `bool`, default=False
            Whether or not to return the argmin and argmax

        Returns
        -------
        sol_values: (N, 2) `np.array`
            Min/Max values of each of the N objectives
        sol_inputs: (2, N, d) `np.array`
            argmin/argmax inputs of each of the N objectives. Only returned
            if `return_input_sol=True`.
        """
        size = self.size_fun(epsilon)
        if size <= 0:
            raise Exception("Empty Ellipsoid")
        A_half_inv = self.A_half_inv * np.sqrt(size)
        a_prime = A_half_inv.dot(a).T # (N, d)
        norm_a_prime = np.linalg.norm(a_prime, axis=1, keepdims=True) # (N, 1)
        sol_values = np.array([-1, 1]) * norm_a_prime + a.T.dot(self.mu) # (N, 2)
        if not return_input_sol:
            return sol_values
        else:
            z_star = a_prime.dot(A_half_inv) / norm_a_prime # (N, d)
            z_star[np.isnan(z_star)] = 0
            sol_inputs = np.array([-1, 1]).reshape((2, 1, 1)) * z_star + self.mu.T # (2, N, d)
            return sol_values, sol_inputs
        

    def get_ellipse_border(self, epsilon):
        """ 
        Plot the border of the ellipse 
        Only works if the ellipsoid is 2D 
        (i.e. an ellipse)
        
        Parameters
        ----------
        epsilon: `float`
            Size of the ellipse

        Returns
        -------
        zz: (2, N) `np.array`
            N points on the border of the ellipsoid
        """
        assert self.d == 2, "The method only works for Ellipses"
        size = self.size_fun(epsilon)
        assert size > 0, "The ellipse is empty"
        theta = np.linspace(0, 2 * np.pi, 100)
        zz = np.vstack([np.cos(theta), np.sin(theta)])
        zz = self.A_half_inv.T.dot(zz) * np.sqrt(size) + self.mu
        return zz


    def verif_cross(self, a, b, epsilon):
        """ Does a^t x = b  cross the ellipse ? """
        size = self.size_fun(epsilon)
        assert size > 0, "The ellipse is empty"
        a_prime = self.A_half_inv.dot(a) * np.sqrt(size)
        b_prime = b - a.T.dot(self.mu)
        return bool(np.abs(b_prime) < np.linalg.norm(a_prime))
    

    def sample_boundary(self, N, epsilon):
        size = self.size_fun(epsilon)
        assert size > 0, "The ellipse is empty"
        z = np.random.normal(0, 1, size=(N, self.d))
        z = z / np.linalg.norm(z, axis=1, keepdims=True)
        w_boundary = z.dot(self.A_half_inv) * np.sqrt(size) + self.mu.T
        return w_boundary



def abs_interval(interval):
    """
    Map an interval trough the abs function
            \      /
             \    /
              \  /
               \/
            [----]
    """
    # The interval does not cross the origin
    if interval[0] * interval[1] > 0:
        interval = np.abs(interval)
        return np.array([np.min(interval), np.max(interval)])
    # The interval crosses the origin:
    else:
        return np.array([0, np.max(np.abs(interval))])



# https://stackoverflow.com/a/49179628/  by @Divakar
def intervaled_cumsum(ar, sizes):
    # Make a copy to be used as output array
    out = ar.copy()

    # Get cumumlative values of array
    arc = ar.cumsum(1)

    # Get cumsumed indices to be used to place differentiated values into
    # input array's copy
    idx = sizes.cumsum()

    # Place differentiated values that when cumumlatively summed later on would
    # give us the desired intervaled cumsum
    out[:, idx[0]] = ar[:, idx[0]] - arc[:, idx[0]-1]
    out[:, idx[1:-1]] = ar[:, idx[1:-1]] - np.diff(arc[:, idx[:-1]-1], 1)
    return out.cumsum(1)


# Mimic sklearn i/o
def numpy_forward_pass(X, models, batch_size, embeddings=None, scaler=None):
    """
    Transforms a numpy.array input to a torch.tensor float32 mounted on 
    selected device (e.g. CPU, GPU) to forward pass in batches to the ensemble.
    Moreover, nominal feature are embedded before being fed to the models.

    Args:
        X (np.array): Input for the MLPs.
        batch_size (int): Batch size.

    Returns:
        Tensor (np.array): MLPs predictions.
    """
    #POUR QUE FONCTION PYTORCH FONCTIONNE COMME UNE FONCTION SCIKIT LEARN
    if scaler is not None:
        X = scaler.scaler_x.transform(X)
    #wrapper
    #float32 Pytorch fonctionne pas avec des float64
    if embeddings:
        if X.shape[0] <= batch_size:
            X = embed(X, embeddings)
            out = models(X)
            out = out.squeeze(-1).T.detach().cpu().numpy()
        else:
            out = np.zeros((X.shape[0], models.hparams.size_ensemble))
            for i in range(int(np.ceil(X.shape[0]/batch_size))):
                b = i*batch_size
                e = (i+1)*batch_size
                X_batch = embed(X[b:e], embeddings)
                out[b:e, :] = models(X_batch).squeeze(-1).T.detach().cpu().numpy()
            
    else:
        X = torch.as_tensor(X, dtype=torch.float32)
        if X.shape[0] <= batch_size:
            out = models(X)
            out = out.squeeze(-1).T.detach().cpu().numpy()
        else:
            out = np.zeros((X.shape[0], models.hparams.size_ensemble))
            for i in range(int(np.ceil(X.shape[0]/batch_size))):
                out[i*batch_size:(i+1)*batch_size, :] = \
                    models(X[i*batch_size:(i+1)*batch_size, :]).squeeze(-1).T.detach().cpu().numpy()
    
    # Explain the sigmoid output
    if models.hparams.task == "classification":
        out = 1 / (1 + np.exp(-out))
    elif scaler is not None:
        out = scaler.inverse_transform_y(out)
    return out



def as_sklearn(models, batch_size, embeddings=None, scaler=None):
    """Wrap Ensemble class to allow the use of a Numpy array input instead of a Pytorch Tensor.

    Args:
        models (Ensemble (see ensembles.py)): Ensemble of models.
        batch_size (int): Batch size.

    Returns:
        partial: Wrapped ensemble class.
    """
    #models: class Ensemble
    return partial(numpy_forward_pass, models=models, 
                   batch_size=batch_size, embeddings=None, scaler=None)



def get_data_loader(X, y, features, batch_size, embeddings=None, shuffle=False):
    """ Return Pytorch dataloaders """
    
    # Modify the Data Loaders if embeddings categorical features
    if False:#features.nominal:
        
        split = features.nominal[0]
        # Convert to Pytorch Datasets
        dataset = TensorDataset(torch.as_tensor(X[:, :split]).float(),
                                torch.as_tensor(X[:, split:]).long(),
                                torch.as_tensor(y).unsqueeze(-1).float())
        data_loader = EmbedderDataLoader(dataset, embeddings,
                                         batch_size=batch_size, shuffle=shuffle)
    else:
        # Convert to Pytorch Datasets
        dataset = TensorDataset(X, y)
        # Get Data Loaders
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
    return data_loader
