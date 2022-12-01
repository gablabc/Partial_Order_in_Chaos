import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from functools import partial

from .features import embed


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



class EmbedderDataLoader:
    """ 
    A custom DataLoader that loads num (float) and cat (int) 
    feature separately, embeds categorical features, and concatenates them.
    """
    def __init__(self, dataset, embeddings, **kwargs):
        self.data_loader = DataLoader(dataset, **kwargs)
        self.dataset = dataset
        self.embeddings = embeddings

    def embedding_iterator(self):
        for x_num, x_cat, y in self.data_loader:
            embedded_instances = []
            for i, embedding in enumerate(self.embeddings):
                embedded_instances.append(embedding[x_cat[:, i], :])
            yield torch.cat([x_num] + embedded_instances, dim=1), y

    def __iter__(self):
        return self.embedding_iterator()



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
