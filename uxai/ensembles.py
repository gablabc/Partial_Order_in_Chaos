""" Deep Neural Networks Ensembles """
from math import sqrt
import numpy as np
import torch
from torch import nn
from torch.autograd import grad
from dataclasses import dataclass

from simple_parsing import ArgumentParser


ACTIVATION_MAPPING = {
    "ReLU" : nn.ReLU(),
    "SeLU" : nn.SELU(),
    "Sigmoid" : nn.Sigmoid(),
    "Tanh" : nn.Tanh(),
}


# Parameters:
#     x (Tensor): Input of the network of size NbExemples X NbDimensions
#     input_dim (Int): Dimensions of NN's inputs (=NbDimensions)
#     layerwidth (Int or List): Number of hidden units per layer
#     activation (Module/Function): activation function of the neural network

# Returns:
#     Tensor: Predictions Tensor with dimensions NbModels X NbExemples X NbDimensions

# Example:

# input_dim=11
# activation=nn.Tanh()
# layerwidth = [20, 20, 20]
# nblayers = 3  (derived from layerwidth)
# param_count = (input_dim+1)*layerwidth+(nblayers-1)*(layerwidth**2+layerwidth)+layerwidth+1

# x=torch.rand(3,input_dim)
# theta=torch.rand(5,param_count)
# model=mlp(input_dim=input_dim,layerwidth=layerwidth,nb_layers=nblayers,activation=activation)
# model(x,theta)

class mlp(nn.Module):
    """
    Low level implementation of a Multi-Layered Perceptron optimized for 
    GPU parallelization.
    """
    def __init__(self, input_dim, layerwidths, activation):
        super().__init__()
        
        self.input_dim = input_dim
        self.layerwidths = layerwidths
        
        if type(self.layerwidths) == list:
            self.nblayers = len(self.layerwidths)
        elif type(self.layerwidths) == int:
            self.nblayers = 1
            self.layerwidths = [self.layerwidths]
        else:
            raise Exception("provide either a integer of list as layerwidths")


        self.activation = activation
        self.split_sizes = [self.input_dim * self.layerwidths[0]] + [self.layerwidths[0]]
        self.param_count_per_layer = [(self.input_dim + 1) * self.layerwidths[0]]
        
        for i in range(1, self.nblayers):
            self.split_sizes += [self.layerwidths[i - 1] * self.layerwidths[i], self.layerwidths[i]]
            self.param_count_per_layer += [self.layerwidths[i - 1] * self.layerwidths[i] + self.layerwidths[i]]
            
        self.split_sizes += [self.layerwidths[-1], 1]
        self.param_count_per_layer += [self.layerwidths[-1]+ 1]


        self.in_features = [self.input_dim] + self.layerwidths
        self.param_count = np.sum(self.param_count_per_layer)
        


    def forward(self, x, theta):
        # Split parameters
        nb_theta = theta.shape[0]
        nb_x = x.shape[0]

        theta = theta.split(self.split_sizes, dim=1)
        
        # Forward pass through the network
        input_x = x.view(nb_x, self.input_dim, 1)
        m = torch.matmul(theta[0].view(
                        nb_theta, 1, self.layerwidths[0], self.input_dim), input_x)
        m = m.add(theta[1].view(nb_theta, 1, self.layerwidths[0], 1))
        m = self.activation(m)
        for i in range(self.nblayers - 1):
            m = torch.matmul(
                theta[2 * i + 2].view(-1, 1, self.layerwidths[i + 1], self.layerwidths[i]), m)
            m = m.add(theta[2 * i + 3].view(-1, 1, self.layerwidths[i + 1], 1))
            m = self.activation(m)
        m = torch.matmul(
            theta[2 * (self.nblayers - 1) + 2].view(nb_theta, 1, 1, self.layerwidths[-1]), m)
        m = m.add(theta[2 * (self.nblayers - 1) + 3].view(nb_theta, 1, 1, 1))

        return m.squeeze(-1)




class Ensemble(nn.Module):
    """
    High level class representing an ensemble of Neural Networks. Used as a base 
    class for more sophisticated ensembles through inheritance.
    """
    @dataclass
    class HParams():
        """
        Args:
            size_ensemble (int, optional): Number of models in the ensemble. Defaults to 5.
            input_dim (int, optional): Input dimension, inferred from data when possible. Defaults to 1.
            task (string, optional): Type of task. Specified by the name of the dataset. Default is "regression".
            layerwidths (string, optional): Layerwidths of the ensemble models. Help give a list of int in the form "100,50,20,10". Defaults to "100,50,20,10".
            activation (string, optional): Activation function to use. Defaults to "ReLU".
            sigma_init (float, optional): Scale for Gaussian initialisation if pytorch_init=False. Defaults to 1.
            pytorch_init (bool, optional): To use Pytorch initialisation. Defaults to True.
            regul_lambda (float, optional): L2 regularisation. Defaults to 0.
        """
        size_ensemble: int = 5
        input_dim: int = 1
        task: str = "regression"
        layerwidths: str =  ""
        def __post_init__(self):  # Hack
            if type(self.layerwidths)==str:
                    self.layerwidths: list = [100, 50, 20, 10] \
                                                    if self.layerwidths == "" \
                              else [int(i) for i in self.layerwidths.split(",")]

        activation: str = "ReLU"
        sigma_init: float = 1.
        pytorch_init: bool = True
        regul_lambda: float = 0
        device: str = "Default"
        aggregate: bool = False


    def __init__(self, hparams: HParams=None, **kwargs):
        """
        Initialization of the Ensemble class. The user can either give a premade 
        hparams object made from the Hparams class or give the keyword arguments 
        to make one.

        Args:
            hparams (HParams, optional): HParams object to specify the models 
            characteristics. Defaults to None.
        """
        self.hparams= hparams or self.HParams(**kwargs)  
    # def __init__(self, size_ensemble, input_dim, layerwidths, activation, 
    #                                       sigma_init = 1, pytorch_init = False):
        super().__init__()
        # Unless device is specified, use GPUs if available
        if self.hparams.device == "Default":
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.hparams.device)
    
        self.predictor = mlp(self.hparams.input_dim, 
                             layerwidths=self.hparams.layerwidths, 
                             activation=ACTIVATION_MAPPING[self.hparams.activation])
        
        if self.hparams.task == "classification":
            self.bce = nn.BCEWithLogitsLoss()
        
        self.preprocess= None
        self.postprocess = None
        self.param_count = self.predictor.param_count
        self.init_parameters()


    def init_parameters(self):
        """
        Initialization of all the Ensemble parameters in the nn.Parameter object 
        self.all_params.
        """
        if self.hparams.pytorch_init:
            param_per_layers = self.predictor.param_count_per_layer
            k_list = [np.sqrt(1 / k) for k in self.predictor.in_features]
            parameters_per_layers = [2 * k * torch.rand(self.hparams.size_ensemble, n) - k\
                                     for n, k in zip(param_per_layers, k_list)]
            parameters = torch.cat(parameters_per_layers, dim = 1)
        else:
            parameters = self.hparams.sigma_init * torch.randn(self.hparams.size_ensemble, 
                                                       self.param_count)

        self.all_params = nn.Parameter(parameters.to(self.device))

    def model_set_selection(self, select_models):
        self.hparams.size_ensemble = len(select_models)
        self.all_params = nn.Parameter(self.all_params[select_models])
    
    @property
    def name(self):
        return 'Ensemble'

    def aggregate(self, mode=True):
        self.hparams.aggregate = mode
        self.hparams.size_ensemble = 1 if mode else self.all_params.shape[0]

    # forward pass through the whole ensemble
    def forward(self, X):
        """Forward pass through the whole ensemble.

        Args:
            X (Tensor): Tensor (batch_size, nb_features) to pass through the ensemble.

        Returns:
            Tensor: Tensor (nb_models, batch_size, 1) when self.aggregate is False and
            (1, batch_size, 1) otherwise. Contains the predictions made by the ensemble.
        """

        # Fast forward pass on GPU
        X = X.to(self.device)

        # Pre-process the input
        if not self.preprocess is None:
            X = self.preprocess.transform(X)

        y_pred = self.predictor(X, self.all_params)

        # Post-process the output
        if not self.postprocess is None:
            y_pred = self.postprocess.inv_transform(y_pred)

        if self.hparams.aggregate:
            return y_pred.mean(0).unsqueeze(0)
        else:
            return y_pred
        
    
    # Generic loss function
    def loss(self, X, y):
        """
        Calculates the loss of an (X, y) batch.

        Args:
            X (Tensor): x values of the batch in a (batch_size, n_features).
            y (Tensor): y values of the batch in a (batch_size, 1).

        Returns:
            Tensor: Loss of the whole batch.
        """
        y_pred = self(X)
        y = y.to(self.device)
        
        if self.hparams.task == 'regression':
            loss = (y_pred - y).pow(2).mean()
        elif self.hparams.task == 'classification':
            loss = self.bce(y_pred, y.expand(self.hparams.size_ensemble, -1, -1))
        
        # Regularisation
        if self.hparams.regul_lambda:
            weights = self.all_params.split(self.predictor.split_sizes, dim = 1)[::2]
            for weight in weights: 
                loss += self.hparams.regul_lambda * torch.norm(weight, 'fro') ** 2
        
        return loss


    # Gradient computation for each network in the ensemble
    def gradients(self, X, theta=None, output_value=False):
        """
        Gradient computation for each network in the ensemble.

        Args:
            X (Tensor): Tensor (batch_size, nb_features) where to compute gradients.
            theta (Tensor, optional): Alternative weights and biases of the models. Defaults to None. 
            output_value (bool, optional): Return the ensemble output if set to True. Defaults to False.

        Returns:
            (tuple): tuple containing:
                Tensor: The gradients Tensor. (nb_models, batch_size, nb_features)
                Tensor: The ensemble output Tensor (nb_models, nb_features) 
        """
        X.requires_grad_()
        if theta is not None:
            N = theta.shape[0]
            output = self.predictor(X, theta)

        else:
            N = self.hparams.size_ensemble
            output = self(X)

        #  print(N, output.shape)
        gradient_list = []
        for i in range(N):
            gradient_list.append(grad(
                    outputs = output[i].sum(), 
                    inputs = X,
                    create_graph = True, 
                    retain_graph = True)[0])
        gradients = torch.stack(gradient_list)

        # print(f"grad shape:{gradients.shape}")
        gradients = gradients.view(N, X.shape[0], -1)
        X.requires_grad_(False)

        if output_value:
            return output.squeeze(-1).detach(), gradients
        else:
            return gradients


    @classmethod
    def from_argparse_args(cls, args):
        """Creates an instance of the Class from the parsed arguments."""
        hparams: cls.HParams = args.model
        return cls(hparams)


    @classmethod
    def add_argparse_args(cls, parser: ArgumentParser):
        """Adds command-line arguments for this Class to an argument parser."""
        parser.add_arguments(cls.HParams, "model")



def evaluate_ensemble(models, data_loader, return_predictions=False):
    """ 
    Evaluate each MLP of the ensemble on multiple datasets
    
    Parameters
    ----------
        models: `uxai.ensembles.Ensemble`
            fitted models to evaluate
        data_loaders: (m,) `List(torch.DataLoader)`
            Various dataloader to compute the performance on
        return_predictions: `bool`, default=False
            Whether or not to return the predictions on the 
            provided data

    Returns
    -------
        errors: (n_models+1, m) `torch.tensor`
            For each model, and the average model, return the empirical error
        pred_list: (m,) `List(torch.tensor)`
            Predictions of all models on all m datasets. Only returns if
            `return_predictions=True`.
    """
    size_ensemble = 0 if models.hparams.aggregate else models.hparams.size_ensemble
    if type(data_loader) == list:
        data_loaders = data_loader
        errors = torch.zeros(size_ensemble + 1, len(data_loader))
    else:
        data_loaders = [data_loader]
        errors = torch.zeros(size_ensemble + 1, 1)

    with torch.no_grad():
        pred_list=[]
        for i, data_loader in enumerate(data_loaders):
            predictions=[]
            n_examples = len(data_loader.dataset)
            for (x, y) in data_loader:
                y_pred = models(x)
                y_pred = y_pred.cpu()
                if return_predictions:
                    predictions.append(y_pred)
    
                if models.hparams.task == "regression":
                    # Evaluate individual models
                    if size_ensemble > 0:
                        errors[:-1, [i]] += (y_pred - y).pow(2).sum(dim=1)
                    # Aggregate Model
                    mean_pred = torch.mean(y_pred, dim=0)
                    errors[-1, i] += (mean_pred - y).pow(2).sum(dim=0).item()
                else:
                    y_pred = torch.sigmoid(y_pred)
                    # Evaluate individual models
                    if size_ensemble > 0:
                        pred = (y_pred >= 0.5).int()
                        errors[:-1, [i]] += (pred != y).float().sum(dim=1)
                    # Aggregate Model
                    mean_pred = torch.mean(y_pred, dim=0) >= 0.5
                    errors[-1, i] += (mean_pred != y).float().sum(dim=0).item()
            if return_predictions:
                pred_list.append(torch.cat(predictions, 1))
            errors[:, i] /= n_examples

    if models.hparams.task == "regression":
        # Take RMSE
        errors = torch.sqrt(errors)
    else:
        errors *= 100
    if return_predictions:
        return errors, pred_list
    return errors
