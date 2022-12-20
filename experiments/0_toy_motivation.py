# %%
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif', 'serif':['Computer Modern Roman'], 'size':15})

import torch
import numpy as np
import pandas as pd
import random
from simple_parsing import ArgumentParser
from utils import SchulzLeik

import os, sys
sys.path.append(os.path.join(".."))

from uxai.ensembles import Ensemble, evaluate_ensemble
from uxai.methods import method_reduce_lr

# Parse arguments
parser = ArgumentParser()
parser.add_argument("--seed", type=int, default=5)
args, unknown = parser.parse_known_args()

seed = args.seed
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# %% Generate synthetic data

from torch.distributions import MultivariateNormal

rho = 0.75
cov = 1 * torch.eye(4)
cov[0, 1] = rho
cov[1, 0] = rho

limit = False

center = torch.zeros(4)#2*torch.ones(3)
m = MultivariateNormal(center, cov)

M = 1000
if limit:
    M = 10000
sigma_noise = 0.1

c_0 = -8.
c_1 = 0.
c_2 = 1.5

def f(x):
    return c_0 * (x[:, 0] - x[:, 1]).cos() * (x[:, 0] + x[:, 1]).cos()\
           + c_1*(x[:, 0] + x[:, 1]) ** 1 + c_2 * x[:, 2]

X = m.sample([int(M)]).squeeze()
y = f(X) + sigma_noise*torch.randn((M,))
y = y.reshape((-1, 1))


# %% Prepare training set

from sklearn.model_selection import train_test_split
from uxai.features import Features, PytorchStandardScaler
from uxai.utils import get_data_loader
seed = seed
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

names = ["x1", "x2", "x3", "x4"]
f_types = ["num"] * 4
features = Features(X, names, feature_types=f_types)
scaler_x = PytorchStandardScaler(4).fit(X)
scaler_y = PytorchStandardScaler(1).fit(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                    random_state=42)

batch_size = 100
# Scale training set for faster training
train_loader_s = get_data_loader(scaler_x.transform(x_train),
                                 scaler_y.transform(y_train), 
                                 features, batch_size, shuffle=True)
# Unscaled training and test for eval
train_loader = get_data_loader(x_train, y_train, features, batch_size, shuffle=False)
test_loader  = get_data_loader(x_test, y_test, features, batch_size, shuffle=False)

# %% Hyperparameters

# predictive model architecture
size_ensemble = 5
if limit:
    size_ensemble = 200
layerwidths = [50, 20, 10]
activation = "ReLU"

# Training constraints
n_epochs = 800
learning_rate = 0.001

# %% Train models

models = Ensemble(size_ensemble=size_ensemble, 
                  layerwidths=layerwidths, 
                  input_dim=x_train.shape[1],
                  activation=activation)
# Train
MyMethod = method_reduce_lr(n_epochs=n_epochs, 
                            learning_rate=learning_rate, 
                            use_scheduler=False)
MyMethod.apply(models, train_loader_s)
models.preprocess  = scaler_x.to(models.hparams.device)
models.postprocess = scaler_y.to(models.hparams.device)

# Test results
models.aggregate(False)
perfs = evaluate_ensemble(models, [train_loader, test_loader])
perf_df = pd.DataFrame(perfs.numpy(), columns=["Train", "Test"],
        index=[f"h_{i}" for i in range(models.hparams.size_ensemble)] + ['h_mean'])
print(perf_df)

# target stddev as a reference
print(f"Target standard deviation : {y.std():.4f}")

# %%
import math
# Point to explain
x_plain = math.pi / 2 * torch.Tensor([[1. ,1. ,1. ,1.]])
y_plain = f(x_plain)
feature_map = [f"x{i+1}=1.57" for i in range(4)]
latex_feature_map = [f"$x_{i+1}=1.57$" for i in range(4)]
y_pred = models(x_plain).detach().cpu()

# Useful prints
print(f"Point to explain : {x_plain}")
print(f"Predictions : {y_pred.squeeze()}")
print(f"Aggregated Prediction : {y_pred.mean():.4f}")
print(f"True target : {y_plain.item():.4f}")
error = torch.abs(y_pred - y_plain).max()
print(f"Max absolute error in the ensemble :{error.item():.4f}")

# Background reference ( all of the training points )
background_value = models(x_train).mean(1).detach().cpu()
print(f"Background values : {background_value.squeeze()}")
print(f"Background mean value : {background_value.mean().squeeze():.4f}")


# %%
import shap
from shap.maskers import Independent
from uxai.utils import as_sklearn

print("Compute exact SHAP attributions for each model")

# Exact computation
background = x_train
masker = Independent(background.numpy(), max_samples=background.shape[0])
# Wrap our models into a sklearn input-output API
models_wrap = as_sklearn(models, batch_size=batch_size)
explainer = shap.explainers.Exact(models_wrap, masker)

# %%
phis = explainer(x_plain.numpy()).values.squeeze().T
mean_phi = phis.mean(0)
std_phi = phis.std(0)
print(f"Attribution of aggregated model : {phis.mean(0)}")

# %%
from uxai.plots import pcp

print(phis)

model_idxs = np.argsort(perfs[:-1, 1])
ordered_test_perf = np.concatenate([perfs[model_idxs, 1], perfs[-1:, 1]])
pcp(phis[model_idxs, :], latex_feature_map, test_error=ordered_test_perf)

plt.ylim(0, 3)
plt.savefig(os.path.join("Images", "Motivation", f"attrib_{seed}.pdf"), bbox_inches='tight')
plt.show()

# %%

# How current methods propose to deal with the uncertainty
# of feature attributions
df = pd.DataFrame()

# Shaikana
df["Mean"] = mean_phi
df["Std"] = std_phi
df.index = latex_feature_map

# Schultz
mean_rank, ordinal_consensus = SchulzLeik(phis)
df["Mean rank"] = mean_rank
df["Ordinal Consensus"] = ordinal_consensus
print(df.to_latex())


# %% Total order aggregated model (mean attributions)
from uxai.partial_orders import intersect_total_orders
PO = intersect_total_orders(np.expand_dims(phis.mean(0),0), feature_map, threshold=0, attribution=True)
dot = PO.print_hasse_diagram()

filename = os.path.join("Images", "Motivation", f"total_{args.seed}")
dot.render(filename, format='pdf')
dot.render(filename, format='png')
dot

# %% Partial order of consensus
PO = intersect_total_orders(phis, feature_map, threshold=0., attribution=True)
dot = PO.print_hasse_diagram(show_ambiguous=False)

filename = os.path.join("Images", "Motivation", f"partial_{args.seed}")
dot.render(filename, format='pdf')
dot.render(filename, format='png')
dot
