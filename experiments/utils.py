""" 
General utility functions for experiments 

"""

import json
from math import sqrt
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass

from sklearn.metrics import roc_curve, auc
from scipy.stats import gaussian_kde, mannwhitneyu

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

# Local imports
from data_utils import DATASET_MAPPING, TASK_MAPPING, THRESHOLDS_MAPPING

# UXAI
import sys, os
sys.path.append(os.path.join('../..'))
from uxai.features import PytorchScaler, PytorchStandardScaler, PytorchMinMaxScaler, PytorchOHE
#from uxai.ensembles import Ensemble



def setup_pyplot_font(size=11):
    from matplotlib import rc
    rc('font',**{'family':'serif', 'serif':['Computer Modern Roman'], 'size':size})
    rc('text', usetex=True)
    from matplotlib import rcParams
    rcParams["text.latex.preamble"] = r"\usepackage{bm}\usepackage{amsfonts}"


############################## General Utilities ##############################

# Custom train/test split for reproducability (random_state is always 42 !!!)
def custom_train_test_split(X, y, task):
    if task == "regression":
        return train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)



def get_cross_validator(k, task, split_seed, split_type):
    # Train / Test split and cross-validator. Dont look at the test yet...
    if task == "regression":
        if split_type == "Shuffle":
            cross_validator = ShuffleSplit(n_splits=k, 
                                           test_size=0.1,
                                           random_state=split_seed)
        elif split_type == "K-Fold":
            cross_validator = KFold(n_splits=k, shuffle=True, 
                                    random_state=split_seed)
        else:
            raise ValueError("Wrong type of cross-validator")
        
    # Binary Classification
    else:
        if split_type == "Shuffle":
            cross_validator = StratifiedShuffleSplit(n_splits=k, 
                                                     test_size=0.1, 
                                                     random_state=split_seed)
        elif split_type == "K-Fold":
            cross_validator = StratifiedKFold(n_splits=k,
                                              shuffle=True,
                                              random_state=split_seed)
        else:
            raise ValueError("Wrong type of cross-validator")
    
    return cross_validator



def get_hp_grid(filename):

    def to_eval(string):
        if type(string) == str:
            split = string.split("_")
            if len(split) == 2:
                return split[1]
            else:
                return None
        else:
            return None

    hp_dict = json.load(open(filename, "r"))
    for key, value in hp_dict.items():
        # Iterate over list
        if type(value) == list:
            for i, element in enumerate(value):
                str_to_eval = to_eval(element)
                if str_to_eval is not None:
                    value[i] = eval(str_to_eval)
        # Must be evaluated
        if type(value) == str:
            str_to_eval = to_eval(value)
            if str_to_eval is not None:
                hp_dict[key] = eval(str_to_eval)
    return hp_dict



############################## Tree-based models ##############################
TREES = {
         "rf" : {"regression": RandomForestRegressor(), 
                 "classification": RandomForestClassifier()
                 },
         "gbt" : {"regression": GradientBoostingRegressor(), 
                  "classification": GradientBoostingClassifier()
                 }
        }



def setup_data_trees(name):
    X, y, features = DATASET_MAPPING[name]()
    task = TASK_MAPPING[name]
    
    if len(features.nominal) > 0:
        # One Hot Encoding
        ohe = ColumnTransformer([
                              ('id', FunctionTransformer(), features.non_nominal),
                              ('ohe', OneHotEncoder(sparse=False), features.nominal)])
        ohe.fit(X)
    else:
        ohe = None
        
    return X, y, features, task, ohe



def load_trees(name, model_name, reject=False):
        
    file_path = os.path.join("models", name, model_name)
    
    # Pickle model
    from joblib import load
    models_files = [f for f in os.listdir(file_path) if "joblib" in f]
    # Sort by seed value
    models_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    # print(models_files)
    models = []
    perfs = []
    for model_file in models_files:
        seed = int(model_file.split("_")[2].split(".")[0])
        models.append(load(os.path.join(file_path, model_file)))
        if model_name == "rf":
            models[-1].set_params(n_jobs=1)
        perf = pd.read_csv(os.path.join(file_path, f"perfs_seed_{seed}.csv")).to_numpy()
        perfs.append(perf)      
    perfs = np.array(perfs).squeeze()
    if reject:
        # Get performances
        threshold = THRESHOLDS_MAPPING[name]
        good_model_idx = np.where(perfs[:, 1] < threshold)[0]
        if len(good_model_idx) < len(perfs):
            print("Some models are very bad !!!")
        return [models[i] for i in good_model_idx], perfs[good_model_idx]
    else:
        return models, perfs        



def save_tree(model, args, perf_df):
    # Make folder for dataset models
    folder_path = os.path.join("models", args.data.name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    file_path = os.path.join(folder_path, args.model_name)
    # Make folder for architecture
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    state = args.ensemble.random_state
    # Pickle model
    from joblib import dump
    filename = f"random_state_{state}.joblib"
    dump(model, os.path.join(file_path, filename))

    # Save performance in CSV file
    perf_df.to_csv(os.path.join(file_path,f"perfs_seed_{state}.csv"), index=False)

    # Save model hyperparameters
    json.dump(model.get_params(), open(os.path.join(file_path,
                                       "hparams.json"), "w"), indent=4)



############################ Multi-layer Perceptrons ##########################
def setup_data_mlp(name, scaler="Standard"):
    X, y, features = DATASET_MAPPING[name]()
    task = TASK_MAPPING[name]
    
    # numpy -> torch
    X = torch.Tensor(X)
    y = torch.Tensor(y)
    input_dim = len(features)
    n_num_features = len(features.non_nominal)

    # Scalers
    scaler_x = PytorchMinMaxScaler(n_num_features, features.non_nominal) if scaler == "MinMax" \
               else PytorchStandardScaler(n_num_features, features.non_nominal)
    scaler_x.fit(X)
    if task == "regression":
        scaler_y = PytorchMinMaxScaler(1) if scaler == "MinMax" else PytorchStandardScaler(1)
        scaler_y.fit(y)
    else:
        scaler_y = None

    # OHE
    ohe_encoder = None
    if len(features.nominal) > 0:
        ohe_encoder = PytorchOHE()
        input_dim = ohe_encoder.fit(X, features)
    
    return X, y, features, scaler_x, scaler_y, ohe_encoder, task, input_dim



def load_mlps(name, model_name, seed=1, reject=False):
    
    # Folder containing the fitted models
    file_path = os.path.join("models", name, model_name)
    
    hparams = json.load( open(os.path.join(file_path, "architecture.json"), "r"))
    models = Ensemble(**hparams)
    models.preprocess = PytorchScaler(hparams["input_dim"]).to(models.device)
    if hparams["task"] == "regression":
        models.postprocess = PytorchScaler(1).to(models.device)
    models.load_state_dict(torch.load(os.path.join(file_path,
                                      f"models_seed_{seed}.pt"),
                                      map_location=models.device))
    perfs = pd.read_csv(os.path.join(file_path, f"perfs_seed_{seed}.csv")).to_numpy()
    
    if reject:
        # Get performances
        threshold = THRESHOLDS_MAPPING[name]
        good_model_idx = np.where(perfs[:, 1] < threshold)[0][:-1]
        ratio_good = 100 * len(good_model_idx) / models.hparams.size_ensemble
        print(f"{ratio_good:.1f} % of models have an error < {threshold}")
        return models, perfs, good_model_idx
    else:
        return models, perfs



def save_mlp(models, method, args, perf_df):
    # Make folder for dataset models
    folder_path = os.path.join("models", args.data.name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # When no name is provided, the name is the architecture
    if args.model_name is None:
        args.model_name = "mlp_" + ','.join([str(i) for i in args.model.layerwidths])
    
    file_path = os.path.join(folder_path, args.model_name)
    # Make folder for architecture
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    # Save models locally
    torch.save(models.state_dict(), os.path.join(file_path,
                f"models_seed_{args.method.seed}.pt"))

    # Save performances locally
    perf_df.to_csv(os.path.join(file_path, 
                f"perfs_seed_{args.method.seed}.csv"),index = False)

    # Save model architectures
    json.dump(models.hparams.__dict__, open(os.path.join(file_path,
                            "architecture.json"), "w"), indent=4)

    # Save learning method
    json.dump(method.hparams.__dict__, open(os.path.join(file_path,
                            "method.json"), "w"), indent=4)



# def evaluate_aggregate(models, valid_loader, scaler):
#     """ 
#     Evaluate the aggregated predictor on train and test sets
#     """

#     n_examples = len(valid_loader.dataset)
#     error = 0

#     with torch.no_grad():
#         for (x, y) in valid_loader:
#             y_pred = models(x)
#             y_pred = y_pred.cpu()

#             if models.hparams.task == "regression":
#                 # Aggregate the ensemble
#                 y_pred = torch.mean(y_pred, dim=0)
#                 error += (y_pred - y).pow(2).sum(dim=0).item()
#             else:
#                 y_pred = torch.mean(torch.sigmoid(y_pred), dim=0)
#                 pred = (y_pred >= 0.5).int()
#                 error += (pred != y).float().sum(dim=0).item()

#     error /= n_examples

#     if models.hparams.task == "regression":
#         # Take RMSE
#         error = scaler.invscale_target(sqrt(error))
#     else:
#         error *= 100

#     return error





def base_attribution(models, background_loader, scaler):
    E_h = torch.zeros(models.hparams.size_ensemble, 1)
    # Average over whole background

    with torch.no_grad():
        for (x, y) in background_loader:
            y_pred = models(x)
            y_pred = y_pred.cpu().squeeze()
            if models.hparams.task == "classification":
                y_pred = torch.sigmoid(y_pred)
            E_h += y_pred.sum(dim=1, keepdims=True)
    E_h /= len(background_loader.dataset)
    if models.hparams.task == "regression":
        E_h = scaler.inverse_transform_y(E_h)
        
    return E_h


############################## Distributional Shift ###########################

def U_tests_repeat(models, in_distr_var, ood_sampler, log_var, **kwargs):
    """ 
    Compute the log_var on multiple reruns of the explanation 
    distributions, show U-stats  p-vals with standard deviations and return
    the last run results
    """
        
    U_stats = np.array([0.0] * 10)
    p_vals  = np.array([0.0] * 10)
    # Repeat 10 times
    for i in range(10):
        ood_var = log_var(models, ood_sampler, **kwargs)
        # Significance test
        U_stats[i], p_vals[i] = \
            mannwhitneyu(ood_var, in_distr_var, alternative='greater')
    # Normalize the U stat
    U_stats /= (len(ood_var) * len(in_distr_var))
    print(f"U-stat : {U_stats.mean():.3f}  +- {U_stats.std():.3f} "+\
          f"with p-value {p_vals.mean():e} +- {p_vals.std():e}")
    
    return ood_var



def plot_hists(model_vars, labels):
    # Histogram
    plt.figure()
    cmap = plt.get_cmap("tab10")
    for i, log_var in enumerate(model_vars):
        plt.hist(log_var, bins=40, density=True, alpha=0.25,
                                                label=labels[i], color=cmap(i))
        xx = np.linspace(log_var.min(), log_var.max(), 100)
        plt.plot(xx, gaussian_kde(log_var).pdf(xx), color=cmap(i))
    plt.xlabel(r"$\log \Delta (\bm{x})$")
    plt.ylabel("Density")
    plt.legend(prop={'size': 11})
    
from matplotlib import rc
rc('font',**{'family':'serif', 'serif':['Computer Modern Roman'], 'size':15})
rc('text', usetex=True)
from matplotlib import rcParams
rcParams['text.latex.preamble']=r"\usepackage{bm}\usepackage{amsfonts}"


def plot_var_hists(model_vars, labels, path=None):
    # Histogram
    fig, ax=plt.subplots()
    cmap = plt.get_cmap("tab10")
    for i, log_var in enumerate(model_vars):
        log_var=log_var
        ax.hist(log_var, bins=40, density=True, alpha=0.25,
                                                label=labels[i], color=cmap(i))
        xx = np.linspace(log_var.min(), log_var.max(), 100)
        ax.plot(xx, gaussian_kde(log_var).pdf(xx), color=cmap(i))
    ax.set_xlabel(r"$\log \Delta (\bm{x})$")
    ax.set_ylabel("Density")
    ax.legend()
    if path is not None:
        fig.savefig(path, bbox_inches='tight')
        #plt.show()
        #plt.close()  


# def plot_AUROC(in_distr_var, out_distr_vars, labels):
#     plt.figure(figsize=(6, 6))
#     cmap = plt.get_cmap("tab10")
#     for i, out_distr_var in enumerate(out_distr_vars):
#         ground_truth = torch.cat( (torch.zeros(in_distr_var.shape),
#                                    torch.ones(out_distr_var.shape)) )
#         scores = torch.cat( (in_distr_var, out_distr_var) )
#         fpr, tpr, _ = roc_curve(ground_truth, scores)
#         roc_auc = auc(fpr, tpr)
                   
#         lw = 2
#         plt.plot(fpr, tpr, color=cmap(i),
#                  lw=lw, label=f"{labels[i]} AUC = {roc_auc:.3f}")
#     plt.plot([0, 1], [0, 1], color='k', lw=lw, linestyle='--')
#     plt.xlim([-0.05, 1.05])
#     plt.ylim([-0.05, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.legend(loc="lower right", prop={'size': 11})



def get_explain_instances(instances_str):
    if ":" in instances_str:
        split = instances_str.split(":")
        if len(split) == 2:
            return list(range(int(split[0]), int(split[1])+1))
        else:
            raise ValueError("Please set instance to min_idx:max_idx")
    else:
        return [int(instances_str)]



@dataclass
class TreeEnsembleHP:
    n_estimators: int = 100 # Number of trees in the forest
    max_depth: int = -1 # Maximal depth of each tree
    min_samples_leaf: int = 1 # No leafs can have less samples than this
    min_samples_split: int = 2 # Nodes with fewer samples cannot be split
    max_features: str = "auto" # Number of features considered for a split
    random_state: int = 0 # Random seed of the learning algorithm

@dataclass
class RandomForestHP():
    max_samples : float = 0.99 # Number of samples (relative to N) for boostrap
    criterion: str = "gini" # Split criterion, automatically set to MSE in regression


@dataclass
class GradientBoostingHP:
    subsample: float = 1.0 # Ratio of samples used to fit one tree (SGD)
    learning_rate: float = 0.1 # Learning rate of the algorithm
    n_iter_no_change: int = 50 # Early stopping, which will generate a valid set


@dataclass
class Config:
    save: bool = False # Save results of runs e.g. models, explainations ...


@dataclass
class Wandb_Config:
    wandb: bool = False  # Use wandb logging
    wandb_project: str = "XUncertainty"  # Which wandb project to use
    wandb_entity: str = "galabc"  # Which wandb entity to use


@dataclass
class Data_Config:
    name: str = "bike"  # Name of dataset "bike", "california", "boston"
    batch_size: int = 50  # Mini batch size
    scaler: str = "Standard"  # Scaler used for features and target


@dataclass
class Search_Config:
    n_splits: int = 5  # Number of train/valid splits
    split_type: str = "Shuffle" # Type of cross-valid "Shuffle" "K-fold"
    split_seed: int = 1 # Seed for the train/valid splits reproducability


@dataclass
class Explain_Config():
    explainer: str = "EG" # Type fo explainer "EG", "SHAP"
    instance: str = "2" # Instances to explain, can be 0:2 to explain 0, 1, 2
    test_instances: str = ':200'   # All test instances to explain
    MC_seed: int = 42 # Seed for Monte Carlo estimations
    MC_samples: int = 2 # Number of Monte Carlo samples
    attrib_cutoff: float = 0 # Cutoff to define Negligible attributions



def SchulzLeik(phis):
    """
    Compute Mean ranks and Ordinal Consensus as in Schulz et al (https://arxiv.org/abs/2111.09121)

    Oridinal consensus computation is adapted from:
    https://rdrr.io/rforge/agrmt/src/R/Leik.R
    """
    # Number of models
    n = phis.shape[0]
    # Number of features
    m = phis.shape[1]
    order = np.argsort(phis, axis=-1)
    rank  = np.argsort(order, axis=-1)
    print(rank)
    rank_where = [np.where(rank==r, 1, 0) for r in range(m)]
    frequencies = np.transpose(np.sum(np.stack(rank_where), axis=1))
    # Percentages
    P = frequencies / n
    # Cumulative frequency distribution
    R = np.cumsum(P, axis=1)
    SDi = np.where(R<=0.5, R, 1-R)
    maxSDi = 0.5 * (m - 1)
    D = np.sum(SDi, axis=1) / maxSDi
    return np.mean(rank, axis=0), 1 - D