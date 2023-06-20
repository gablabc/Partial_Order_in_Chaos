""" 
General utility functions for experiments 

"""

import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.tree import DecisionTreeRegressor

# Local imports
from data_utils import DATASET_MAPPING, TASK_MAPPING




def setup_pyplot_font(size=11):
    from matplotlib import rc
    rc('font',**{'family':'serif', 'serif':['Computer Modern Roman'], 'size':size})
    rc('text', usetex=True)
    from matplotlib import rcParams
    rcParams["text.latex.preamble"] = r"\usepackage{bm}\usepackage{amsfonts}"


############################## Additive Models ################################

def main_effect_score(X, y):
    """ Regress the target on one feature and look at the R^2 """
    d = X.shape[1]
    scores = np.zeros(d)
    model = DecisionTreeRegressor(max_depth=3, random_state=42)
    for i in range(d):
        scores[i] = model.fit(X[:, [i]], y).score(X[:, [i]], y)
    return scores


def get_complex_features(X, y, k, features):
    scores = main_effect_score(X, y)
    complex_feature_idx = []
    for i in np.argsort(scores)[::-1]:
        # Dont consider features with few unique values (usually counts 0-4)
        # Nor features with many zero values since some quantiles would overlap
        if len(np.unique(X[:, i])) > 5 and np.mean(X[:, i]==0) < 0.05:
            print(f"{features.names[i]} : {scores[i]:.3f}")
            if k > 0:
                complex_feature_idx.append(i)
                k -= 1
    print(f"fitting Splines on {[features.names[i] for i in complex_feature_idx]}")
    
    # Spline preprocessing on these features with high R^2
    simple_feature_idx = [i for i in range(X.shape[1]) if i not in complex_feature_idx]
    return complex_feature_idx, simple_feature_idx


def kaggle_submission(model, remove_correlations):
    X, _, _, Id = DATASET_MAPPING["kaggle_houses"](remove_correlations, submission=True)
    preds = np.exp(model.predict(X))
    output = pd.DataFrame({'Id': Id,
                       'SalePrice': preds.squeeze()})
    filename = f"submission_remove_corr_{remove_correlations}.csv"
    output.to_csv(os.path.join("models", "Kaggle-Houses", filename), index=False)


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



def SchulzLeik(phis):
    """
    Compute Mean ranks and Ordinal Consensus as 
    in Schulz et al (https://arxiv.org/abs/2111.09121)

    Oridinal consensus computation is adapted from:
    https://rdrr.io/rforge/agrmt/src/R/Leik.R

    Parameters
    ----------
        phis: (n, d) `np.array`
            feature attributions for the n models

    Returns
    -------
        mean_rank: (d,) `np.array`
        ordinal_consensus: (d,) `np.array`
    """
    # Number of models
    n = phis.shape[0]
    # Number of features
    m = phis.shape[1]
    order = np.argsort(phis, axis=-1)
    rank  = np.argsort(order, axis=-1)
    rank_where = [np.where(rank==r, 1, 0) for r in range(m)]
    frequencies = np.transpose(np.sum(np.stack(rank_where), axis=1))
    # Percentages
    P = frequencies / n
    # Cumulative frequency distribution
    R = np.cumsum(P, axis=1)
    SDi = np.where(R <= 0.5, R, 1-R)
    maxSDi = 0.5 * (m - 1)
    D = np.sum(SDi, axis=1) / maxSDi
    return np.mean(rank, axis=0), 1 - D