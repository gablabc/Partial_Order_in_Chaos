""" Train Random Forest Classifiers on Adult-Income """

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif', 'sans-serif':['Computer Modern Sans Serif'], 'size':15})
rc('text', usetex=True)
from joblib import dump
import os

# Local
from utils import Data_Config, Search_Config, setup_data_trees
from utils import get_cross_validator, custom_train_test_split


if __name__ == "__main__":
    from simple_parsing import ArgumentParser

    ################################### Setup #################################
    
    # Parse arguments
    parser = ArgumentParser()
    parser.add_arguments(Search_Config, "search")
    
    
    args, unknown = parser.parse_known_args()
    print(args)

    X, y, features, task, ohe, _ = setup_data_trees("adult_income")
    # Encode for training
    X = ohe.transform(X)
    X_train, X_test, y_train, y_test = custom_train_test_split(X, y, task)
    
    # Hyperparameter grid
    hp_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": list(range(2, 20)),
        "min_samples_leaf": list(range(1, 50, 2)),
        "n_estimators": list(range(50, 500, 50)),
        "max_features": ['sqrt', 'log2', None],
        "max_samples": [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    }
    
    # Cross validator for train/valid splits
    cross_validator = get_cross_validator(args.search.n_splits, task, 
                                          args.search.split_seed, 
                                          args.search.split_type)
    
    # Init model
    model = RandomForestClassifier(n_jobs=-1, random_state=42)
    
    n_repetitions = 100
    cv_search = RandomizedSearchCV(model, hp_grid, scoring="accuracy",
                                    cv=cross_validator, n_iter=n_repetitions,
                                    verbose=2, random_state=42).fit(X_train, y_train.ravel())
    # Find the best Hyper-Parameters
    best_hp = cv_search.best_params_
    print(best_hp)

    # Train M=1000 trees with five different seeds
    model = RandomForestClassifier(**best_hp)

    # Repeat training 5 times for variability considerations
    for seed in range(5):
        print(f"\nseed : {seed}\n")
        model.set_params(random_state=int(seed), n_estimators=1000)
        model.fit(X_train, y_train.ravel())

        # Save the model
        dump(model, os.path.join("models", "Adult-Income", f"RF_M_1000_seed_{seed}.joblib"))
