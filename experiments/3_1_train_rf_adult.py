from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif', 'sans-serif':['Computer Modern Sans Serif'], 'size':15})
rc('text', usetex=True)
from joblib import dump

# Local
from utils import Data_Config, Search_Config, setup_data_trees
from utils import get_cross_validator, custom_train_test_split

import sys, os
sys.path.append(os.path.join('../'))
from uxai.trees import all_tree_preds, epsilon_upper_bound



if __name__ == "__main__":
    from simple_parsing import ArgumentParser

    ################################### Setup #################################
    
    # Parse arguments
    parser = ArgumentParser()
    parser.add_arguments(Data_Config, "data")
    parser.add_arguments(Search_Config, "search")
    
    
    args, unknown = parser.parse_known_args()
    print(args)

    X, y, features, task, ohe = setup_data_trees("adult_income")
    # Encode for training
    X = ohe.transform(X)
    X_train, X_test, y_train, y_test = custom_train_test_split(X, y, task)
    
    # Hyperparameter grid
    hp_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": list(range(2, 20)),
        "min_samples_leaf": list(range(1, 50, 2)),
        "n_estimators": list(range(50, 500, 50)),
        "max_features": ['auto', 'log2', None],
        "max_samples": [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    }
    
    # Cross validator for train/valid splits
    cross_validator = get_cross_validator(args.search.n_splits, task, 
                                          args.search.split_seed, 
                                          args.search.split_type)
    
    # Init model
    model = RandomForestClassifier(n_jobs=-1)
    
    n_repetitions = 100
    cv_search = RandomizedSearchCV(model, hp_grid, scoring="accuracy",
                                    cv=cross_validator, n_iter=n_repetitions,
                                    verbose=2, random_state=42).fit(X_train, y_train.ravel())
    best_hp = cv_search.best_params_
    print(best_hp)

    for M in [10, 100, 1000]:
        model = RandomForestClassifier(**best_hp)

        ms = []
        epsilons_upper = []
        confidences = []
        for seed in range(5):
            print(f"M : {M} \nseed : {seed}\n")
            model.set_params(random_state=int(seed), n_estimators=M)
            model.fit(X_train, y_train.ravel())

            # Pickle the model
            dump(model, os.path.join("models", "Adult-Income", f"RF_M_{M}_seed_{seed}.joblib"))

            # Get the predictions of all trees on the test set
            tree_preds = all_tree_preds(X_test, model, task="classification")
            # Get the upper bound epsilon^+(m) on the error over H_{m:}
            m, epsilon_upper = epsilon_upper_bound(tree_preds, y_test.reshape((-1, 1)), task="classification")

            true_score = 1 - model.score(X_test, y_test)
            N = X_test.shape[0]
            confidence = 1 - np.exp(-0.5 * N * (epsilon_upper - true_score) ** 2)

            ms.append(m)
            epsilons_upper.append(epsilon_upper)
            confidences.append(confidence)
        
        ms = np.array(ms)
        epsilons_upper = np.array(epsilons_upper)
        confidences = np.array(confidences)

        plt.figure()
        for i in range(len(ms)):
            plt.plot(ms[1], epsilons_upper[i], label=i)
        plt.xlabel("m")
        plt.ylabel(r"$\epsilon$")
        plt.legend()
        plt.savefig(os.path.join("Images", "RFs", f"RF_epsilon+_M_{M}.pdf"), bbox_inches='tight', pad_inches=0)


        plt.figure()
        for i in range(len(ms)):
            plt.plot(ms[1], confidences[i], label=i)
        plt.xlabel("m")
        plt.ylabel("Confidence")
        plt.savefig(os.path.join("Images", "RFs", f"RF_confidence_M_{M}.pdf"), bbox_inches='tight', pad_inches=0)
