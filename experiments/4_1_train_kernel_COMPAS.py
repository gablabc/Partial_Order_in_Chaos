""" Train Kernel Ridge Regression to predict 1-10 COMPAS scores """
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import  cross_val_score, train_test_split
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from simple_parsing import ArgumentParser

from data_utils import DATASET_MAPPING
from utils import setup_pyplot_font

import os, sys
sys.path.append(os.path.join('..'))
from uxai.kernels import KernelRashomon



if __name__ == "__main__":

    setup_pyplot_font(20)

    parser = ArgumentParser()
    parser.add_argument("--kernel", type=str, default="rbf", help="Kernel : rbf or poly")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of processes")
    parser.add_argument("--n_steps", type=int, default=10, help="Steps per hyper-parameter")
    args, unknown = parser.parse_known_args()
    print(args)

    # Load dataset
    X, y, features, _ = DATASET_MAPPING["compas"]()
    # Scale numerical features
    scaler = StandardScaler().fit(X[:, features.non_nominal])
    X[:, features.non_nominal] = scaler.transform(X[:, features.non_nominal])
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                            shuffle=True, random_state=0)

    # Kfold object
    kfold = KFold()

    # Fit a RF as a reference
    cv_rf_score = -1 * cross_val_score(RandomForestRegressor(),
                                        X_train, y_train,
                                        scoring="neg_root_mean_squared_error", cv=kfold)
    print("#### Random Forest ####")
    print(f"RF CV RMSE {cv_rf_score.mean():.2f}")
    print(f"Target Std {y.std():.2f}\n")


    print("#### Kernel Ridge ####")
    # Search for best Kernel Ridge Model
    if args.kernel == "rbf":
        base_estimator = KernelRashomon(kernel="rbf", gamma=0.1, n_jobs=args.n_jobs)
    elif args.kernel == "poly":
        base_estimator= KernelRashomon(kernel="poly", gamma=0.1, degree=3, n_jobs=args.n_jobs)
    else:
        raise Exception("the Kernel must either be rbf or poly")

    lambd_space = np.logspace(-6, -1, args.n_steps)
    gamma_space = np.logspace(-5, 5, args.n_steps)
    search = GridSearchCV(
        base_estimator,
        cv=kfold,
        scoring='neg_root_mean_squared_error',
        param_grid={"lambd": lambd_space,
                    "gamma": gamma_space},
        verbose=2,
    )
    search.fit(X_train, y_train)
    kr = search.best_estimator_
    res = search.cv_results_

    # Train fit the Rashomon Parameter
    kr.fit(X_train, y_train, fit_rashomon=True)

    # Results of CV
    best_idx = np.argmax(res['mean_test_score'])
    lambdas = res['param_lambd'].data.astype(np.float64)
    gammas = res['param_gamma'].data.astype(np.float64)

    # Sizes of dots will encode gamma
    sizes = np.log10(res['param_gamma'].data.astype(np.float64))
    sizes -= sizes.min() - 1
    sizes = np.sort(np.unique(sizes))

    # Plot the results of hyper-parameter optimization
    plt.figure()
    for i, gamma in enumerate(gamma_space):
        idx_select = np.where(gammas == gamma)[0]
        plt.plot(lambdas[idx_select], -res['mean_test_score'][idx_select], 
                    "b-o", alpha=0.5, markersize=2*sizes[i])

    plt.plot(lambdas[best_idx], -search.best_score_, 'r*', markersize=10, markeredgecolor='k')

    plt.plot([7.5e-7, 0.2], cv_rf_score.mean() * np.ones(2), 'k--')
    plt.plot([7.5e-7, 0.2], y_train.std() * np.ones(2), 'k--')
    plt.xlim(7.5e-7, 0.2)
    plt.xlabel(r"Regularization $\lambda$")
    plt.ylabel("Cross-Validated RMSE")
    plt.xscale('log')
    plt.savefig(os.path.join("Images", "COMPAS", f"performance_{args.kernel}.pdf"), bbox_inches='tight')
    plt.show()

    # Test perf
    print(f"Ridge Test RMSE {mean_squared_error(kr.predict(X_test), y_test, squared=False):.2f}")
    print(f"Target Test Std {y_test.std():.2f}")

    # Pickle the model
    from joblib import dump
    dump(kr, os.path.join("models", "COMPAS", f"kernel_{args.kernel}.joblib"))
