""" Train Additive Splines on the Kaggle Housing dataset """

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.preprocessing import SplineTransformer, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score

# Local imports
from utils import setup_pyplot_font, get_complex_features, kaggle_submission
from data_utils import DATASET_MAPPING

import sys, os
sys.path.append(os.path.join('../'))
from uxai.linear import LinearRashomon


if __name__ == "__main__":

    setup_pyplot_font(20)

    # Hyperparameter search space
    n_knots_space = [3, 4, 5]
    knots_space = ['quantile', 'uniform']
    degree_space = [1, 2, 3]

    # Repeat the experiments with and without correlated features
    for remove_correlations in [False, True]:
        print(f"\n\n\n\n######## Remove Correlations ? : {remove_correlations} ############\n\n")

        # Collect the number of parameters
        n_params = []
        # Collect the CV RMSE
        cv_rmse = []
        # Keep the optimal model
        optimal_model = None
        optimal_cv_rmse = 1e10

        ##### Get the data #####
        X, y, features, _ = DATASET_MAPPING["kaggle_houses"](remove_correlations)
        print(X.shape)
        print(y.std())
        kfold = KFold()

        # Fit a Linear Regression as a reference
        cv_baseline_score = -1 * cross_val_score(LinearRashomon(),
                                X, y, scoring="neg_root_mean_squared_error", cv=kfold).mean()
        
        # Iterate over the number of features k on which we fit splines
        for top_k in [1, 2, 3, 4, 5]:

            # Spline preprocessing on these features with high R^2
            complex_feature_idx, simple_feature_idx = get_complex_features(X, y, top_k, features)

            # Initialize the model
            encoder = ColumnTransformer([
                                        ('identity', FunctionTransformer(), simple_feature_idx),
                                        ('spline', SplineTransformer(n_knots=4, degree=3, 
                                                    include_bias=False, knots='quantile'), complex_feature_idx)
                                        ])
            rashomon_set = LinearRashomon()
            model = Pipeline([('encoder', encoder), ('predictor', rashomon_set)])

            # Hyperparameter search space
            search = GridSearchCV(
                model,
                cv=KFold(),
                scoring='neg_root_mean_squared_error',
                param_grid={"encoder__spline__n_knots": n_knots_space,
                            "encoder__spline__knots": knots_space,
                            "encoder__spline__degree": degree_space},
            )
            search.fit(X, y)

            # Recover the optimal CV model
            model = search.best_estimator_
            res = search.cv_results_
            curr_cv_rmse = np.nan_to_num(-res['mean_test_score'], nan=1e10)
            if np.min(curr_cv_rmse) < optimal_cv_rmse:
                optimal_model = model
                optimal_cv_rmse = np.min(curr_cv_rmse)

            # Results of CV
            cv_rmse.append(curr_cv_rmse)
            n_params.append(res['param_encoder__spline__n_knots'].data.astype(np.float64) + \
                            res['param_encoder__spline__degree'].data.astype(np.float64) - 2)
            n_params[-1] = top_k * n_params[-1] + 1 + len(features) - top_k

        # Aggregate results
        cv_rmse = np.concatenate(cv_rmse)
        n_params = np.concatenate(n_params)
        min_params = np.min(n_params)
        max_params = np.max(n_params)

        # Plot the results of hyper-parameter optimization
        best_idx = np.argmin(cv_rmse)
    
        plt.figure()
        plt.scatter(n_params, cv_rmse, c='b', alpha=0.75)
        plt.plot(n_params[best_idx], optimal_cv_rmse, 'r*', markersize=10, markeredgecolor='k')
        plt.plot([min_params-2, max_params+2], cv_baseline_score * np.ones(2), 'k--')
        plt.xlim(min_params-2, max_params+2)
        plt.ylim(0.13, y.std())
        plt.xlabel("Number of free Parameters")
        plt.ylabel("Cross-Validated RMSE")
        plt.savefig(os.path.join("Images", "Kaggle-Houses", f"cv_remove_correls_{remove_correlations}.pdf"), bbox_inches='tight')
        plt.show()


        # Train errors
        train_preds = optimal_model.predict(X)
        train_errors = (train_preds - y) ** 2
        train_RMSE = np.sqrt(np.mean(train_errors))
        # Delta method to compute the CI
        N = X.shape[0]
        train_RMSE_CI = norm.ppf(1-0.025) * np.std(train_errors) / (2 * train_RMSE* np.sqrt(N))
        print("Train RMSE ", train_RMSE, " +/- ", train_RMSE_CI)
        print(f"Number of parameters {n_params[best_idx]}")

        # Submit on Kaggle to get the test performance
        kaggle_submission(model, remove_correlations)

        # Pickle the model
        from joblib import dump
        dump(optimal_model, os.path.join("models", "Kaggle-Houses", f"splines_remove_correls_{remove_correlations}.joblib"))
