""" Train Degree 1, 2, and 3 Splines on the Kaggle Housing dataset """

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.preprocessing import SplineTransformer, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Local imports
from utils import custom_train_test_split, setup_pyplot_font
from data_utils import DATASET_MAPPING

import sys, os
sys.path.append(os.path.join('../'))
from uxai.linear import LinearRashomon


if __name__ == "__main__":

    setup_pyplot_font(20)

    ##### Fit Linear Model #####
    X, y, features = DATASET_MAPPING["kaggle_houses"]()
    x_train, x_test, y_train, y_test = custom_train_test_split(X, y, 'regression')

    # Spline preprocessing on these features with high R^2
    complex_feature_idx = [1, 2, 5, 10]
    simple_feature_idx = [i for i in range(X.shape[1]) if i not in complex_feature_idx]
    n_knots = 4
    degrees = [1, 2, 3]
    train_errors = []
    train_RMSE = []
    train_RMSE_CI = []
    test_errors = []
    test_RMSE = []
    test_RMSE_CI = []
    for degree in degrees:
        # Fit the model
        encoder = ColumnTransformer([
                                    ('identity', FunctionTransformer(), simple_feature_idx),
                                    ('spline', SplineTransformer(n_knots=n_knots, degree=degree, 
                                                include_bias=False, knots='quantile'), complex_feature_idx)
                                    ])
        rashomon_set = LinearRashomon()
        model = Pipeline([('encoder', encoder), ('predictor', rashomon_set)])
        model.fit(x_train, y_train)


        # Train errors
        train_preds = model.predict(x_train)
        train_errors.append( (train_preds - y_train) ** 2 )
        train_RMSE.append( np.sqrt(np.mean(train_errors[-1])) )
        # Delta method to compute the CI
        N = x_train.shape[0]
        train_RMSE_CI.append( norm.ppf(1-0.025) * np.std(train_errors[-1]) / (2 * train_RMSE[-1] * np.sqrt(N)) )


        # Test errors
        test_preds = model.predict(x_test)
        test_errors.append( (test_preds - y_test) ** 2 )
        test_RMSE.append( np.sqrt(np.mean(test_errors[-1])) )
        # Delta method to compute the CI
        N = x_test.shape[0]
        test_RMSE_CI.append( norm.ppf(1-0.025) * np.std(test_errors[-1]) / (2 * test_RMSE[-1] * np.sqrt(N)) )
        
        # Pickle the model
        from joblib import dump
        dump(model, os.path.join("models", "Kaggle-Houses", f"splines_degree_{degree}.joblib"))


    print(train_RMSE)
    print(test_RMSE)
    
    # Examine the performance
    for i, degree in enumerate(degrees):

        # Bar chart of RMSE
        ind = np.arange(3)  # the x locations for the groups
        width = 0.3  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(ind - width/2, train_RMSE, width, yerr=train_RMSE_CI,
                        label='Train', capsize=10)
        rects2 = ax.bar(ind + width/2, test_RMSE, width, yerr=test_RMSE_CI, 
                        label='Test', capsize=10)

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('RMSE')
        ax.set_xlabel('Polynomial Degree')
        ax.set_xticks(ind)
        ax.set_xticklabels(degrees)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.legend(framealpha=1)
        plt.savefig(os.path.join("Images", "Kaggle-Houses", f"performance.pdf"), bbox_inches='tight')
