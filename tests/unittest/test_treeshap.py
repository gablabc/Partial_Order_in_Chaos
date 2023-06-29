""" Assert that TreeSHAP returns the right values """

import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
from scipy.stats import chi2

import shap
from shap.explainers import Tree
from shap.maskers import Independent

import sys, os
sys.path.append(os.path.join("..", ".."))
from uxai.trees import interventional_treeshap



def compare_shap_implementations(X, model):
    # Run the original treeshap
    background = X[:100]
    masker = Independent(background, max_samples=100)
    explainer = Tree(model, data=masker)
    orig_shap = explainer(X).values
    if orig_shap.ndim == 3:
        orig_shap = orig_shap[..., -1]

    # Run the custom treeshap
    custom_shap, _ = interventional_treeshap(model, X, background)
    custom_shap = custom_shap.mean(-1)

    # Make sure we output the same result
    assert np.isclose(orig_shap, custom_shap).all()



def check_shap_additivity(X, model, I_map, task="regression"):
    # Run the original treeshap
    background = X[:100]

    # Prediction Gaps
    if task == "regression":
        gaps = model.predict(X) - model.predict(background).mean()
    else:
        gaps = model.predict_proba(X)[:, 1] - model.predict_proba(background)[:, 1].mean()
    
    # Run the custom treeshap
    custom_shap, _ = interventional_treeshap(model, X, background, I_map=I_map)
    custom_shap = custom_shap.mean(-1)

    assert custom_shap.shape[1] == I_map[-1]+1, "Not one SHAP value per coallition"

    # Make sure the SHAP values add up to the gaps
    assert np.isclose(gaps, custom_shap.sum(1)).all()



@pytest.mark.parametrize("d", range(1, 21))
def test_regression_uncorrelated(d):
    np.random.seed(42)

    # Generate data
    X = np.random.normal(0, 1, size=(1000, d))
    y = X.mean(1)

    # Fit model
    model = RandomForestRegressor(n_estimators=40, max_depth=10).fit(X, y)

    # Compute SHAP values
    compare_shap_implementations(X, model)



@pytest.mark.parametrize("d", range(1, 21))
def test_classification_uncorrelated(d):
    np.random.seed(42)

    # Generate data
    X = np.random.normal(0, 1, size=(1000, d))
    y = (np.linalg.norm(X, axis=1) > np.sqrt(chi2(df=d).ppf(0.5))).astype(int)

    # Fit model
    model = RandomForestClassifier(n_estimators=40, max_depth=10).fit(X, y)

    # Compute SHAP values
    compare_shap_implementations(X, model)



@pytest.mark.parametrize("d", range(4, 21, 4))
def test_regression_coallition(d):
    np.random.seed(42)

    # Generate data
    X = np.random.normal(0, 1, size=(1000, d))
    y = X.mean(1)

    # Determine coallitions
    n_coallitions = d // 4
    coallition_size = int(d / n_coallitions)
    I_map = np.ravel([[i]*coallition_size for i in range(n_coallitions)])

    # Fit model
    model = RandomForestRegressor(n_estimators=40, max_depth=10).fit(X, y)

    # Compute SHAP values
    check_shap_additivity(X, model, I_map)



@pytest.mark.parametrize("d", range(4, 21, 4))
def test_classification_coallition(d):
    np.random.seed(42)

    # Generate data
    X = np.random.normal(0, 1, size=(1000, d))
    y = (np.linalg.norm(X, axis=1) > np.sqrt(chi2(df=d).ppf(0.5))).astype(int)

    # Determine coallitions
    n_coallitions = d // 4
    coallition_size = int(d / n_coallitions)
    I_map = np.ravel([[i]*coallition_size for i in range(n_coallitions)])

    # Fit model
    model = RandomForestClassifier(n_estimators=40, max_depth=10).fit(X, y)

    # Compute SHAP values
    check_shap_additivity(X, model, I_map, task="classification")



@pytest.mark.parametrize("d", range(1, 21))
def test_regression_correlated(d):
    np.random.seed(42)

    # Generate data
    mu = np.zeros(d)
    sigma = 0.5 * np.eye(d) + 0.5 * np.ones((d, d))
    X = np.random.multivariate_normal(mean=mu, cov=sigma, size=(1000,))
    y = X.mean(1)

    # Fit model
    model = RandomForestRegressor(n_estimators=40, max_depth=10).fit(X, y)

    # Compute SHAP values
    compare_shap_implementations(X, model)



@pytest.mark.parametrize("d", range(1, 21))
def test_classification_correlated(d):
    np.random.seed(42)

    # Generate data
    mu = np.zeros(d)
    sigma = 0.5 * np.eye(d) + 0.5 * np.ones((d, d))
    X = np.random.multivariate_normal(mean=mu, cov=sigma, size=(1000,))
    y = (np.linalg.norm(X, axis=1) > np.sqrt(chi2(df=d).ppf(0.5))).astype(int)

    # Fit model
    model = RandomForestClassifier(n_estimators=40, max_depth=10).fit(X, y)

    # Compute SHAP values
    compare_shap_implementations(X, model)



# Test with adult without one-hot-encoding
def test_adult_no_ohe():
    # Get data without OHE directly from SHAP library
    X, y = shap.datasets.adult()

    # Fit model
    model = RandomForestClassifier(random_state=23, n_estimators=50, 
                                   max_depth=4, min_samples_leaf=50)
    model.fit(X, y)

    # Compute SHAP values
    compare_shap_implementations(X, model)


# Test with adult with ohe


if __name__ == "__main__":
    test_classification_coallition(8)

