import numpy as np
from joblib import load, dump
import matplotlib.pyplot as plt
from simple_parsing import ArgumentParser

# Local imports
from utils import custom_train_test_split, setup_pyplot_font, setup_data_trees

import sys, os
sys.path.append(os.path.join('../'))
from uxai.trees import all_tree_preds, epsilon_upper_bound, tree_attributions


if __name__ == "__main__":

    setup_pyplot_font(20)

    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random Seed")
    parser.add_argument("--background_size", type=int, default=100, help="Number of background samples")
    args, unknown = parser.parse_known_args()
    print(args)

    X, y, features, task, ohe = setup_data_trees("adult_income")
    X = ohe.transform(X)
    X_train, X_test, y_train, y_test = custom_train_test_split(X, y, task)
    M = 1000

    model = load(os.path.join("models", "Adult-Income", f"RF_M_{M}_seed_{args.seed}.joblib"))

    # The upper bound on preformance
    tree_preds = all_tree_preds(X_test, model, task="classification")
    m, epsilon_upper = epsilon_upper_bound(tree_preds, y_test.reshape((-1, 1)), task="classification")

    # Shap feature attribution
    foreground = X_test[:2000]
    background = X_train[:args.background_size]
    gaps = model.predict_proba(foreground)[:, 1] - model.predict_proba(X_train)[:, 1].mean()
    rashomon_po = tree_attributions(model, foreground, background, epsilon_upper, features, ohe)

    # Record the Gap Error
    gap_errors = rashomon_po.phi_mean.sum(1) - gaps
    tmp_filename = f"GapErrors_M_{M}_seed_{args.seed}_background_{args.background_size}"
    np.save(os.path.join("models", "Adult-Income", tmp_filename), gap_errors)

    # Save the Rashomon Partial Order Object
    tmp_filename = f"TreeSHAP_M_{M}_seed_{args.seed}_background_{args.background_size}.joblib"
    dump(os.path.join("models", "Adult-Income", tmp_filename), rashomon_po)
