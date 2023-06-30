""" Explain Random Forests with Interventional-Partition-TreeSHAP """

import numpy as np
from joblib import load
from simple_parsing import ArgumentParser

# Local imports
from utils import custom_train_test_split, setup_pyplot_font, setup_data_trees

import sys, os
sys.path.append(os.path.join('../'))
from uxai.trees import interventional_treeshap


if __name__ == "__main__":

    setup_pyplot_font(20)

    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random Seed")
    parser.add_argument("--background_size", type=int, default=50, help="Number of background samples")
    args, unknown = parser.parse_known_args()
    print(args)

    X, y, features, task, ohe, I_map = setup_data_trees("adult_income")
    X = ohe.transform(X)
    X_train, X_test, y_train, y_test = custom_train_test_split(X, y, task)

    # Get the RF for the given train seed
    model = load(os.path.join("models", "Adult-Income", f"RF_M_1000_seed_{args.seed}.joblib"))

    # SHAP feature attribution
    foreground = X_test[:2000]
    background = X_train[:args.background_size]
    gaps = model.predict_proba(foreground)[:, 1] - model.predict_proba(X_train)[:, 1].mean()
    phis, _ = interventional_treeshap(model, foreground, background, I_map)

    # Save the LFA if using the whole background
    if args.background_size == 500:
        tmp_filename = f"TreeSHAP_M_1000_seed_{args.seed}_background_{args.background_size}"
        np.save(os.path.join("models", "Adult-Income", tmp_filename), phis)
    
    # Record the Gap Error
    gap_errors = phis.mean(-1).sum(1) - gaps
    tmp_filename = f"GapErrors_M_1000_seed_{args.seed}_background_{args.background_size}"
    np.save(os.path.join("models", "Adult-Income", tmp_filename), gap_errors)
