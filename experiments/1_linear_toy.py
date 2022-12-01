import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from simple_parsing import ArgumentParser

from utils import setup_pyplot_font

import sys, os
sys.path.append(os.path.join('../..'))

from uxai.linear import LinearRashomon

coeffs = np.array([2, -2.5, 3, 0.5, -0.2, 0.1]).reshape((-1, 1))

# Generate Data
def generate_data(N, sigma, rho, seed):
    np.random.seed(seed)

    cov = np.eye(6)
    cov[1, 0] = rho
    cov[0, 1] = rho
    cov[1, 2] = rho
    cov[2, 1] = rho
    cov[2, 0] = rho
    cov[0, 2] = rho

    X = np.random.multivariate_normal(mean=np.zeros(6), cov=cov, size=(N,))
    y = X.dot(coeffs) + sigma * np.random.normal(size=(N, 1))
    return X, y


def experiment(N, sigma, rho, seed, explain_max_y=False):
    # Generate the data and fit model
    X, y = generate_data(N, sigma, rho, seed)
    linear_rashomon = LinearRashomon().fit(X, y)
    RMSE = linear_rashomon.RMSE[0]

    # Compute feature attributions
    rashomon_po = linear_rashomon.attributions(X)

    # Plot confidence vs utility as epsilon is increased
    tolerance = RMSE + np.linspace(1e-5, 1, 200)
    utility = rashomon_po.get_utility(linear_rashomon.get_epsilon(tolerance))
    confidence = chi2(df=N).cdf(N*tolerance**2/sigma**2)

    # Explain the maximal value of y
    if explain_max_y:
        idx = np.argmax(y)
        x_study = X[idx]
        feature_names = [f"x{j}={x_study[j]:.2f}" for j in range(6)]

        # Ground truth attribution
        ground_truth = (x_study - X.mean(0)) * coeffs.ravel()
        print(ground_truth)

        utility_star = []
        confidence_star = []
        POs = []
        for max_confidence in [0.975, 0.999]:
            # Compute the best trade-off for epsilon
            idx_tolerance_star = np.argmax(confidence>max_confidence)
            tolerance_star = tolerance[idx_tolerance_star]
            utility_star.append(utility[idx_tolerance_star])
            confidence_star.append(confidence[idx_tolerance_star])

            # Return the partial order
            PO = rashomon_po.get_poset(idx, linear_rashomon.get_epsilon(tolerance_star), feature_names)
            POs.append(PO)
        return utility, confidence, utility_star, confidence_star, POs
    else:
        return utility, confidence


if __name__ == "__main__":
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument("--vary", type=str, default="rho", help="What to vary: sigma, N, rho")
    parser.add_argument("--seed", type=int, default=42, help="Random Seed")
    args, unknown = parser.parse_known_args()
    print(args)

    vary_name = {"sigma" : r"$\sigma$", "N" : r"$N$", "rho" : r"$\rho$"}

    # Experiment : increase sigma
    if args.vary == "sigma":
        sigmas = [0.01, 0.1, 0.25, 0.5, 1]
        varying_values = sigmas
        N = 1000
        rho = 0
        all_confidence = []
        all_utilities = []
        for sigma in sigmas:
            utility, confidence = experiment(N, sigma, rho, args.seed)
            all_confidence.append(confidence)
            all_utilities.append(utility)
            
        all_confidence = np.array(all_confidence)
        all_utilities = np.array(all_utilities)

    # Experiment : increase N
    elif args.vary == "N":
        sigma = 0.1
        Ns = [10, 25, 50, 100, 1000]
        varying_values = Ns
        rho = 0
        all_confidence = []
        all_utilities = []
        for N in Ns:
            utility, confidence = experiment(N, sigma, rho, args.seed)
            all_confidence.append(confidence)
            all_utilities.append(utility)
            
        all_confidence = np.array(all_confidence)
        all_utilities = np.array(all_utilities)

    # Experiment : increase rho
    elif args.vary == "rho":
        sigma = 0.2
        N = 500
        rhos = [0, 0.9, 0.99, 0.999, 0.9999]
        varying_values = rhos
        all_confidence = []
        all_utilities = []
        for rho in rhos:
            if rho == 0.999:
                utility, confidence, utility_star, confidence_star, POs = \
                            experiment(N, sigma, rho, args.seed, explain_max_y=True)
                for i, PO in enumerate(POs):
                    dot = PO.print_hasse_diagram(show_ambiguous=False)
                    dot.render(filename=os.path.join('Images', 'Toy', f"PO_Linear_{i}"), format='pdf')
            else:
                utility, confidence = experiment(N, sigma, rho, args.seed)
            all_confidence.append(confidence)
            all_utilities.append(utility)
            
        all_confidence = np.array(all_confidence)
        all_utilities = np.array(all_utilities)

    setup_pyplot_font(20)

    fig, ax = plt.subplots()
    ax.set_prop_cycle(color=['green', 'blue','purple', 'red','orange'])
    for i in range(all_confidence.shape[0]):
        ax.plot(all_utilities[i], all_confidence[i], label=vary_name[args.vary]+f"={varying_values[i]}")

    # Show an example of optimal tradeoff
    if args.vary == "rho":
        plt.plot(utility_star[0], confidence_star[0], 'r*', markersize=20)
        plt.text(utility_star[0], confidence_star[0]-0.025, '(d)', horizontalalignment='center')
        plt.plot(utility_star[1], confidence_star[1], 'r*', markersize=20)
        plt.text(utility_star[1], confidence_star[1]-0.025, '(e)', horizontalalignment='center')

    ax.set_xlabel('Utility')
    ax.set_ylabel('Confidence')
    ax.set_xlim(0.2, 1)
    ax.legend(loc="lower left")
    plt.savefig(os.path.join("Images", "Toy", f"Linear_vary_{args.vary}_seed_{args.seed}.pdf"),
                                                                         bbox_inches='tight')
