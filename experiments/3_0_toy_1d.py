""" Visualize the Rashomon Set of Random Forest Regression """

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif', 'serif':['Computer Modern Roman'], 'size':20})
rc('text', usetex=True)

from sklearn.ensemble import RandomForestRegressor
import sys, os
sys.path.append(os.path.join('../'))
from uxai.trees import RandomForestRashomon


def experiment(forest):
    
    np.random.seed(42)
    # plt.figure()
    N = 600
    # Generate forest
    X = np.random.normal(0, 1, size=(N, 1))
    y = X ** 2 + 0.9 * np.random.normal(size=(N, 1))
    forest.fit(X, y.ravel())

    # Compute Rashomon Set on test set
    rashomon = RandomForestRashomon(forest)
    rashomon.fit(X, y, M_min=500)

    # Fix an error tolerance
    epsilon = 1
    # Obtain the least number of trees that guarantees being 
    # under the error threshold
    m_epsilon = rashomon.get_m_epsilon(epsilon)
    print(f"m(epsilon) = {m_epsilon}")

    # Plot the epsilon upper bound used to set the minimum number of trees
    plt.figure()
    plt.plot(rashomon.m, rashomon.epsilon_upper, 'r', label=r"$\epsilon^+(m)$")
    plt.plot([500, 1000], [epsilon, epsilon], "k--")
    plt.text(510, epsilon+0.003, r"$\epsilon$" )
    plt.plot([m_epsilon, m_epsilon], [rashomon.epsilon_upper[-1], 1], "k--")
    plt.text(m_epsilon+10, rashomon.epsilon_upper[-1]+0.003, r"$m(\epsilon)$" )
    plt.xlabel(r"$m$")
    plt.ylabel("RMSE")
    plt.xlim(500, 1000)
    plt.ylim(rashomon.epsilon_upper[-1], rashomon.epsilon_upper[0])
    plt.legend()
    plt.savefig(os.path.join("Images", "RFs", "RF_1d_example.pdf"), bbox_inches='tight', pad_inches=0)


    # Plot the min-max predictions accross the Rashomon Set
    x_lin = np.linspace(-4, 4, 200).reshape((-1, 1))
    plt.figure()
    plt.scatter(X, y)
    x_lin = np.linspace(-4, 4, 100).reshape((-1, 1))
    _, min_max_preds = rashomon.predict(x_lin, epsilon)
    plt.plot(x_lin, min_max_preds[:, 0], 'r--')
    plt.plot(x_lin, min_max_preds[:, 1], 'r--')
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.savefig(os.path.join("Images", "RFs", "RF_1d_example_preds.pdf"), bbox_inches='tight', pad_inches=0)




experiment(forest=RandomForestRegressor(n_estimators=1000, max_depth=4, 
                                        max_samples=0.9, random_state=42))


plt.show()
