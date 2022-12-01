import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif', 'serif':['Computer Modern Roman'], 'size':20})
rc('text', usetex=True)

from sklearn.ensemble import RandomForestRegressor

import sys, os
sys.path.append(os.path.join('../'))
from uxai.trees import all_tree_preds, epsilon_upper_bound

np.random.seed(42)
x = np.random.normal(0, 1, size=(600, 1))
y = x ** 2 + 0.9 * np.random.normal(size = (600, 1))

x_lin = np.linspace(-4, 4, 200).reshape((-1, 1))

def experiment(forest):
    
    forest.fit(x, y.ravel())
    
    tree_preds = all_tree_preds(x, forest)
    m, epsilon_upper = epsilon_upper_bound(tree_preds, y, M_min=500)
    epsilon = 1
    m_epsilon_idx = np.argmax(epsilon_upper < epsilon)
    m_epsilon = m[m_epsilon_idx]
    print(f"m(epsilon) = {m_epsilon}")
    plt.figure()
    plt.plot(m, epsilon_upper, "r-", label=r"$\epsilon^+(m)$")
    plt.plot([m.min(), m.max()], [epsilon, epsilon], "k--")
    plt.text(10+m[0], epsilon+0.003, r"$\epsilon$" )
    plt.plot([m_epsilon, m_epsilon], [epsilon_upper[-1], epsilon_upper[m_epsilon_idx]], "k--")
    plt.text(m_epsilon+10, epsilon_upper[-1]+0.003, r"$m(\epsilon)$" )
    plt.xlabel(r"$m$")
    plt.ylabel("RMSE")
    plt.xlim(m.min(), m.max())
    plt.ylim(epsilon_upper[-1], epsilon_upper[0])
    plt.legend()
    plt.savefig(os.path.join("Images", "RFs", "RF_1d_example.pdf"), bbox_inches='tight', pad_inches=0)


    plt.figure()
    plt.scatter(x, y)
    x_lin = np.linspace(-4, 4, 100).reshape((-1, 1))
    tree_preds = all_tree_preds(x_lin, forest)
    # Cherry pick 30 trees with lowest/largest pred
    cherry_picked_min = np.partition(tree_preds, kth=m_epsilon)[:, :m_epsilon]
    cherry_picked_max = -np.partition(-tree_preds, kth=m_epsilon)[:, :m_epsilon]
    print(cherry_picked_min.shape)
    plt.plot(x_lin, cherry_picked_min.mean(1), 'r--')
    plt.plot(x_lin, cherry_picked_max.mean(1), 'r--')
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.savefig(os.path.join("Images", "RFs", "RF_1d_example_preds.pdf"), bbox_inches='tight', pad_inches=0)



experiment(forest=RandomForestRegressor(n_estimators=1000, max_depth=4, 
                                        max_samples=0.9, random_state=42))

#experiment(forest=ExtraTreesRegressor(n_estimators=2000, max_depth=4, 
#                                        max_samples=0.9, random_state=42))

plt.show()
