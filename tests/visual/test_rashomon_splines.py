"""
Study the Rashomon Set of Spline Regression on a
toy 2D problem. This is used for validating the
`uxai.linear.LinearRashomon` class
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import SplineTransformer

import sys, os
sys.path.append(os.path.join('../..'))

from uxai.linear import LinearRashomon

np.random.seed(42)
X = np.random.uniform(-1, 1, size=(2000, 2))
splt = SplineTransformer(n_knots=4, degree=3, include_bias=False).fit(X)
coallitions = [[0,1,2,3,4], [5,6,7,8,9]]
H = splt.transform(X)
#print(H.shape)

y = H.dot(np.random.uniform(-2, 2, size=(H.shape[1], 1))) + \
                0.1 * np.random.normal(0, 1, size=(2000, 1))
#print(y.shape)


linear_rashomon = LinearRashomon().fit(H, y)
RMSE = linear_rashomon.RMSE
print(f"RMSE : {RMSE}")
print(f"Maxmimum tolerable RMSE : {RMSE + 0.05}")
epsilon = linear_rashomon.get_epsilon(RMSE + 0.05)


# Sample points on the boundary of the Rashomon set
w_boundary = linear_rashomon.ellipsoid.sample_boundary(20000, epsilon)
print(w_boundary.shape)

H_ = (H - linear_rashomon.X_mean) / linear_rashomon.X_std
H_tilde = np.column_stack( (np.ones(len(H_)), H_) )

# Ensure that all models have RMSE bellow epsilon
all_RMSE = np.sqrt(np.mean((y - linear_rashomon.y_std * H_tilde.dot(w_boundary.T) - \
                            linear_rashomon.y_mean) ** 2, axis=0))
print(np.min(all_RMSE), np.max(all_RMSE))


# Compute Predictions for Partial Dependence
line = np.linspace(-1, 1, 100)
xx, yy = np.meshgrid(line, line)
hh = splt.transform(np.column_stack( (xx.ravel(), yy.ravel()) ))
print(hh.shape)
preds, minmax_preds = linear_rashomon.predict(hh, epsilon)
preds = preds.reshape(xx.shape)
uncertainty = np.diff(minmax_preds, axis=1).reshape(xx.shape)

# Plot function
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), s=10)
plt.contourf(xx, yy, preds, alpha=0.5)

# Plot uncertainty
plt.figure()
varmap = plt.contourf(xx, yy, uncertainty, cmap='Blues', alpha=0.5)
plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), s=5)
plt.colorbar(varmap, fraction=0.046, pad=0.04)


# Plot the basis and PDPs
lines = np.column_stack((line, line))
H_lines = splt.transform(lines)
for d in [0, 1]:
    plt.figure()
    plt.plot(line, H_lines[:, coallitions[d]])
    knots = splt.bsplines_[d].t
    plt.vlines(knots[3:-3], ymin=0, ymax=0.8, linestyles="dashed", color="k")

    plt.figure()
    idxs = coallitions[d]
    pdp = linear_rashomon.partial_dependence(H_lines[:, idxs], idx=idxs, epsilon=epsilon)
    plt.fill_between(line, pdp[:, 0], pdp[:, 1], alpha=0.5)

    i = np.array(idxs)+1
    sorted_idx = np.argsort(X[:, d])
    coord_preds = linear_rashomon.y_std * H_tilde[:, i].dot(w_boundary[:20, i].T) + linear_rashomon.y_mean
    plt.plot(X[sorted_idx, d], coord_preds[sorted_idx], 'r')



## Global Feature Importance ##
plt.figure()
importances = np.zeros((w_boundary.shape[0], 2))
for d in [0, 1]:
    idxs = coallitions[d]
    i = np.array(idxs)+1
    coord_attrib = linear_rashomon.y_std * H_tilde[:, i].dot(w_boundary[:, i].T) 
    importances[:, d] = np.std(coord_attrib, axis=0)
# Plot the FI and compare with sampling
bp = plt.boxplot(importances, patch_artist=True)

for patch in bp['boxes']:
    patch.set_facecolor('#9999FF')

extreme_imp, global_po = linear_rashomon.feature_importance(epsilon, ["x1", "x2"], idxs=coallitions)
plt.plot(range(1, 3), extreme_imp[:, 0], 'r')
plt.plot(range(1, 3), extreme_imp[:, 1], 'r')


dot = global_po.print_hasse_diagram()
dot.render(filename=os.path.join('Images', 'PO_Global_Splines'), format='png')



#### Local Feature Attributions ####

# Plot the LFA and compare with sampling
x = np.array([[-0.75, 0]])
h = splt.transform(x)
f_x = float(linear_rashomon.predict(splt.transform(x)))
E_f = float(linear_rashomon.predict(H).mean(0))
print(f"Prediction Gap is : {f_x - E_f:2f}")

print(w_boundary[:, idxs].shape)
per_coord_attrib = w_boundary[:, 1:] * y.std() * (h - np.mean(H, 0)) / H.std(0)
print(per_coord_attrib.shape)
attribs = []
for d in range(2):
    attribs.append(per_coord_attrib[:, coallitions[d]].sum(1))

plt.figure()
bp = plt.boxplot(attribs, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('#9999FF')
rashomon_po = linear_rashomon.feature_attributions(h[0], idxs=coallitions)
extreme_attribs = rashomon_po.minmax_attrib(epsilon)[0]
plt.plot(range(1, 3), extreme_attribs[:, 0], 'r')
plt.plot(range(1, 3), extreme_attribs[:, 1], 'r')

# Partial Order of Local Feature Attribution
local_po = rashomon_po.get_poset(0, epsilon, ["x1", "x2"])
dot = local_po.print_hasse_diagram()
dot.render(filename=os.path.join('Images', 'PO_Lobal_Spline'), format='png')

plt.show()


