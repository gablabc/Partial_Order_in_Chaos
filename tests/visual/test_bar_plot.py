""" Verify the bar and pcp plot function on local and global """

import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.abspath(os.path.join("..")))
from uxai.partial_orders import CLT_CIs_samples
from uxai.plots import bar, pcp

features_names = ['x1', 'x2', 'x3', 'x4']

# %%
# FI One Model   
# all FIs samples of shape (MC_samples, n_features)
# FIs of shape (n_features,)
# CIs of shape (n_features, 2)
FIs = np.array([[1, 2, 3, 4]]) + 0.5 * np.random.randn(50, 4)
FI_mean = FIs.mean(0)

# No CIs
bar(FI_mean, features_names)
plt.title("FI Bar plot no CIs one Model")
pcp(FI_mean, features_names)
plt.title("FI PCP plot no CIs one Model")

# With symmetric CIs
CIs = CLT_CIs_samples(FIs)
bar(CIs.phis, features_names, xerr=CIs.widths(0.01).T)
plt.title("FI Bar plot symmetric CIs one Model")
pcp(CIs.phis, features_names, xerr=CIs.widths(0.01))
plt.title("FI PCP plot symmetric CIs one Model")

# With asymmetric CIs
CIs_asym = np.column_stack((0.1 * np.ones(4), 0.4 * np.ones(4)))
bar(CIs.phis, features_names, xerr=CIs_asym.T)
plt.title("FI Bar plot asymmetric CIs one Model")
pcp(CIs.phis, features_names, xerr=CIs_asym)
plt.title("FI PCP plot asymmetric CIs one Model")

# %%
# Attrib One Model
phis = np.array([[1, -1, 2, 3]]) + 0.5 * np.random.randn(50, 4)
phi_mean = phis.mean(0)
attribs = phi_mean.sum(keepdims=True)

# No CIs
bar(phi_mean, features_names)
plt.title("Attrib Bar plot no CIs one Model")
pcp(phi_mean, features_names, total_attrib=attribs)
plt.title("Attrib PCP plot no CIs one Model")

# With symmetric CIs
CIs = CLT_CIs_samples(phis)
bar(CIs.phis, features_names, xerr=CIs.widths(0.01).T)
plt.title("Attrib Bar plot symmetric CIs one Model")
pcp(CIs.phis, features_names, xerr=CIs.widths(0.01), total_attrib=attribs)
plt.title("Attrib PCP plot symmetric CIs one Model")

# With asymmetric CIs
CIs_asym = np.column_stack((0.1 * np.ones(4), 0.4 * np.ones(4)))
bar(CIs.phis, features_names, xerr=CIs_asym.T)
plt.title("Attrib Bar plot asymmetric CIs one Model")
pcp(CIs.phis, features_names, xerr=CIs_asym, total_attrib=attribs)
plt.title("Attrib PCP plot asymmetric CIs one Model")

# %%
# FI Two Models
# all FIs samples of shape (MC_samples, 2, n_features)
# FIs of shape (2, n_features)
# CIs of shape (2, n_features, 2)
FIs = np.array([[[1, 2, 3, 4], [1.2, 2.2, 4, 3.5]]]) + \
                0.5*np.random.randn(100, 2, 4)
FI_mean = FIs.mean(0)

# No CIs
pcp(FI_mean, features_names)
plt.title("FI PCP plot no CIs two Models")

# With symmetric CIs
CIs = CLT_CIs_samples(FIs)
pcp(CIs.phis, features_names, xerr=CIs.widths(0.01))
plt.title("FI PCP plot symmetric CIs two Models")

# %%
# Attrib Two Models
phis = np.array([[[1, -1, 2, 4], [0.9, -1.1, 4, 3.4]]]) + \
                  0.5 * np.random.randn(50, 2, 4)
phi_mean = phis.mean(0)
attribs = phi_mean.sum(1, keepdims=True)

# No CIs
pcp(phi_mean, features_names, total_attrib=attribs)
plt.title("Attrib PCP plot no CIs two Models")

# With symmetric CIs
CIs = CLT_CIs_samples(phis)
pcp(CIs.phis, features_names, xerr=CIs.widths(0.01), total_attrib=attribs)
plt.title("Attrib PCP plot symmetric CIs two Models")


# # No CIs feature importance (absolute value)
# bar(phis.mean(0), features.map_values(x_test[i]), absolute=True)

# # With symmetric CIs feature importance (absolute value)
# bar(phis.mean(0), features.map_values(x_test[i]), 0.005, xerr=CIs, absolute=True)
