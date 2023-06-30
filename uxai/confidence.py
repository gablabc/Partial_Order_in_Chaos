import numpy as np
from scipy.stats import norm
from graphviz import Digraph
from functools import partial



class ConfidenceInterval(object):
    """ Class representing a CI for any delta in ]0, 1[ """
    
    def __init__(self, phis, widths, phis_sign=None):
        self.phis = phis
        self.widths = widths
        self.phis_sign = phis_sign
    
        
    def min(self, delta):
        return self.phis - self.widths(delta)[..., 0]
    
    
    def max(self, delta):
        return self.phis + self.widths(delta)[..., 1]
    
    
    def abs(self):
        def get_width_abs(delta, phis, widths):
            w = widths(delta)
            cross_origin = ( (phis > 0) & ( phis < w[..., 0]) ) | \
                           ( (phis < 0) & (-phis < w[..., 1]) ) 
            # Min and Max of CIs
            map_bottom = np.abs(phis - w[..., 0])
            map_top = np.abs(phis + w[..., 1])
            
            min_CIs = np.minimum(map_bottom, map_top)
            # Crossing the origin -> min CI = 0
            min_CIs[cross_origin] = 0
            max_CIs = np.maximum(map_bottom, map_top)
            FI = np.abs(phis)
            return np.stack((FI - min_CIs, max_CIs - FI), axis=-1)
        
        return ConfidenceInterval(np.abs(self.phis),
                                  partial(get_width_abs, 
                                          phis=self.phis,
                                          widths=self.widths),
                                  np.sign(self.phis))
    
    

### General Functions ###
def CLT_CIs_std(means, std):
    """ 
    Takes in the sample mean and the scaled standard deviation s / sqrt(N)
    and returns a Confidence Interval function of delta.
    """
    
    widths = lambda delta : -1*norm.ppf(delta/2) * np.stack((std, std), axis=-1)
    return ConfidenceInterval(means, widths)



def CLT_CIs_samples(all_samples):
    """ 
    Takes in all MC samples as a (n_samples, ..., n_features) array
    and returns returns a Confidence Interval function of delta with outputs of
    shape (..., n_features, 2)
    """
    n_samples = all_samples.shape[0]
    std = all_samples.std(axis=0) / np.sqrt(n_samples)
    return CLT_CIs_std(all_samples.mean(0), std)



def hoeffding_CI(delta, n_samples, nb_features, bound=1, u_order=1, n_comparisons=None):
    # Adjust delta for the union bound
    if n_comparisons:
        delta /= (2 * n_comparisons)
    else:
        delta /= (nb_features * (nb_features-1))
    # Effective number of samples
    n_samples = np.floor(n_samples / u_order)
    CIs = bound * np.ones(nb_features)
    return CIs * np.sqrt(np.log(2/delta) / (2*n_samples)) 



def abs_map_CIs(phi, CIs):
    """
    Map CIs on phi to CIs on |phi|
    """
    FI = np.abs(phi)
    cross_origin = FI < CIs
    # Min and Max of CIs
    map_bottom = np.abs(phi - CIs)
    map_top = np.abs(phi + CIs)
    min_CIs = np.minimum(map_bottom, map_top)
    max_CIs = np.maximum(map_bottom, map_top)
    # Minimum of CIs that cross origin is zero
    min_CIs[cross_origin] = 0
    return min_CIs, max_CIs
