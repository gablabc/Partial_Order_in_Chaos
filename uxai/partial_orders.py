""" Compute partial orders of Feature Importance """


import numpy as np
from graphviz import Digraph



def transitive_reduction(adjaency):
    """ 
    Reduce the number of 1's in th adjecency by considering the
    transitive properties of order relations
    """
    nb_features = adjaency.shape[0]
    # Transitive reduction
    for j in range(nb_features):
        for i in range(nb_features):
            if adjaency[i, j] == 1:
                for k in range(nb_features):
                    # There is i<=j and j<=k
                    if adjaency[j, k] == 1:
                        # Reduction
                        adjaency[i, k] = 0



def force_ground(adjacency, ground):
    """ Force any feature in the ground to be smaller than others"""
    list_ground = np.array(list(ground)).reshape((-1, 1))
    list_non_ground = [i for i in range(len(adjacency)) if i not in ground]
    # Make ground smaller than non_ground
    adjacency[list_ground, list_non_ground] = 1
    # Make the ground not larger than anything else
    adjacency[list_non_ground, list_ground] = 0
    # Make ground imcomparable
    adjacency[list_ground, list_ground.T] = 0



def compute_ranks(adjaency, top_bottom=True):
    """ Given a adjaency matrix compute the ranks """
    
    # Compute Ranks
    rank = 1
    nb_features = adjaency.shape[0]
    ranks = [0] * nb_features
    indexes = range(nb_features)
    if top_bottom:
        order_copy = adjaency.copy()
    else:
        order_copy = adjaency.T.copy()
    # while not 1, because at the end you end up with a [[-1]] array
    while order_copy.size:
        index_to_remove = []
        # Identify features with no more ancestors
        for i, row in enumerate(order_copy):
            if not row.any():
                index_to_remove.append(i)

        # Remove these features and store their rank
        for ind_to_remove in reversed(index_to_remove):
            ranks[ indexes[ind_to_remove] ] = rank
            # Delete row and column
            order_copy = np.delete(order_copy, ind_to_remove, 0)
            order_copy = np.delete(order_copy, ind_to_remove, 1)
            indexes = np.delete(indexes, ind_to_remove, 0)

        rank += 1
        
    ranks = np.array(ranks)
    if not top_bottom:
        # flip the ranks
        ranks = np.max(ranks) + 1 - ranks
    return ranks



class PartialOrder(object):
    """ Object representing a Partial Order """
    
    def __init__(self, phi_mean, adjacency, ambiguous, features_names, top_bottom=False):
        """
        Create a partial order

        Parameters
        ----------
        phi_mean : (d,) `np.array`
            The feature attributions of the mean model
        adjacency : (d, d) `np.array`
            The True/False adjacency matrix
        ambiguous : `set`
            The set of features whose attribution has an ambiguous sign
        feature_names : `List(str)`
            List of the name of each feature
        top_bottom : `bool`, optional
            Whether or not to compute the partial from the top to the bottom

        """
        self.phi_mean = phi_mean
        self.ambiguous = ambiguous
        self.adjacency = adjacency
        
        # Grounded and reduced adjacencies (useful for plots)
        if len(self.ambiguous) == 0:
            self.grounded_adjacency = self.adjacency
        else:
            self.grounded_adjacency = adjacency.copy()
            # Force ambiguous attributions to be less important than the rest
            force_ground(self.grounded_adjacency, self.ambiguous)
        self.grounded_reduced_adjacency = self.grounded_adjacency.copy()
        transitive_reduction(self.grounded_reduced_adjacency)
        
        self.features_names = features_names
        self.n_features = len(features_names)
        
        # Compute ranks
        self.ranks = compute_ranks(self.grounded_adjacency, 
                                   top_bottom=top_bottom)
    
    
    
    def intersect(self, other_po):
        """ Intersect the PO with another one """
        # A = {i in [d]\N' : phi_i1 * phi_i1 < 0}
        ambiguous = set([f for f in range(self.n_features) 
                         if self.phi_mean[f] * other_po.phi_mean[f] < 0])
        return PartialOrder((self.phi_mean + other_po.phi_mean)/2,
                            self.adjacency * other_po.adjacency, 
                            ambiguous, self.features_names)
    
    
    
    def cardinality(self):
        # The cardinality does not take into account the ground
        return np.sum(self.adjacency)
    
    
    
    def print_hasse_diagram(self, top_ranks=None, show_ambiguous=True):
        """
        Create a partial order

        Parameters
        ----------
        top_ranks : `int`, optional
            Only show the top-k features. By default we show all ranks.
        show_ambiguous : `bool`, optional
            Whether or not to show the ambiguous features

        Returns
        -------
        dot : Digraph
            A directed graph object from graphviz
        """

        # Directed Graph of partial ordering
        dot = Digraph(comment='Feature Importance', graph_attr={'ranksep': "0.6"},
                      node_attr={'shape': 'rectangle', 'color': 'black',
                                 'style': 'filled'})
        color_range = color_nodes(self.phi_mean)
    
        # Dont consider grounded features
        keep_features_idx = [f for f in range(self.n_features) if f not in self.ambiguous]
    
        # Dont consider features that have a higher rank than top_ranks
        if top_ranks is not None:
            # We dont show the ground when restricting top_ranks
            show_ambiguous = False
            keep_features_idx = [keep_features_idx[i] for i in range(len(keep_features_idx))
                                if self.ranks[keep_features_idx[i]] <= top_ranks]
            # Store ranks in sets
            n_ranks = min(max(self.ranks), top_ranks)
        else:
            n_ranks = max(self.ranks)
            
        ranks_set = []
        for _ in range(n_ranks):
            ranks_set.append(set())
    
        # Add each feature to the right set
        for i in keep_features_idx:
            ranks_set[ self.ranks[i]-1 ].add(i)
    
        textcolor = ["black", "white"]
        
        # Print
        for elements in ranks_set:
            with dot.subgraph() as s:
                s.attr(rank='same')
                # Loop over all features of the same rank
                for i in elements:
                    s.node(f"x{i}",
                            f"{self.features_names[i]}\n"+\
                            f"mean={self.phi_mean[i]:.3f}",
                            fillcolor=f'#{color_range[i][0]}',
                            fontcolor=textcolor[int(color_range[i][1])])
        
        for i in keep_features_idx:
            for j in keep_features_idx:
                if self.grounded_reduced_adjacency[i, j]==1:
                    dot.edge(f"x{j}", f"x{i}")
                    
        # Show the ground
        if show_ambiguous and len(self.ambiguous) != 0:
            node_label = ""
            if len(self.ambiguous) != 0:
                node_label += "Ambiguous"
                text = ["\n"+self.features_names[idx] if count%4==0 \
                        else self.features_names[idx] \
                            for count, idx in enumerate(self.ambiguous)]
                node_label += ",".join(text) + "\n"
            dot.node("ground", node_label, fillcolor='white')
            # Choose a representer of the ground
            i = next(iter(self.ambiguous))
            for j in keep_features_idx:
                if self.grounded_reduced_adjacency[i, j]==1:
                    dot.edge(f"x{j}", "ground", color='transparent')
        
        return dot
    
    
    
    def print_ranks(self):
        """ Print list of all features by ranks."""

        n_ranks = max(self.ranks)
        ranks_set = []
        for rank in range(n_ranks):
            ranks_set.append(set())
    
        # Add each feature to the right set
        for i, name in enumerate(self.features_names):
            ranks_set[ self.ranks[i]-1 ].add(name)
                
        # Print
        for rank, elements in enumerate(ranks_set):
            if rank == len(ranks_set)-1 and len(self.ambiguous) != 0:
                print("Ambiguous :")
            else:
                print(rank+1, ':')
            for feature in elements:
                print("    ", feature)



class RashomonPartialOrders(object):
    """ Class that encodes all possible poset for a range of tolerance epsilon """
    def __init__(self, phi_mean, minmax_attribs_lambda, gap_eps, 
                neg_eps, pos_eps, adjacency_eps, top_bottom):
        """
        Create an instance of the class
        
        Parameters
        ----------
        phi_mean : (N, d) `np.array`
            The mean feature attributions on each data instance
        minmax_attribs : lambda
            Function that takes epsilon and returns min_max_attribss
        gap_eps : (N,) `np.array`
            The critical threshold for pos/neg gap statements
        neg_eps : (N, d) `np.array`
            The critical threshold for neg attribution statements
        pos_eps : (N, d) `np.array`
            The critical threshold for pos attribution statements
        adjacency_eps : (N, d, d) `np.array`
            The critical threshold for relative importance statements
        """
        self.phi_mean = phi_mean
        self.minmax_attribs_lambda = minmax_attribs_lambda
        self.n_features = phi_mean.shape[1]
        self.gap_crit_eps = gap_eps
        self.pos_crit_eps = pos_eps
        self.neg_crit_eps = neg_eps
        self.adjacency_crit_eps = adjacency_eps
        self.top_bottom = top_bottom
    
    
    def minmax_attrib(self, epsilon):
        return self.minmax_attribs_lambda(epsilon)


    def get_poset(self, idx, epsilon, feature_names):
        """
        Return a PartialOrder object given the tolerance level
        and index of the instance to explain.
        
        Parameters
        ----------
        idx : `int`
            The index of the instance to explain
        epsilon : `float`
            The tolerance to error
        feature_names : `List(str)`
            The name of the input features
        
        Returns
        -------
        `uxai.PartialOrder`
            PO object representing the local explanation
        """
        # Gap is well defined
        if self.gap_crit_eps[idx] >= epsilon:
            # Positive attribution
            positive = self.pos_crit_eps[idx] >= epsilon
            # Negative attribution
            negative = self.neg_crit_eps[idx] >= epsilon
            # Ambiguous what the sign is
            ambiguous = set(np.where(~(positive | negative))[0])
            # Adjacency
            adjacency = np.zeros((self.n_features, self.n_features))
            adjacency[self.adjacency_crit_eps[idx] >= epsilon] = 1
            return PartialOrder(self.phi_mean[idx], adjacency, 
                                ambiguous, feature_names, self.top_bottom)
        # Gap is not well-defined so return None
        else:
            return None


    def get_utility(self, epsilon_space):
        """ 
        Search the space of possible epsilon values and return
        the utility u(epsilon) i.e. the number of statements on feature
        attributions one can make given the tolerance to error.
        """
        N = len(self.gap_crit_eps)
        n_features = self.pos_crit_eps.shape[1]
        cardinality = np.zeros((N, len(epsilon_space)))
        for i, epsilon in enumerate(epsilon_space):
            # For which instances can we define a gap?
            defined_gaps = np.where(self.gap_crit_eps > epsilon)[0]
            # Positive/Negative attribution
            pos_neg = (self.pos_crit_eps[defined_gaps] > epsilon) | \
                      (self.neg_crit_eps[defined_gaps] > epsilon)
            # Adjacency
            adjacency = self.adjacency_crit_eps[defined_gaps] > epsilon
            # Count the number positive/negative attribution statements
            count = np.sum(pos_neg, 1)
            # Count the number of relative importance statements
            for j in range(n_features):
                for k in range(n_features):
                    both_sign = pos_neg[:, j] & pos_neg[:, k]
                    count[both_sign] += adjacency[both_sign, j, k]
            cardinality[defined_gaps, i] = count

        return cardinality / (n_features * (n_features + 1) / 2 )



def intersect_total_orders(FI, feature_names, threshold=None, attribution=False):
    nb_features = FI.shape[1]
    adjacency = np.zeros((nb_features, nb_features)) # <= Adjacency Matrix

    if attribution:
        sign = np.sign(FI)
        FI = np.abs(FI)
    else:
        sign = np.ones(FI.shape)

    
    # Feature whose attribution changes sign are ambiguous
    ambiguous = set()
    for feature in range(nb_features):
        if len(np.unique(sign[:, feature]))==2:
            #print(f"Ambiguous : {feature_names[feature]}")
            ambiguous.add(feature)
        
        
        
    # Fill out the adjacency matrix
    for feature in range(nb_features):
        for second_feature in range(nb_features):
            # ignore diagonal
            if feature != second_feature:
                # All models agree that second_feature is more important
                if np.all(FI[:, feature] < FI[:, second_feature]):
                    adjacency[feature, second_feature] = 1

    return PartialOrder((sign*FI).mean(0), adjacency, ambiguous,
                        feature_names, top_bottom=True)



def color_nodes(phi_mean):
    def interpolate_rgb(rgb_list_1, rgb_list_2, interp):
        """ Linear interpolation interp * rbg_1 + (1 - interp) * rbg_2 """
        out = ''
        L1_norm = 0
        interp_value = [0, 0, 0]
        for color in range(3):
            interp_value[color] = round(interp * rgb_list_1[color] + \
                                        (1 - interp) * rgb_list_2[color])
            hex_color = hex( interp_value[color] )[2:]
            if len(hex_color) == 1:
                hex_color = '0' + hex_color
                
            L1_norm += interp_value[color]
            out += hex_color
            
        is_dark = L1_norm <= 350
        return out, is_dark
    
    # Get colors
    from .plots import color_dict
    colors = color_dict["default"]
    
    # Color the node w.r.t phi(h_mean)
    max_mean = max(map(abs, phi_mean))
    color_range = []
    for feature in range(len(phi_mean)):
        to_white = 1 - abs(phi_mean[feature]/max_mean)

        if phi_mean[feature] >= 0:
            color_range.append( interpolate_rgb(colors['zero'], 
                                                colors['pos'], to_white) )
        else:
            color_range.append( interpolate_rgb(colors['zero'], 
                                                colors['neg'], to_white) )
            
    return color_range