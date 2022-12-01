""" Plotly interactive plot """
import torch
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from copy import deepcopy
from graphviz import Digraph
import numpy as np

from .confidence import abs_map_CIs


color_dict = {}
color_dict["default"] = {'zero' : [255, 255, 255], 
                         'pos':  [0, 102, 255], 
                         'neg' : [255, 69, 48]}
color_dict["DEEL"] = {'zero' : [255, 255, 255], 
                      'pos':  [0, 69, 138], 
                      'neg' : [255, 69, 48]}



def plot_QoIs(qois, qois_labels, to_sort_qois=None, to_sort_qois_labels=None, descending = True):
    """Parallel Coordinate Plot of the QoIs and
    the testset performance (Rashomon Set)

    Args:
        qois (Tensor): Tensor of qois (e.g. hstack(Performance, predictions)).
        qois_labels (list): List of qois labels (e.g. ["performance", "prediction"]).
        to_sort_qois (tensor, optional): Tensor of values to sort qois with. Defaults to None.
        to_sort_qois_labels (list, optional): List of the labels of the qois to sort. Defaults to None.
        descending (bool, optional): Sort qois in a descending order instead of an ascending one. Defaults to True.

    Returns:
        plotly.graph_objects.Figure: Plotly Parallel Coordinate Plot of QoIs.
    """
    # QoIs
    if len(qois.size()) == 1:
        qois = qois.reshape(-1, 1)
    qois = qois.detach().cpu().numpy()
    dims = [dict(range = [qois[:, i].min(), qois[:, i].max()],
                 label = qois_labels[i], values = qois[:, i].ravel())
                                                 for i in range(qois.shape[1])]
    
    if to_sort_qois != None:
        # Sorted QoIs
        min_qois = to_sort_qois.min().item()
        max_qois = to_sort_qois.max().item()
        sorted_idx = torch.argsort(to_sort_qois.min(dim = 0)[0], descending = descending)
        to_sort_qois = to_sort_qois.cpu().numpy()
        dims += [dict(range = [min_qois, max_qois], 
                      label = to_sort_qois_labels[sorted_idx[i]], 
                      values = to_sort_qois[:, sorted_idx[i]]) \
                                         for i in range(to_sort_qois.shape[1])]
    
    # Plotly Parallel Coordinate Plot of QoIs
    fig = go.Figure(
        data=go.Parcoords(
            dimensions = dims
        )
    )
    return fig



def pcp(phis, feature_labels, total_attrib=None, test_error=None, xerr=None, 
                                                        threshold=None, linewidth=2):
    """
    Plot the Parallel Coordinate Plot of the feature attribution/importance
    of each model

    Parameters
    ----------
    phis : np.array
        Array of size (n_features,) or (n_lines, n_features)
    feature_labels : List
        Shape (n_features,)
    total_attrib : np.array, optional
        Array of size (1,) when phis is (n_features,) or (n_lines, 1) when
        phis is (n_lines, n_features). The default is None.
    xerr : np.array, optional
        Array of size (n_features, 2) when phis is (n_features,) or 
        (n_lines, n_features, 2) when phis is (n_lines, n_features). 
        The default is None.
    threshold : float, optional
        Value of the attribution threshold used to define negligible. 
        The default is None.
    """
    n_features = phis.shape[-1]
    n_values = n_features
    
    # check if n_lines = 1
    if phis.ndim == 1:
        n_lines = 1
        # phis must be (n_lines, n_features)
        phis = phis.reshape((1, -1))
        if xerr is not None:
            # reshape xerr into (n_lines, n_features, 2)
            xerr = xerr[np.newaxis, :]
        if total_attrib is not None:
            # reshape total attrib into (n_lines, n_features)
            total_attrib = total_attrib[:, np.newaxis]
    else:
        n_lines = phis.shape[0]
        
    # At this point we can assume that 
    # phis (n_lines, n_features)
    # xerr (n_lines, n_features, 2)
    # total_attrib (n_lines, 1)    
    
    # Importance
    phi_mean = phis.mean(0)
    FI_mean = np.abs(phi_mean)
    # Sort feature importance
    sorted_idx = np.argsort(FI_mean)
    
    # Sort the values
    values = phis[:, sorted_idx]
    mean_values = phi_mean[sorted_idx]
    if xerr is not None:
        xerr = xerr[:, sorted_idx]
    y_labels = [feature_labels[i] for i in sorted_idx]
        
    # Add attribution on top of all values if requested
    if total_attrib is not None:
        values = np.hstack((values, total_attrib))
        mean_values = np.append(mean_values, total_attrib.mean())
        if xerr is not None:
            row_shape = [xerr.shape[0]] + [1] + [xerr.shape[2]]
            xerr = np.concatenate((xerr, np.zeros(row_shape)), axis=1)
        y_labels.append("Attribution")
        n_values += 1


    
    # Add test error as labels if requested
    labels =[None]*len(values)
    if test_error is not None:
        labels=[f"{te.item():.3f}" for te in test_error]

        
    # alpha = 100 / len(phi)
    alpha = 1
    colors = deepcopy(color_dict["DEEL"])
    colors['pos'] = np.array(colors['pos'])/255.
    colors['neg'] = np.array(colors['neg'])/255.
    
    # Attributions
    fig, ax= plt.subplots()

    # Add error bars
    if xerr is not None:
        for line, bar in zip(values, xerr):
            plt.errorbar(line, range(n_values), xerr=bar.T, 
                         solid_capstyle='projecting', capsize=5, alpha=alpha, 
                         color=colors['pos'], zorder=2)
    else:
        for i,line in enumerate(values):
            if test_error is not None:
                cmap=plt.get_cmap('winter')
                if len(values)<10:
                    color = cmap(i/len(values))
                else:
                    color = colors['pos']
            else:
                color = colors['pos']
            plt.errorbar(line, range(n_values), solid_capstyle='projecting',
                         linewidth=linewidth, alpha=alpha, color=color, 
                         label=labels[i], zorder=2)
            
    # With multiple lines, draw the mean line in red
    if n_lines >= 2:
       plt.plot(mean_values, range(n_values), color=colors['neg'], 
                           linewidth=1.25*linewidth, label=labels[-1], zorder=3)
    
    plt.yticks(ticks=range(n_values), labels=y_labels)
    plt.ylim(0, n_values-1)
    plt.plot(0, 0, 'b')
    plt.grid('on', zorder=1)
    
    if test_error is not None:
        plt.legend(title = "Test RMSE", framealpha=1)

    fig.tight_layout()




def bar(phis, feature_labels, threshold=None, xerr=None, absolute=False, ax=None):

    num_features = len(feature_labels)

    if absolute:
        bar_mapper = lambda x : np.abs(x)
        if xerr is not None:
            min_CIs, max_CIs = abs_map_CIs(phis, xerr)
            xerr = np.abs(np.abs(phis) - np.vstack((min_CIs, max_CIs)))
    else:
        bar_mapper = lambda x : x
        
    ordered_features = np.argsort(bar_mapper(phis))
    y_pos = np.arange(0, len(ordered_features))
        
    if ax is None:
        plt.figure()
        # plt.gcf().set_size_inches(16, 10)
        ax = plt.gca()
        
    negative_phis = (phis < 0).any() and not absolute
    if negative_phis:
        ax.axvline(0, 0, 1, color="k", linestyle="-", linewidth=1, zorder=1)
    if threshold:
        ax.axvline(threshold, 0, 1, color="k", linestyle="--", linewidth=2, zorder=1)
        if negative_phis:
            ax.axvline(-threshold, 0, 1, color="k", linestyle="--", linewidth=2, zorder=1)
    # draw the bars
    bar_width = 0.7
    
    # Get DEEL colors
    colors = deepcopy(color_dict["DEEL"])
    colors['pos'] = np.array(colors['pos'])/255.
    colors['neg'] = np.array(colors['neg'])/255.
    
    if xerr is None:
        ax.barh(
            y_pos, bar_mapper(phis[ordered_features]),
            bar_width, align='center',
            color=[colors['neg'] if phis[ordered_features[j]] <= 0 
                   else colors['pos'] for j in range(len(y_pos))], 
            edgecolor=(1,1,1,0.8), capsize=5
        )
    else:
        if xerr.ndim == 2 and xerr.shape[0] == 2:
            xerr = xerr[:, ordered_features]
        else:
            xerr = xerr[ordered_features]
        ax.barh(
            y_pos, bar_mapper(phis[ordered_features]),
            bar_width, xerr=xerr, align='center',
            color=[colors['neg'] if phis[ordered_features[j]] <= 0 
                   else colors['pos'] for j in range(len(y_pos))], 
            edgecolor=(1,1,1,0.8), capsize=5
        )

    yticklabels = [feature_labels[j] for j in ordered_features]
    ax.set_yticks(ticks=list(y_pos))
    ax.set_yticklabels(yticklabels, fontsize=15)


    # put horizontal lines for each feature row
    for i in range(num_features):
        ax.axhline(i, color="k", lw=0.5, alpha=0.5, zorder=-1)

    xmin,xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    
    if negative_phis:
        ax.set_xlim(xmin - (xmax-xmin)*0.05, xmax + (xmax-xmin)*0.05)
    else:
        ax.set_xlim(xmin, xmax + (xmax-xmin)*0.05)
    
    plt.gcf().tight_layout()
        


def plot_hasse_diagram(phi_mean, adjaency, ranks, top_ranks, node_labels, 
                                                               drop_zeros=True):
    """Plots local Hasse diagram and exports it as filename.pdf .

    Args:
        phi_mean (np.array): Mean attribution for each feature. (nb_features, )
        adjacency (np.array): Partial order adjacency matrix. (nb_features, nb_features)
        ranks (list): List of ranks based on the partial order adjacency matrix.
        top_ranks (int): Largest rank to show in Hasse diagram
        features_names (list): List of names of all features.
        x_map (list): List of the features' values of the instance.
        filename (string): Filename to save the diagram locally in.
    """

    def interpolate_rgb(rgb_list_1, rgb_list_2, interp):
        """ Linear interpolation interp * rbg_1 + (1 - interp) * rbg_2 """
        out = ''
        for color in range(3):
            hex_color = hex( round(interp * rgb_list_1[color] + \
                                   (1 - interp) * rgb_list_2[color]) )[2:]
            if len(hex_color) == 1:
                hex_color = '0' + hex_color
    
            out += hex_color
        return out
    
    
    # Directed Graph of partial ordering
    dot = Digraph(comment='Feature Importance', graph_attr={'ranksep': "1.0"},
                  node_attr={'shape': 'rectangle', 'color': 'black',
                             'style': 'filled'})
    
    my_pink = [255, 86, 242]
    my_blue = [0, 102, 255]
    # Color the node w.r.t phi(h_mean)
    max_mean = max(map(abs, phi_mean))
    color_range = []
    for feature in range(len(node_labels)):
        to_white = 1 - abs(phi_mean[feature]/max_mean)

        if phi_mean[feature] >= 0:
            color_range.append( interpolate_rgb([255] * 3, my_blue, to_white) )
        else:
            color_range.append( interpolate_rgb([255] * 3, my_pink, to_white) )

    # Dont consider features with a zero attribution
    if drop_zeros:
        keep_features_idx = [f for f in range(len(node_labels)) if not phi_mean[f]==0]
    else:
        keep_features_idx = [f for f in range(len(node_labels))]

    # Dont consider feature that are incomparable to all others
    # sub_adjaency = adjaency[np.ix_(keep_features_idx,keep_features_idx)]
    # uncertain_features = set(np.intersect1d(np.where((sub_adjaency==0).all(axis=1))[0], 
    #                                     np.where((sub_adjaency==0).all(axis=0))[0]))
    # keep_features_idx = [keep_features_idx[i] for i in range(len(keep_features_idx))\
    #                                             if not i in uncertain_features]
        
    # Dont consider features that have a higher rank than top_ranks
    keep_features_idx = [keep_features_idx[i] for i in range(len(keep_features_idx))
                            if ranks[keep_features_idx[i]] <= top_ranks]

    
    # Store ranks in sets
    n_ranks = min(max(ranks), top_ranks)
    ranks_set = []
    for rank in range(n_ranks):
        ranks_set.append(set())

    # Add each feature to the right set
    for i in keep_features_idx:
        ranks_set[ ranks[i]-1 ].add(i)

    # Print
    for elements in ranks_set:
        with dot.subgraph() as s:
            s.attr(rank='same')
            # Loop over all features of the same rank
            for i in elements:
                s.node(f"x{i}",
                        f"{node_labels[i]}\n"+\
                        f"mean_imp={phi_mean[i]:.3f}",
                        fillcolor=f'#{color_range[i]}')
    
    for i in keep_features_idx:
        for j in keep_features_idx:
            if adjaency[i, j] == 1:
                dot.edge(f"x{j}", f"x{i}")
              
    # if args.wandb.wandb:
    #     wandb.log({f"QoIs/{explainer}": fig})
    #     im = Image.open(path)
    #     wandb.log({f"QoIs/Hasse - {explainer}": [wandb.Image(im, caption=f"Hasse Diagram - {explainer}")]})
    # else:
    #     fig.show()

    # if args.data.save_matrix:
    #     #save partial_order matrix
    #     with open("global.txt", mode='w') as fich:
    #         np.savetxt(fich, partial_order.numpy().astype(int), delimiter=',', fmt='%d')

    
    return dot
    