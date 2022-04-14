#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model family graph visualization
Author: @jdiedrichsen
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.cm as cm
import matplotlib.colors as colors
import seaborn as sb
import rsatoolbox as rsa

def family_graph(model_family,
                node_edgecolor = 'k',
                node_facecolor = 'w',
                node_size = 0.3,
                edge_width=1.0,
                edge_color='k',
                labels='name'):
    """Plots a model family graph with a flexible
    way of determining the way information is presented

    Args:
        model_family (ModelFamily): ModelFamily object
        node_edgecolor (ndarray):
            Color of the nodes edges
                None: no edge
                1-d ndarray: mapped to the current color mapping
                plt.color: Constant color
        node_facecolor (plt.color or ndarray):
            Determines the color or the node faces
                None: no edge
                1-d ndarray: mapped to the current color mapping
                plt.color: Constant color
        edge_width (float or ndarray):
            Determines the width of edges to the Edges
                None: No edge
                1d-ndarray: scales proportional to absolute difference between each pair of nodes
        edge_color (float or ndarray):
            Edge color
    """

    # Get the layout from the model family
    [x,y]   = model_family.get_layout()

    # generate axis with appropriate labels
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_xlim(np.min(x)-1,np.max(x)+1)
    ax.set_ylim(np.min(y)-1,np.max(y)+1)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Draw connections, if desired
    if edge_width is not None:
        connect = model_family.get_connectivity()
        [fr,to]=np.where(connect==1)

        # Determine width and color of all the edges
        if type(edge_width) is not np.ndarray:
            edge_width = np.ones(fr.shape) * edge_width
        elif edge_width.shape[0] == model_family.n_models:
            edge_width = np.abs(edge_width[to]-edge_width[fr])


        for i in range(fr.shape[0]):
            l = mlines.Line2D([x[fr[i]], x[to[i]]],
                              [y[fr[i]], y[to[i]]],
                              color=[0,0,0,0.3],
                              linewidth=edge_width[i],
                              zorder=1)
            ax.add_line(l)

    # Determine color range and map
    if type(node_facecolor) is np.ndarray:
        if (node_facecolor.ndim==1):
            cmap = cm.Reds
            normc = colors.Normalize(vmin=np.min(node_facecolor), vmax=np.max(node_facecolor))
            node_facecolor = cmap(normc(node_facecolor))
    elif type(node_facecolor) is str:
        node_facecolor = [colors.to_rgba(node_facecolor)]*model_family.n_models

    if type(node_size) is not np.ndarray:
        node_size = np.ones((model_family.n_models,)) * node_size

    # Draw model circles
    for i in range(x.shape[0]):
        circle = plt.Circle((x[i], y[i]), node_size[i],
                facecolor=node_facecolor[i],
                edgecolor=node_edgecolor,
                zorder = 30)
        ax.add_patch(circle)
    pass
    # Add labels to models

def component_barplot(data,type='posterior'):
    """Plots the result of a component analysis
    Args:
        data (pd.DataFrame): Data Frame with observations in rows and components
        type (str, optional): 'posterior' or 'bf' (posterior is default)
    """
    D = data.melt()
    ax = plt.gca()
    sb.barplot(data=D,x='variable',y='value')
    if (type=='posterior'):
        ax.set_ylabel('Posterior')
        plt.axhline(1/(1+np.exp(1)),color='k',ls=':')
        plt.axhline(0.5,color='k',ls='--')
    elif(type=='bf'):
        ax.set_ylabel('Bayes Factor')
        plt.axhline(0,color='k',ls='--')
    ax.set_xlabel('Component')