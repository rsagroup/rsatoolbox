#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model family graph visualization
TODO:
1. labeling option binary or indicators
2.
"""
import bisect
import operator as op
from functools import reduce
import numpy as np
#import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt


def ncr(n, r):
    """Calculates n choose r nCr.

    Parameters
    ----------
    n : int
        Total number of possible items
    r : int
        Number of items to choose

    Returns
    -------
    int
        Total number of possible ways of choosing r items from n items (n>r)

    """
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom


def get_node_position(index, num_models):
    """Get node position of a family member in model family graph visualization

    Parameters
    ----------
    index : int
        index of the family member
    num_models : int
        number of models
    family_size : int
        total number of family members

    Returns
    -------
    tuple
        x,y coordinates of the node

    """
    family_bounds = []
    num_members_at_level = []
    index = index+2

    for i in range(num_models+1):
        if i == 0:
            family_bounds.append(ncr(num_models, i))
        else:
            family_bounds.append(ncr(num_models, i)+family_bounds[i-1])
        num_members_at_level.append(ncr(num_models, i))
    y = bisect.bisect_left(family_bounds, index)
    x_range = np.linspace(-num_members_at_level[y]//2, num_members_at_level[y]//2,
                          num=num_members_at_level[y])
    x = x_range[index-family_bounds[y]-1]
    pos = x, y
    return pos


def show_family_graph(model_family, results, node_labels='presence',
                      node_property="color", method='corr', **kwargs):
    """visualizes the model family results in a graph

    Parameters
    ----------
    model_family : model_family object
        Model family for which we need to visualize the graph
    results : results object
        Evaluation results
    node_property : string
        To ilustrate evaluation results of each node by either color or area

    Returns
    -------
    None

    """
    G = nx.DiGraph()
    scores_model_family = np.mean(results.evaluations, axis=2)[0]
    pos = {}
    node_sizes = []
    node_colors = []
    num_models = len(model_family.models)

    # some plotting utility numbers
    min_node_area = 10
    min_edge_width = 0.05
    node_area_multiplier = 1000
    edge_width_multiiplier = 4
    labeldict = {}
    for index, family_member_id in enumerate(model_family.family_list):
        node_id = ''.join(map(str, family_member_id))
        if node_labels == 'presence':
            labeldict[node_id] = node_id
        elif node_labels == 'binary':
            binary_id = list(model_family.model_indices[index].astype(np.uint8))
            binary_id = ''.join(map(str, binary_id))
            labeldict[node_id] = binary_id

        node_area = int(min_node_area + node_area_multiplier * (
            scores_model_family[index]/scores_model_family.max()))
        node_color = scores_model_family[index]
        node_sizes.append(node_area)
        node_colors.append(node_color)
        G.add_node(node_id)
        pos[node_id] = get_node_position(index, num_models)

    for fi1, family_member_id1 in enumerate(model_family.family_list):
        id1 = ''.join(map(str, family_member_id1))
        for fi2, family_member_id2 in enumerate(model_family.family_list):
            id2 = ''.join(map(str, family_member_id2))
            if set(family_member_id2).issubset(set(family_member_id1)) and \
                len(family_member_id1) == len(family_member_id2) + 1:
                diff = scores_model_family[fi1] - scores_model_family[fi2]
                weight = abs(diff/(scores_model_family.max()-scores_model_family.min()))
                color = 'gray'
                if diff > 0:
                    G.add_edge(id2, id1, weight=min_edge_width + edge_width_multiiplier * weight,
                               color=color)
                else:
                    G.add_edge(id1, id2, weight=min_edge_width + edge_width_multiiplier * weight,
                               color=color)

    edges = G.edges()
    colors = [G[u][v]['color'] for u, v in edges]
    weights = [G[u][v]['weight'] for u, v in edges]
    if node_property == 'color':
        nx.draw_networkx(
            G, labels=labeldict, with_labels=True, pos=pos, node_color=node_colors,
            edgelist=list(edges), edge_color=colors, width=weights, node_size=800, **kwargs)
        vmin = min(node_colors)
        vmax = max(node_colors)
        sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), **kwargs)
        cbar = plt.colorbar(sm)
        cbar.set_label(method, rotation=270, size='x-large', labelpad=20)
        plt.show()
    elif node_property == 'area':
        nx.draw_networkx(
            G, labels=labeldict, with_labels=True, pos=pos, node_size=node_sizes,
            edgelist=list(edges), edge_color=colors, width=weights, **kwargs)
