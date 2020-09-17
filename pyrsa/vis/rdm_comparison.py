#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2020-09-17

@author: caiw
"""

from matplotlib import pyplot
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pyrsa.rdm import RDMs


def rdm_comparison_scatterplot(rdms,
                               show_marginal_distributions: bool = True,
                               **kwargs):
    """

    Args:
        rdms (RDMs object or list-like of 2 RDMs objects):
            If one RDMs object supplied, each RDM within is compared against each other
            If two RDMs objects supplied (as list, tuple, etc.), each RDM in the first is compared against each RDM in the second
        show_marginal_distributions (bool):
            True (default): Show marginal distributions
            False: Don't show marginal distributions

        additional kwargs pass through to scatterplot


    Returns:
        axes object of produced figure

    """

    _msg_arg_rdms = "Argument `rdms` must be an RDMs or low"

    rdms_x: RDMs  # RDM for the x-axis, or RDMs for facet columns
    rdms_y: RDMs  # RDM for the y-axis, or RDMs for facet rows

    # Handle rdms arg
    if isinstance(rdms, RDMs):
        # 1 supplied
        rdms_x, rdms_y = rdms, rdms
    else:
        # Check that only 2 supplied
        try:
            assert len(rdms) == 2
        except TypeError:
            raise ValueError(_msg_arg_rdms)
        except AssertionError:
            raise ValueError(_msg_arg_rdms)
        rdms_x, rdms_y = rdms[0], rdms[1]

    n_rdms_x = rdms_x.n_rdm
    n_rdms_y = rdms_y.n_rdm

    fig: Figure = pyplot.figure(figsize=(8, 8))

    # Set up gridspec
    grid_n_rows = n_rdms_y
    grid_n_cols = n_rdms_x
    grid_width_ratios = tuple(6 for _ in range(grid_n_cols))
    grid_height_ratios = tuple(6 for _ in range(grid_n_rows))
    if show_marginal_distributions:
        # Add extra row & col for marginal distributions
        grid_n_rows += 1
        grid_n_cols += 1
        grid_width_ratios = (1, *grid_width_ratios)
        grid_height_ratios = (*grid_height_ratios, 1)
    gridspec = fig.add_gridspec(
        nrows=grid_n_rows,
        ncols=grid_n_cols,
        width_ratios=grid_width_ratios,
        height_ratios=grid_height_ratios,
        figure=fig,
    )

    # TODO: rename these vars
    for ix, rx in enumerate(rdms_x):
        for iy, ry in enumerate(rdms_y):

            sub_axis: Axes = fig.add_subplot(gridspec[ix, iy])

            pyplot.scatter(rx.get_vectors(), ry.get_vectors())

            # Hide the right and top spines
            sub_axis.spines['right'].set_visible(False)
            sub_axis.spines['top'].set_visible(False)

            # Square axes
            sub_axis.set_aspect('equal', adjustable='box')

    return fig
