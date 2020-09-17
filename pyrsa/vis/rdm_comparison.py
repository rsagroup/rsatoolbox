#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2020-09-17

@author: caiw
"""

from matplotlib import pyplot
from matplotlib.axes import Axes

from pyrsa.rdm import RDMs
from pyrsa.util.vis_utils import subplot_idx


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

    # main_axes = pyplot.subplot(n_rdms_x, n_rdms_y, sharex=True, sharey=True)

    for ix, rx in enumerate(rdms_x):
        for iy, ry in enumerate(rdms_y):
            pyplot.figure()

            sub_axis: Axes = pyplot.subplot(n_rdms_x, n_rdms_y, subplot_idx(ix, iy, n_rdms_x, n_rdms_y))

            pyplot.scatter(rx.get_vectors(), ry.get_vectors())

            # Hide the right and top spines
            sub_axis.spines['right'].set_visible(False)
            sub_axis.spines['top'].set_visible(False)

            # Square axes
            sub_axis.set_aspect('equal', adjustable='box')

    # return main_axes
