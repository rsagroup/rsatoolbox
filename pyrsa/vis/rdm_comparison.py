#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2020-09-17

@author: caiw
"""
from typing import Tuple

from matplotlib import pyplot
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from pyrsa.rdm import RDMs


def rdm_comparison_scatterplot(rdms,
                               show_marginal_distributions: bool = True,
                               axlim: Tuple[float, float] = None,
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

    # TODO: make this a parameter
    HIST_BINS = 30

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
    gridspec = GridSpec(
        nrows=grid_n_rows,
        ncols=grid_n_cols,
        width_ratios=grid_width_ratios,
        height_ratios=grid_height_ratios,
    )

    # To share x and y axes when using gridspec you need to specify which axis to use as references.
    # The reference axes will be those in the first column and those in the last row.
    reference_axis = None
    # TODO: rename these vars
    for ix, rx in enumerate(rdms_x):
        is_leftmost_col = (ix == 0)
        if show_marginal_distributions:
            # distributions show in the first column, so need to bump the column index
            ix += 1
        # Since matplotlib ordering is left-to-right, top-to-bottom, we need to process the rows in reverse to get the
        # correct reference axis.
        for iy in reversed(range(n_rdms_y)):
            is_bottom_row = (iy == n_rdms_y - 1)

            # RDMs objects aren't iterators, so while we can do `for r in rdms`, we can't do `reversed(rdms)`.
            # Hence we have to pull the rdm out by its index.
            ry = rdms_y[iy]

            # To do
            if reference_axis is None:
                sub_axis: Axes = fig.add_subplot(gridspec[iy, ix])
                reference_axis = sub_axis
            else:
                sub_axis: Axes = fig.add_subplot(gridspec[iy, ix], sharex=reference_axis, sharey=reference_axis)

            sub_axis.scatter(rx.get_vectors(), ry.get_vectors())

            # Hide the right and top spines
            sub_axis.spines['right'].set_visible(False)
            sub_axis.spines['top'].set_visible(False)

            # Hide all but the outermost ticklabels
            if not is_bottom_row:
                pyplot.setp(sub_axis.get_xticklabels(), visible=False)
            if not is_leftmost_col:
                pyplot.setp(sub_axis.get_yticklabels(), visible=False)

            # Square axes
            sub_axis.set_aspect('equal', adjustable='box')

    # Apply specified axlim to the reference axis:
    if axlim is not None:
        reference_axis.set_xlim(axlim[0], axlim[1])
        reference_axis.set_ylim(axlim[0], axlim[1])

    if show_marginal_distributions:
        # Add marginal distributions along the x axis
        reference_hist = None
        for ix, rx in enumerate(rdms_x):
            if reference_hist is None:
                hist_axis: Axes = fig.add_subplot(gridspec[-1, ix + 1], sharex=reference_axis)
                reference_hist = hist_axis
            else:
                hist_axis: Axes = fig.add_subplot(gridspec[-1, ix + 1], sharex=reference_axis, sharey=reference_hist)
            hist_axis.hist(rx.get_vectors().flatten(), histtype='step', fill=False, orientation='vertical',
                           bins=HIST_BINS)
            hist_axis.xaxis.set_visible(False)
            hist_axis.yaxis.set_visible(False)
            hist_axis.set_frame_on(False)
        # Flip to pointing downwards
        reference_hist.set_ylim(hist_axis.get_ylim()[::-1])

        # Add marginal distributions along the y axis
        reference_hist = None
        for iy, ry in enumerate(rdms_y):
            if reference_hist is None:
                hist_axis: Axes = fig.add_subplot(gridspec[iy, 0], sharey=reference_axis)
                reference_hist = hist_axis
            else:
                hist_axis: Axes = fig.add_subplot(gridspec[iy, 0], sharey=reference_axis, sharex=reference_hist)
            hist_axis.hist(ry.get_vectors().flatten(), histtype='step', fill=False, orientation='horizontal',
                           bins=HIST_BINS)
            hist_axis.xaxis.set_visible(False)
            hist_axis.yaxis.set_visible(False)
            hist_axis.set_frame_on(False)
        # Flip to pointing leftwards
        reference_hist.set_xlim(hist_axis.get_xlim()[::-1])

    return fig
