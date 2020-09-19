#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2020-09-17

@author: caiw
"""

from typing import Tuple, List, Dict, Union

from matplotlib import pyplot
from matplotlib.axes import Axes
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from pyrsa.rdm import RDMs
from pyrsa.util.rdm_utils import category_condition_idxs


def rdm_comparison_scatterplot(rdms,
                               show_marginal_distributions: bool = True,
                               show_identity_line: bool = True,
                               highlight_category_selector: Union[str, List[int]] = None,
                               highlight_categories: List = None,
                               axlim: Tuple[float, float] = None,
                               hist_bins: int = 30,
                               ):
    """
    Plot dissimilarities for 2 or more RDMs

    Args:
        rdms (RDMs object or list-like of 2 RDMs objects):
            If one RDMs object supplied, each RDM within is compared against each other
            If two RDMs objects supplied (as list, tuple, etc.), each RDM in the first is compared against each RDM in the second
        show_marginal_distributions (bool):
            True (default): Show marginal distributions.
            False: Don't.
        show_identity_line (bool):
            True (default): Show identity line in each graph.
            False: Don't.
        highlight_category_selector (Optional. str or List[int]):
            EITHER: A RDMs.pattern_descriptor defining category labelling for conditions.
                OR: A list of ints providing category labels for each condition.
            If None or not supplied, no categories will be highlighted, in which case `highlight_categories` must also
            be None.
        highlight_categories (Optional. List):
            List of category labels to highlight. Must be compatible with `highlight_category_selector`.
        axlim (Optional. Tuple[float, float]):
            Set the axis limits for the figure.
            If None or not supplied, axis limits will be automatically determined.
        hist_bins (int, default 30):
            The number of bins to use in the histogram.

    Returns:
        matplotlib.pyplot.Figure containing the scatter plot (not shown).

    """

    rdms_x: RDMs  # RDM for the x-axis, or RDMs for facet columns
    rdms_y: RDMs  # RDM for the y-axis, or RDMs for facet rows

    # Handle rdms arg
    _msg_arg_rdms = "Argument `rdms` must be an RDMs or low."
    try:
        if isinstance(rdms, RDMs):
            # 1 supplied
            rdms_x, rdms_y = rdms, rdms
        else:
            # Check that only 2 supplied
            assert len(rdms) == 2
            rdms_x, rdms_y = rdms[0], rdms[1]
    except TypeError:
        raise ValueError(_msg_arg_rdms)
    except AssertionError as exc:
        raise ValueError(_msg_arg_rdms) from exc

    # Handle category highlighting args
    _msg_arg_highlight = "Arguments `highlight_category_selector` and `highlight_categories` must be compatible."
    try:
        if highlight_category_selector is None:
            assert highlight_categories is None
        else:
            assert highlight_categories is not None
            category_idxs = category_condition_idxs(rdms_x, highlight_category_selector)
            category_names = category_idxs.keys()
            assert all(c in category_names for c in highlight_categories)
    except AssertionError as exc:
        raise ValueError(_msg_arg_highlight) from exc

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
    # Remember axes for scatter plots now so we can draw to them all later
    scatter_axes: List[Axes] = []
    for scatter_col_idx, rdm_for_col in enumerate(rdms_x):
        is_leftmost_col = (scatter_col_idx == 0)
        if show_marginal_distributions:
            # distributions show in the first column, so need to bump the column index
            scatter_col_idx += 1
        # Since matplotlib ordering is left-to-right, top-to-bottom, we need to process the rows in reverse to get the
        # correct reference axis.
        for scatter_row_idx in reversed(range(n_rdms_y)):
            is_bottom_row = (scatter_row_idx == n_rdms_y - 1)

            # RDMs objects aren't iterators, so while we can do `for r in rdms`, we can't do `reversed(rdms)`.
            # Hence we have to pull the rdm out by its index.
            rdm_for_row = rdms_y[scatter_row_idx]

            if reference_axis is None:
                sub_axis: Axes = fig.add_subplot(gridspec[scatter_row_idx, scatter_col_idx])
                reference_axis = sub_axis
            else:
                sub_axis: Axes = fig.add_subplot(gridspec[scatter_row_idx, scatter_col_idx],
                                                 sharex=reference_axis, sharey=reference_axis)

            sub_axis.scatter(rdm_for_col.get_vectors(), rdm_for_row.get_vectors())
            scatter_axes.append(sub_axis)

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
        for col_idx, rdm_for_col in enumerate(rdms_x):
            if reference_hist is None:
                hist_axis: Axes = fig.add_subplot(gridspec[-1, col_idx + 1], sharex=reference_axis)
                reference_hist = hist_axis
            else:
                hist_axis: Axes = fig.add_subplot(gridspec[-1, col_idx + 1], sharex=reference_axis, sharey=reference_hist)
            hist_axis.hist(rdm_for_col.get_vectors().flatten(), histtype='step', fill=False, orientation='vertical',
                           bins=hist_bins)
            hist_axis.xaxis.set_visible(False)
            hist_axis.yaxis.set_visible(False)
            hist_axis.set_frame_on(False)
        # Flip to pointing downwards
        reference_hist.set_ylim(hist_axis.get_ylim()[::-1])

        # Add marginal distributions along the y axis
        reference_hist = None
        for row_idx, rdm_for_row in enumerate(rdms_y):
            if reference_hist is None:
                hist_axis: Axes = fig.add_subplot(gridspec[row_idx, 0], sharey=reference_axis)
                reference_hist = hist_axis
            else:
                hist_axis: Axes = fig.add_subplot(gridspec[row_idx, 0], sharey=reference_axis, sharex=reference_hist)
            hist_axis.hist(rdm_for_row.get_vectors().flatten(), histtype='step', fill=False, orientation='horizontal',
                           bins=hist_bins)
            hist_axis.xaxis.set_visible(False)
            hist_axis.yaxis.set_visible(False)
            hist_axis.set_frame_on(False)
        # Flip to pointing leftwards
        reference_hist.set_xlim(hist_axis.get_xlim()[::-1])

    # Add identity lines
    if show_identity_line:
        for ax in scatter_axes:
            # Prevent autoscale, else plotting from the origin causes the axes to rescale
            ax.autoscale(False)
            ax.plot([reference_axis.get_xlim()[0], reference_axis.get_xlim()[1]],
                    [reference_axis.get_ylim()[0], reference_axis.get_ylim()[1]])

    return fig
