#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot showing an RDMs object
"""

import os.path
import inspect
import collections
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pyrsa import vis
from pyrsa.rdm import rank_transform
from pyrsa.vis.colors import rdm_colormap

MODULE_DIR = os.path.dirname(inspect.getfile(inspect.currentframe()))
RDM_STYLE = os.path.join(MODULE_DIR, "rdm.mplstyle")


def show_rdm(
    rdm,
    do_rank_transform=False,
    pattern_descriptor=None,
    cmap=None,
    rdm_descriptor=None,
    n_column=None,
    n_row=None,
    show_colorbar=None,
    gridlines=None,
    num_pattern_groups=None,
    figsize=None,
    nanmask=None,
    style=RDM_STYLE,
    vmin=None,
    vmax=None,
    icon_spacing=1.0,
    linewidth=0.5,
):
    """Heatmap figure for RDMs instance, with one panel per RDM.

    Parameters
    ----------
    rdm : pyrsa.rdm.RDMs
        RDMs object to be plotted
    do_rank_transform : bool
        whether we should do a rank transform before plotting
    pattern_descriptor : str
        name of a pattern descriptor which will be used as an axis label
    cmap : color map
        colormap or identifier for a colormap to be used
        conventions as for matplotlib colormaps
    rdm_descriptor : str or key into rdm_descriptor
        key for rdm_descriptor to use as panel title, or str for direct labeling
    n_column : int
        number of columns in subplot arrangement
    n_row : int
        number of rows in subplot arrangement
    show_colorbar : str
        whether to display a colorbar. If 'panel' a colorbar is added next to
        each RDM. If 'figure' a shared colorbar (and scale) is used across panels.
    gridlines : list
        Add gridlines at these positions (by default inferred from num_pattern_groups)
    num_pattern_groups : int
        Number of rows/columns for any image labels. Also determines gridlines frequency
        by default (so e.g., num_pattern_groups=3 with results in gridlines
        every 3 rows/columns)
    figsize : tuple
        mpl.Figure argument. By default we auto-scale to achieve a figure that fits on a
        standard A4 / US Letter page in portrait orientation
    nanmask : np.array
        boolean mask defining RDM parts to suppress (by default, the diagonals)
    style : str
        path to mplstyle file (default pyrsa/vis/rdm.mplstyle)
    vmin : float
        imshow argument
    vmax : float
        imshow argument
    icon_spacing : float
        control spacing of image labels - 1. means no gap, 1.1 means pad 10%, .9 means
        overlap 10% etc
    linewidth : float
    """

    if show_colorbar and not show_colorbar in ("panel", "figure"):
        raise ValueError(
            f"show_colorbar can be None, panel or figure, got: {show_colorbar}"
        )
    if do_rank_transform:
        rdm = rank_transform(rdm)
        if not all(var is None for var in [vmin, vmax]):
            raise ValueError(
                "manual limits (vmin, vmax) unsupported when do_rank_transform"
            )
    if nanmask is None:
        nanmask = np.eye(rdm.n_cond, dtype=bool)
    n_panel = rdm.n_rdm
    if show_colorbar == "figure":
        n_panel += 1
        # need to keep track of global CB limits
        if any(var is None for var in [vmin, vmax]):
            # need to load the RDMs here (expensive)
            rdmat = rdm.get_matrices()
            if vmin is None:
                vmin = rdmat[:, (nanmask == False)].min()
            if vmax is None:
                vmax = rdmat[:, (nanmask == False)].max()
    if n_column is None and n_row is None:
        n_column = np.ceil(np.sqrt(n_panel))
    if n_row is None:
        n_row = np.ceil(n_panel / n_column)
    if n_column is None:
        n_column = np.ceil(n_panel / n_row)
    if (n_column * n_row) < rdm.n_rdm:
        raise ValueError(
            f"invalid n_row*n_column specification for {n_panel} rdms: {n_row}*{n_column}"
        )
    if figsize is None:
        # scale with number of RDMs, up to a point (the intersection of A4 and us
        # letter)
        figsize = (min(2 * n_column, 8.3), min(2 * n_row, 11))
    if cmap is None:
        cmap = rdm_colormap()
    if not np.any(gridlines):
        # empty list to disable gridlines
        gridlines = []
        if num_pattern_groups:
            # grid by pattern groups if they exist and explicit grid setting does not
            gridlines = np.arange(
                num_pattern_groups - 0.5, rdm.n_cond + 0.5, num_pattern_groups
            )
    if num_pattern_groups is None or num_pattern_groups == 0:
        num_pattern_groups = 1
    # we don't necessarily have the same number of RDMs as panels, so need to stop the
    # loop when we've plotted all the RDMs
    rdms_gen = (this_rdm for this_rdm in rdm)
    # return values are
    # image, axis, colorbar, x_labels, y_labels
    # some are global for figure, others local. Perhaps dicts indexed by axis is easiest
    return_handles = collections.defaultdict(dict)
    with plt.style.context(style):
        fig, ax_array = plt.subplots(
            nrows=int(n_row),
            ncols=int(n_column),
            sharex=True,
            sharey=True,
            squeeze=False,
            figsize=figsize,
        )
        # reverse panel order so unfilled rows are at top instead of bottom
        ax_array = ax_array[::-1]
        for row_ind, row in enumerate(ax_array):
            for col_ind, panel in enumerate(row):
                try:
                    return_handles[panel]['image'] = show_rdm_panel(
                        next(rdms_gen),
                        ax=panel,
                        cmap=cmap,
                        nanmask=nanmask,
                        rdm_descriptor=rdm_descriptor,
                        gridlines=gridlines,
                        vmin=vmin,
                        vmax=vmax,
                    )
                except StopIteration:
                    # hide empty panels
                    panel.set_visible(False)
                    continue
                except:
                    raise
                if show_colorbar == "panel":
                    # needs to happen before labels because it resizes the axis
                    return_handles[panel]['colorbar'] = _rdm_colorbar(
                        mappable=return_handles[panel]['image'],
                        fig=fig,
                        ax=panel,
                        title=rdm.dissimilarity_measure,
                    )
                if col_ind == 0 and pattern_descriptor:
                    return_handles[panel]['y_labels'] = add_descriptor_y_labels(
                        rdm,
                        pattern_descriptor,
                        ax=panel,
                        num_pattern_groups=num_pattern_groups,
                        icon_spacing=icon_spacing,
                        linewidth=linewidth,
                    )
                if row_ind == 0 and pattern_descriptor:
                    return_handles[panel]['x_labels'] = add_descriptor_x_labels(
                        rdm,
                        pattern_descriptor,
                        ax=panel,
                        num_pattern_groups=num_pattern_groups,
                        icon_spacing=icon_spacing,
                        linewidth=linewidth,
                    )
        if show_colorbar == "figure":
            # key challenge is to obtain a similarly-sized colorbar to the 'panel' case
            # BUT positioned centered on the reserved subplot axes
            cbax_parent = ax_array[-1, -1]
            cbax_parent_orgpos = cbax_parent.get_position(original=True)
            # use last instance of 'image' (should all be yoked at this point)
            return_handles[fig]['colorbar'] = _rdm_colorbar(
                mappable=image, fig=fig, ax=cbax_parent, title=rdm.dissimilarity_measure
            )
            cbax_pos = return_handles[fig]['colorbar'].ax.get_position()
            # halfway through panel, less the width/height of the colorbar itself
            x0 = (
                cbax_parent_orgpos.x0
                + cbax_parent_orgpos.width / 2
                - cbax_pos.width / 2
            )
            y0 = (
                cbax_parent_orgpos.y0
                + cbax_parent_orgpos.height / 2
                - cbax_pos.height / 2
            )
            return_handles[fig]['colorbar'].ax.set_position([x0, y0, cbax_pos.width, cbax_pos.height])

    return fig, ax_array, return_handles


def _rdm_colorbar(mappable=None, fig=None, ax=None, title=None):
    cb = fig.colorbar(
        mappable=mappable,
        ax=ax,
        shrink=0.25,
        aspect=5,
        ticks=matplotlib.ticker.LinearLocator(numticks=3),
    )
    cb.ax.set_title(title, loc="left", fontdict=dict(fontweight="normal"))
    return cb


def show_rdm_panel(
    rdm,
    ax=None,
    cmap=None,
    nanmask=None,
    rdm_descriptor=None,
    gridlines=[],
    vmin=None,
    vmax=None,
):
    if ax is None:
        ax = plt.gca()
    if rdm.n_rdm > 1:
        raise ValueError("expected single rdm - use show_rdm for multi-panel figures")
    rdmat = rdm.get_matrices()[0, :, :]
    if np.any(nanmask):
        rdmat[nanmask] = np.nan
    image = ax.imshow(rdmat, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlim(-0.5, rdm.n_cond - 0.5)
    ax.set_ylim(rdm.n_cond - 0.5, -0.5)
    ax.xaxis.set_ticks(gridlines)
    ax.yaxis.set_ticks(gridlines)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.xaxis.set_ticks(np.arange(rdm.n_cond), minor=True)
    ax.yaxis.set_ticks(np.arange(rdm.n_cond), minor=True)
    # hide minor ticks by default
    ax.xaxis.set_tick_params(length=0, which="minor")
    ax.yaxis.set_tick_params(length=0, which="minor")
    if rdm_descriptor in rdm.rdm_descriptors:
        ax.set_title(rdm.rdm_descriptors[rdm_descriptor][0])
    elif isinstance(rdm_descriptor, str):
        ax.set_title(rdm_descriptor)
    return image


def add_descriptor_x_labels(
    rdm, descriptor, ax=None, num_pattern_groups=None, icon_spacing=None, linewidth=None
):
    if ax is None:
        ax = plt.gca()
    return _add_descriptor_labels(
        rdm,
        descriptor,
        axis=ax.xaxis,
        other_axis=ax.yaxis,
        horizontalalignment="center",
        num_pattern_groups=num_pattern_groups,
        icon_spacing=icon_spacing,
        linewidth=linewidth,
    )


def add_descriptor_y_labels(
    rdm, descriptor, ax=None, num_pattern_groups=None, icon_spacing=None, linewidth=None
):
    if ax is None:
        ax = plt.gca()
    return _add_descriptor_labels(
        rdm,
        descriptor,
        axis=ax.yaxis,
        other_axis=ax.xaxis,
        horizontalalignment="right",
        num_pattern_groups=num_pattern_groups,
        icon_spacing=icon_spacing,
        linewidth=linewidth,
    )


def _add_descriptor_labels(
    rdm,
    descriptor,
    axis,
    other_axis,
    horizontalalignment="center",
    num_pattern_groups=None,
    icon_spacing=None,
    linewidth=None,
):
    """ adds a descriptor as ticklabels to the axis (XAxis or YAxis instance)"""
    desc = rdm.pattern_descriptors[descriptor]
    if isinstance(desc[0], vis.Icon):
        # annotated labels with Icon
        im_width_pix = max(this_desc.final_image.width for this_desc in desc)
        im_height_pix = max(this_desc.final_image.height for this_desc in desc)
        im_max_pix = max(im_width_pix, im_height_pix) * icon_spacing
        n_to_fit = np.ceil(rdm.n_cond / num_pattern_groups)
        axis.figure.canvas.draw()
        extent = axis.axes.get_window_extent(axis.figure.canvas.get_renderer())
        ax_size_pix = max((extent.width, extent.height))
        size = (ax_size_pix / n_to_fit) / im_max_pix
        # from proportion of original size to figure pixels
        offset = im_max_pix * size
        label_handles = []
        for group_ind in range(num_pattern_groups - 1, -1, -1):
            position = offset * 0.2 + offset * group_ind
            ticks = np.arange(group_ind, rdm.n_cond, num_pattern_groups)
            if isinstance(axis, matplotlib.axis.XAxis):
                label_handles.append([
                    this_desc.x_tick_label(
                        this_x,
                        size,
                        offset=position,
                        linewidth=linewidth,
                        ax=axis.axes,
                        )
                    for (this_x, this_desc) in zip(ticks, desc[ticks])
                    ])
            elif isinstance(axis, matplotlib.axis.YAxis):
                label_handles.append([
                    this_desc.y_tick_label(
                        this_y,
                        size,
                        offset=position,
                        linewidth=linewidth,
                        ax=axis.axes,
                    )
                    for (this_y, this_desc) in zip(ticks, desc[ticks])
                ])
            else:
                raise TypeError("expected axis to be XAxis or YAxis instance")
    else:
        # vanilla matplotlib-based
        # need to ensure the minor ticks have some length
        axis.set_tick_params(length=matplotlib.rcParams['xtick.minor.size'],
                which="minor")
        label_handles = axis.set_ticklabels(
            desc,
            verticalalignment="center",
            horizontalalignment=horizontalalignment,
            minor=True,
        )
        if isinstance(axis, matplotlib.axis.XAxis):
            plt.setp(
                axis.get_ticklabels(minor=True),
                rotation=60,
                ha="right",
                rotation_mode="anchor",
            )

    return label_handles
