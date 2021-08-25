#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot showing an RDMs object
"""

from __future__ import annotations
import collections
from typing import TYPE_CHECKING, Union, Tuple
import pkg_resources
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import rsatoolbox.rdm
from rsatoolbox import vis
from rsatoolbox.vis.colors import rdm_colormap
if TYPE_CHECKING:
    import numpy.typing as npt
    import pathlib
    from matplotlib.axes._axes import Axes

RDM_STYLE = pkg_resources.resource_filename('rsatoolbox.vis', 'rdm.mplstyle')


def show_rdm(
    rdm: rsatoolbox.rdm.RDMs,
    pattern_descriptor: str = None,
    cmap: Union[str, matplotlib.colors.Colormap] = None,
    rdm_descriptor: str = None,
    n_column: int = None,
    n_row: int = None,
    show_colorbar: str = None,
    gridlines: npt.ArrayLike = None,
    num_pattern_groups: int = None,
    figsize: Tuple[float, float] = None,
    nanmask: npt.ArrayLike = None,
    style: Union[str, pathlib.Path] = RDM_STYLE,
    vmin: float = None,
    vmax: float = None,
    icon_spacing: float = 1.0,
    linewidth: float = 0.5,
) -> Tuple[
    matplotlib.figure.Figure, npt.ArrayLike, collections.defaultdict
]:
    """show_rdm. Heatmap figure for RDMs instance, with one panel per RDM.

    Args:
        rdm (rsatoolbox.rdm.RDMs): RDMs object to be plotted.
        pattern_descriptor (str): Key into rdm.pattern_descriptors to use for axis
            labels.
        cmap (Union[str, matplotlib.colors.Colormap]): colormap to be used (by
            plt.imshow internally). By default we use rdm_colormap.
        rdm_descriptor (str): Key for rdm_descriptor to use as panel title, or
            str for direct labeling.
        n_column (int): Number of columns in subplot arrangement.
        n_row (int): Number of rows in subplot arrangement.
        show_colorbar (str): Set to 'panel' or 'colorbar' to display a colorbar. If
            'panel' a colorbar is added next to each RDM. If 'figure' a shared colorbar
            (and scale) is used across panels.
        gridlines (npt.ArrayLike): Set to add gridlines at these positions. If
            num_pattern_groups is defined this is used to infer gridlines.
        num_pattern_groups (int): Number of rows/columns for any image labels. Also
            determines gridlines frequency by default (so e.g., num_pattern_groups=3
            with results in gridlines every 3 rows/columns).
        figsize (Tuple[float, float]): mpl.Figure argument. By default we
            auto-scale to achieve a figure that fits on a standard A4 / US Letter page
            in portrait orientation.
        nanmask (npt.ArrayLike): boolean mask defining RDM elements to suppress
            (by default, the diagonals).
        style (Union[str, pathlib.Path]): Path to mplstyle file that controls
            various figure aesthetics (default rsatoolbox/vis/rdm.mplstyle).
        vmin (float): Minimum intensity for colorbar mapping. matplotlib imshow
            argument.
        vmax (float): Maximum intensity for colorbar mapping. matplotlib imshow
            argument.
        icon_spacing (float): control spacing of image labels - 1. means no gap (the
            default), 1.1 means pad 10%, .9 means overlap 10% etc.
        linewidth (float): Width of connecting lines from icon labels (if used) to axis
            margin.  The default is 0.5 - set to 0. to disable the lines.

    Returns:
        Tuple[matplotlib.figure.Figure, npt.ArrayLike, collections.defaultdict]:

        Tuple of

            - Handle to created figure.
            - Subplot axis handles from plt.subplots.
            - Nested dict containing handles to all other plotted
              objects (icon labels, colorbars, etc). The keys at the first level are the
              axis and figure handles.

    """

    if show_colorbar and not show_colorbar in ("panel", "figure"):
        raise ValueError(
            f"show_colorbar can be None, panel or figure, got: {show_colorbar}"
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
                    return_handles[panel]["image"] = show_rdm_panel(
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
                if show_colorbar == "panel":
                    # needs to happen before labels because it resizes the axis
                    return_handles[panel]["colorbar"] = _rdm_colorbar(
                        mappable=return_handles[panel]["image"],
                        fig=fig,
                        ax=panel,
                        title=rdm.dissimilarity_measure,
                    )
                if col_ind == 0 and pattern_descriptor:
                    return_handles[panel]["y_labels"] = add_descriptor_y_labels(
                        rdm,
                        pattern_descriptor,
                        ax=panel,
                        num_pattern_groups=num_pattern_groups,
                        icon_spacing=icon_spacing,
                        linewidth=linewidth,
                    )
                if row_ind == 0 and pattern_descriptor:
                    return_handles[panel]["x_labels"] = add_descriptor_x_labels(
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
            return_handles[fig]["colorbar"] = _rdm_colorbar(
                mappable=return_handles[ax_array[0][0]]["image"],
                fig=fig,
                ax=cbax_parent,
                title=rdm.dissimilarity_measure,
            )
            cbax_pos = return_handles[fig]["colorbar"].ax.get_position()
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
            return_handles[fig]["colorbar"].ax.set_position(
                [x0, y0, cbax_pos.width, cbax_pos.height]
            )

    return fig, ax_array, return_handles


def _rdm_colorbar(
    mappable: matplotlib.cm.ScalarMappable = None,
    fig: matplotlib.figure.Figure = None,
    ax: Axes = None,
    title: str = None,
) -> matplotlib.colorbar.Colorbar:
    """_rdm_colorbar. Add vertically-oriented, small colorbar to rdm figure. Used
    internally by show_rdm.

    Args:
        mappable (matplotlib.cm.ScalarMappable): Typically plt.imshow instance.
        fig (matplotlib.figure.Figure): Matplotlib figure handle.
        ax (matplotlib.axes._axes.Axes): Matplotlib axis handle. plt.gca() by default.
        title (str): Title string for the colorbar (positioned top, left aligned).

    Returns:
        matplotlib.colorbar.Colorbar: Matplotlib handle.
    """
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
    rdm: rsatoolbox.rdm.RDMs,
    ax: Axes = None,
    cmap: Union[str, matplotlib.colors.Colormap] = None,
    nanmask: npt.ArrayLike = None,
    rdm_descriptor: str = None,
    gridlines: npt.ArrayLike = None,
    vmin: float = None,
    vmax: float = None,
) -> matplotlib.image.AxesImage:
    """show_rdm_panel. Add RDM heatmap to the axis ax.

    Args:
        rdm (rsatoolbox.rdm.RDMs): RDMs object to be plotted (n_rdm must be 1).
        ax (matplotlib.axes._axes.Axes): Matplotlib axis handle. plt.gca() by default.
        cmap (Union[str, matplotlib.colors.Colormap]): colormap to be used (by
            plt.imshow internally). By default we use rdm_colormap.
        nanmask (npt.ArrayLike): boolean mask defining RDM elements to suppress
            (by default, the diagonals).
        rdm_descriptor (str): Key for rdm_descriptor to use as panel title, or
            str for direct labeling.
        gridlines (npt.ArrayLike): Set to add gridlines at these positions.
        vmin (float): Minimum intensity for colorbar mapping. matplotlib imshow
            argument.
        vmax (float): Maximum intensity for colorbar mapping. matplotlib imshow
            argument.

    Returns:
        matplotlib.image.AxesImage: Matplotlib handle.
    """
    if rdm.n_rdm > 1:
        raise ValueError("expected single rdm - use show_rdm for multi-panel figures")
    if ax is None:
        ax = plt.gca()
    if cmap is None:
        cmap = rdm_colormap()
    if nanmask is None:
        nanmask = np.eye(rdm.n_cond, dtype=bool)
    if not np.any(gridlines):
        gridlines = []
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
    else:
        ax.set_title(rdm_descriptor)
    return image


def add_descriptor_x_labels(
    rdm: rsatoolbox.rdm.RDMs,
    pattern_descriptor: str,
    ax: Axes = None,
    num_pattern_groups: int = None,
    icon_spacing: float = 1.0,
    linewidth: float = 0.5,
) -> list:
    """add_descriptor_x_labels. Add labels to the X axis in ax by accessing the
    rdm.pattern_descriptors dict with the pattern_descriptor key.

    Args:
        rdm (rsatoolbox.rdm.RDMs): RDMs instance to annotate.
        pattern_descriptor (str): dict key for the rdm.pattern_descriptors dict.
        ax (matplotlib.axes._axes.Axes): Matplotlib axis handle. plt.gca() by default.
        num_pattern_groups (int): Number of rows/columns for any image labels.
        icon_spacing (float): control spacing of image labels - 1. means no gap (the
            default), 1.1 means pad 10%, .9 means overlap 10% etc.
        linewidth (float): Width of connecting lines from icon labels (if used) to axis
            margin.  The default is 0.5 - set to 0. to disable the lines.

    Returns:
        list: Tick label handles.
    """
    if ax is None:
        ax = plt.gca()
    return _add_descriptor_labels(
        rdm,
        pattern_descriptor,
        "x_tick_label",
        ax.xaxis,
        num_pattern_groups=num_pattern_groups,
        icon_spacing=icon_spacing,
        linewidth=linewidth,
        horizontalalignment="center",
    )


def add_descriptor_y_labels(
    rdm: rsatoolbox.rdm.RDMs,
    pattern_descriptor: str,
    ax: Axes = None,
    num_pattern_groups: int = None,
    icon_spacing: float = 1.0,
    linewidth: float = 0.5,
) -> list:
    """add_descriptor_y_labels. Add labels to the Y axis in ax by accessing the
    rdm.pattern_descriptors dict with the pattern_descriptor key.

    Args:
        rdm (rsatoolbox.rdm.RDMs): RDMs instance to annotate.
        pattern_descriptor (str): dict key for the rdm.pattern_descriptors dict.
        ax (matplotlib.axes._axes.Axes): Matplotlib axis handle. plt.gca() by default.
        num_pattern_groups (int): Number of rows/columns for any image labels.
        icon_spacing (float): control spacing of image labels - 1. means no gap (the
            default), 1.1 means pad 10%, .9 means overlap 10% etc.
        linewidth (float): Width of connecting lines from icon labels (if used) to axis
            margin.  The default is 0.5 - set to 0. to disable the lines.

    Returns:
        list: Tick label handles.
    """
    if ax is None:
        ax = plt.gca()
    return _add_descriptor_labels(
        rdm,
        pattern_descriptor,
        "y_tick_label",
        ax.yaxis,
        num_pattern_groups=num_pattern_groups,
        icon_spacing=icon_spacing,
        linewidth=linewidth,
        horizontalalignment="right",
    )


def _add_descriptor_labels(
    rdm: rsatoolbox.rdm.RDMs,
    pattern_descriptor: str,
    icon_method: str,
    axis: Union[matplotlib.axis.XAxis, matplotlib.axis.YAxis],
    num_pattern_groups: int = None,
    icon_spacing: float = 1.0,
    linewidth: float = 0.5,
    horizontalalignment: str = "center",
) -> list:
    """_add_descriptor_labels. Used internally by add_descriptor_y_labels and
    add_descriptor_x_labels.

    Args:
        rdm (rsatoolbox.rdm.RDMs): RDMs instance to annotate.
        pattern_descriptor (str): dict key for the rdm.pattern_descriptors dict.
        icon_method (str): method to access on Icon instances (typically y_tick_label or
            x_tick_label).
        axis (Union[matplotlib.axis.XAxis, matplotlib.axis.YAxis]): Axis to add
            tick labels to.
        num_pattern_groups (int): Number of rows/columns for any image labels.
        icon_spacing (float): control spacing of image labels - 1. means no gap (the
            default), 1.1 means pad 10%, .9 means overlap 10% etc.
        linewidth (float): Width of connecting lines from icon labels (if used) to axis
            margin.  The default is 0.5 - set to 0. to disable the lines.
        horizontalalignment (str): Horizontal alignment of text tick labels.

    Returns:
        list: Tick label handles.
    """
    descriptor_arr = np.asarray(rdm.pattern_descriptors[pattern_descriptor])
    if isinstance(descriptor_arr[0], vis.Icon):
        return _add_descriptor_icons(
            descriptor_arr,
            icon_method,
            n_cond=rdm.n_cond,
            ax=axis.axes,
            icon_spacing=icon_spacing,
            num_pattern_groups=num_pattern_groups,
            linewidth=linewidth,
        )
    is_x_axis = "x" in icon_method
    return _add_descriptor_text(
        descriptor_arr,
        axis=axis,
        horizontalalignment=horizontalalignment,
        is_x_axis=is_x_axis,
    )


def _add_descriptor_text(
    descriptor_arr: npt.ArrayLike,
    axis: Union[matplotlib.axis.XAxis, matplotlib.axis.YAxis],
    horizontalalignment: str = "center",
    is_x_axis: bool = False,
) -> list:
    """_add_descriptor_text. Used internally by _add_descriptor_labels to add vanilla
    Matplotlib-based text labels to the X or Y axis.

    Args:
        descriptor_arr (npt.ArrayLike): np.Array-like version of the labels.
        axis (Union[matplotlib.axis.XAxis, matplotlib.axis.YAxis]): handle for
            the relevant axis (ax.xaxis or ax.yaxis).
        horizontalalignment (str): Horizontal alignment of text tick labels.
        is_x_axis (bool): If set, rotate the text labels 60 degrees to reduce overlap on
            the X axis.

    Returns:
        list: Tick label handles.
    """
    # vanilla matplotlib-based
    # need to ensure the minor ticks have some length
    axis.set_tick_params(length=matplotlib.rcParams["xtick.minor.size"], which="minor")
    label_handles = axis.set_ticklabels(
        descriptor_arr,
        verticalalignment="center",
        horizontalalignment=horizontalalignment,
        minor=True,
    )
    if is_x_axis:
        plt.setp(
            axis.get_ticklabels(minor=True),
            rotation=60,
            ha="right",
            rotation_mode="anchor",
        )
    return label_handles


def _add_descriptor_icons(
    descriptor_arr: npt.ArrayLike,
    icon_method: str,
    n_cond: int,
    ax: Axes = None,
    num_pattern_groups: int = None,
    icon_spacing: float = 1.0,
    linewidth: float = 0.5,
) -> list:
    """_add_descriptor_icons. Used internally by _add_descriptor_labels to add
    Icon-based labels to the X or Y axis.

    Args:
        descriptor_arr (npt.ArrayLike): np.Array-like version of the labels.
        icon_method (str): method to access on Icon instances (typically y_tick_label or
            x_tick_label).
        n_cond (int): Number of conditions in the RDM (usually from RDMs.n_cond).
        ax (matplotlib.axes._axes.Axes): Matplotlib axis handle.
        num_pattern_groups (int): Number of rows/columns for any image labels.
        icon_spacing (float): control spacing of image labels - 1. means no gap (the
            default), 1.1 means pad 10%, .9 means overlap 10% etc.
        linewidth (float): Width of connecting lines from icon labels (if used) to axis
            margin.  The default is 0.5 - set to 0. to disable the lines.

    Returns:
        list: Tick label handles.
    """
    # annotated labels with Icon
    n_to_fit = np.ceil(n_cond / num_pattern_groups)
    # work out sizing of icons
    im_max_pix = 20.
    if descriptor_arr[0].final_image:
        # size by image
        im_width_pix = max(this_desc.final_image.width for this_desc in descriptor_arr)
        im_height_pix = max(this_desc.final_image.height for this_desc in descriptor_arr)
        im_max_pix = max(im_width_pix, im_height_pix) * icon_spacing
    ax.figure.canvas.draw()
    extent = ax.get_window_extent(ax.figure.canvas.get_renderer())
    ax_size_pix = max((extent.width, extent.height))
    size = (ax_size_pix / n_to_fit) / im_max_pix
    # from proportion of original size to figure pixels
    offset = im_max_pix * size
    label_handles = []
    for group_ind in range(num_pattern_groups - 1, -1, -1):
        position = offset * 0.2 + offset * group_ind
        ticks = np.arange(group_ind, n_cond, num_pattern_groups)
        label_handles.append(
            [
                getattr(this_desc, icon_method)(
                    this_x, size, offset=position, linewidth=linewidth, ax=ax,
                )
                for (this_x, this_desc) in zip(ticks, descriptor_arr[ticks])
            ]
        )
    return label_handles
