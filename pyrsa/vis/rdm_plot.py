#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot showing an RDMs object
"""

import os.path
import inspect
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pyrsa import vis
from pyrsa.rdm import rank_transform
from pyrsa.vis.colors import rdm_colormap

MODULE_DIR = os.path.dirname(inspect.getfile(inspect.currentframe()))

RDM_STYLE = os.path.join(MODULE_DIR, 'rdm.mplstyle')

def show_rdm(rdm, do_rank_transform=False, pattern_descriptor=None, cmap=None,
        rdm_descriptor=None, n_column=None, n_row=None, show_colorbar=None,
        gridlines=[], num_pattern_groups=None, figsize=None, nanmask=None,
        style=RDM_STYLE, **kwarg):
    """shows an rdm object

    Parameters
    ----------
    rdm : pyrsa.rdm.RDMs
        RDMs object to be plotted
    do_rank_transform : bool
        whether we should do a rank transform before plotting
    pattern_descriptor : String
        name of a pattern descriptor which will be used as an axis label
    cmap : color map
        colormap or identifier for a colormap to be used
        conventions as for matplotlib colormaps
    rdm_descriptor : String
        name of a rdm descriptor which will be used as a title per RDM
    show_colorbar : str
        whether to display a colorbar. If 'panel' a colorbar is added next to
        each RDM. If 'figure' a shared colorbar (and scale) is used across panels.
    nanmask : array
    """


    n_panel = rdm.n_rdm
    if show_colorbar == 'figure':
        n_panel += 1
    if n_column is None and n_row is None:
        n_column = np.ceil(np.sqrt(n_panel))
        n_row = np.ceil((1 + n_panel) / n_column)
    if n_row is None:
        n_row = np.ceil(n_panel / n_column)
    if n_column is None:
        n_column = np.ceil(n_panel / n_row)
    if (n_column * n_row) < rdm.n_rdm:
        raise ValueError(
                f'invalid n_row*n_column specification for {n_panel} rdms: {n_row}*{n_column}')
    if figsize is None:
        # scale with number of RDMs, up to a point (the intersection of A4 and us
        # letter)
        figsize = (min(2 * n_column, 8.3), min(2 * n_row, 11))
    if nanmask is None:
        nanmask = np.eye(rdm.n_cond, dtype=bool)
    if cmap is None:
        cmap = rdm_colormap()
    if do_rank_transform:
        rdm = rank_transform(rdm)
    if not np.any(gridlines) and num_pattern_groups:
        # grid by pattern groups if they exist and explicit grid setting does not
        gridlines = np.arange(num_pattern_groups-.5, rdm.n_cond+.5,
                num_pattern_groups)
    # we don't necessarily have the same number of RDMs as panels, so need to stop the
    # loop when we've plotted all the RDMs
    rdms_gen = (this_rdm for this_rdm in rdm)
    with plt.style.context(style):
        fig, ax_array = plt.subplots(nrows=int(n_row), ncols=int(n_column), sharex=True,
                sharey=True, squeeze=False, figsize=figsize)
        # reverse panel order so unfilled rows are at top instead of bottom
        ax_array = ax_array[::-1]
        for row_ind, row in enumerate(ax_array):
            for col_ind, panel in enumerate(row):
                try:
                    image = show_rdm_panel(next(rdms_gen), ax=panel, cmap=cmap,
                            nanmask=nanmask,
                            rdm_descriptor=rdm_descriptor, gridlines=gridlines, **kwarg)
                except StopIteration:
                    # hide empty panels
                    panel.set_visible(False)
                    continue
                except:
                    raise
                if col_ind == 0:
                    _add_descriptor_y_labels(rdm, pattern_descriptor, ax=panel,
                            num_pattern_groups=num_pattern_groups, **kwarg)
                if row_ind == 0:
                    _add_descriptor_x_labels(rdm, pattern_descriptor, ax=panel,
                            num_pattern_groups=num_pattern_groups, **kwarg)
                if show_colorbar == 'panel':
                    cb = fig.colorbar(mappable=image, ax=panel, shrink=.25, aspect=5,
                            anchor=(0., 1.),
                            ticks=matplotlib.ticker.LinearLocator(numticks=3))
                    cb.ax.set_title(rdm.dissimilarity_measure, loc='left',
                            fontdict=dict(fontweight='normal'))
    # TODO - handle show_colorbar=='figure'
    return fig

def show_rdm_panel(rdm, ax=None, cmap=None, nanmask=None, rdm_descriptor=None,
        gridlines=[], **kwarg):
    if ax is None:
        ax = plt.gca()
    if rdm.n_rdm > 1:
        raise ValueError('expected single rdm - use show_rdm for multi-panel figures')
    rdmat = rdm.get_matrices()[0,:,:]
    if np.any(nanmask):
        rdmat[nanmask] = np.nan
    image = ax.imshow(rdmat, cmap=cmap)
    ax.set_xlim(-0.5, rdm.n_cond - 0.5)
    ax.set_ylim(rdm.n_cond - 0.5, -0.5)
    ax.xaxis.set_ticks(gridlines)
    ax.yaxis.set_ticks(gridlines)
    #ax.xaxis.set_tick_params(length=0)
    #ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.xaxis.set_ticks(np.arange(rdm.n_cond), minor=True)
    ax.yaxis.set_ticks(np.arange(rdm.n_cond), minor=True)
    if rdm_descriptor in rdm.rdm_descriptors:
        ax.set_title(rdm.rdm_descriptors[rdm_descriptor][0])
    elif isinstance(rdm_descriptor, str):
        ax.set_title(rdm_descriptor)
    return image

def _add_descriptor_x_labels(rdm, descriptor, ax=None, **kwarg):
    if ax is None:
        ax = plt.gca()
    _add_descriptor_labels(rdm, descriptor, axis=ax.xaxis, other_axis=ax.yaxis,
            horizontalalignment='center', **kwarg)

def _add_descriptor_y_labels(rdm, descriptor, ax=None, **kwarg):
    if ax is None:
        ax = plt.gca()
    _add_descriptor_labels(rdm, descriptor, axis=ax.yaxis, other_axis=ax.xaxis,
            horizontalalignment='right', **kwarg)

def _add_descriptor_labels(rdm, descriptor, axis, other_axis, num_pattern_groups=None,
        size=.5, offset=7, linewidth=None, horizontalalignment='center'):
    """ adds a descriptor as ticklabels to the axis (XAxis or YAxis instance)"""
    if linewidth is None:
        linewidth = .5
    if num_pattern_groups is None or num_pattern_groups==0:
        num_pattern_groups = 1
    if descriptor is not None:
        desc = rdm.pattern_descriptors[descriptor]
        if isinstance(desc[0], vis.Icon):
            # annotated labels with Icon
            if linewidth > 0:
                axis.set_tick_params(length=0, which='minor')
            # TODO - work out sizing from pixel size transform
            for group_ind in range(num_pattern_groups, 0, -1):
                position = offset * group_ind
                ticks = np.arange(group_ind-1, rdm.n_cond, num_pattern_groups)
                if isinstance(axis, matplotlib.axis.XAxis):
                    [this_desc.x_tick_label(this_x, size, offset=position,
                        linewidth=linewidth, ax=axis.axes) for (this_x, this_desc) in
                            zip(ticks, desc[ticks])]
                elif isinstance(axis, matplotlib.axis.YAxis):
                    [this_desc.y_tick_label(this_y, size, offset=position,
                        linewidth=linewidth, ax=axis.axes) for (this_y, this_desc) in
                            zip(ticks, desc[ticks])]
                else:
                    raise TypeError("expected axis to be XAxis or YAxis instance")
        else:
            # vanilla matplotlib-based
            axis.set_ticklabels(
                    desc,
                    verticalalignment='center',
                    horizontalalignment=horizontalalignment,
                    minor=True)
            if isinstance(axis, matplotlib.axis.XAxis):
                plt.setp(axis.get_ticklabels(minor=True), rotation=60, ha="right",
                    rotation_mode="anchor")
