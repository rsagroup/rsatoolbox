#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot showing an RDMs object
"""

import numpy as np
import matplotlib.pyplot as plt
from pyrsa import vis
from pyrsa.rdm import rank_transform
from pyrsa.vis.colors import rdm_colormap


def show_rdm(rdm, do_rank_transform=False, pattern_descriptor=None,
             cmap=None, rdm_descriptor=None, dpi=300, filename=None,
             show_colorbar=False, **kwarg):
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
    dpi : int
        dots per inch (determines visual resolution of plots)
    filename : str
        relative path to which the plot will be saved
        (if None: do not save plot)
    show_colorbar : bool
        whether to display a colorbar next to each RDM

    """
    plt.figure(dpi=dpi)
    if cmap is None:
        cmap = rdm_colormap()
    if do_rank_transform:
        rdm = rank_transform(rdm)
    rdm_mat = rdm.get_matrices()
    alpha = 1-np.eye(rdm.n_cond)
    if rdm.n_rdm > 1:
        m = np.ceil(np.sqrt(rdm.n_rdm+1))
        n = np.ceil((1 + rdm.n_rdm) / m)
        for idx in range(rdm.n_rdm):
            plt.subplot(n, m, idx + 1)
            image = plt.imshow(rdm_mat[idx], cmap=cmap, alpha=alpha)
            _add_descriptor_labels(rdm, pattern_descriptor, **kwarg)
            if rdm_descriptor in rdm.rdm_descriptors:
                plt.title(rdm.rdm_descriptors[rdm_descriptor][idx])
            elif isinstance(rdm_descriptor, str):
                plt.title(rdm_descriptor)
            if show_colorbar:
                plt.colorbar(image)
        plt.subplot(n, m, n * m)
        image = plt.imshow(np.mean(rdm_mat, axis=0), cmap=cmap, alpha=alpha)
        _add_descriptor_labels(rdm, pattern_descriptor, **kwarg)
        plt.title('Average')
        if show_colorbar:
            plt.colorbar(image)
    elif rdm.n_rdm == 1:
        image = plt.imshow(rdm_mat[0], cmap=cmap, alpha=alpha)
        _add_descriptor_labels(rdm, pattern_descriptor, **kwarg)
        if rdm_descriptor in rdm.rdm_descriptors:
            plt.title(rdm.rdm_descriptors[rdm_descriptor][0])
        elif isinstance(rdm_descriptor, str):
            plt.title(rdm_descriptor)
        if show_colorbar:
            plt.colorbar(image)
    if isinstance(filename, str):
        fig1 = plt.gcf()
        fig1.savefig(filename, bbox_inches='tight')
    plt.show()


def _add_descriptor_labels(rdm, descriptor, num_pattern_groups=1, size=.5, offset=7,
        linewidth=None, ax=None, axis="xy", gridlines=None):
    """ adds a descriptor as ticklabels """
    if ax is None:
        ax = plt.gca()
    if linewidth is None:
        linewidth = .5
    if descriptor is not None:
        desc = rdm.pattern_descriptors[descriptor]
        ax.set_xticks(np.arange(rdm.n_cond), minor=True)
        ax.set_yticks(np.arange(rdm.n_cond), minor=True)
        if isinstance(desc[0], vis.Icon):
            # image labels
            if linewidth > 0:
                # TODO - axis specific
                ax.yaxis.set_tick_params(length=0, which='minor')
                ax.xaxis.set_tick_params(length=0, which='minor')
            # TODO - work out sizing from pixel size transform
            for group_ind in range(num_pattern_groups, 0, -1):
                position = offset * group_ind
                ticks = np.arange(group_ind-1, rdm.n_cond, num_pattern_groups)
                # TODO - let's not plot rows and columns each time
                [this_desc.x_tick_label(this_x, size, offset=position,
                    linewidth=linewidth) for (this_x, this_desc) in
                        zip(ticks, desc[ticks])]
                [this_desc.y_tick_label(this_y, size, offset=position,
                    linewidth=linewidth) for (this_y, this_desc) in
                        zip(ticks, desc[ticks])]
            # grid the groups
            if not np.any(gridlines):
                gridlines = np.arange(num_pattern_groups-.5, rdm.n_cond+.5,
                        num_pattern_groups)
        else:
            # vanilla
            ax.set_xticklabels(
                desc,
                {'fontsize': 'xx-small',
                 'fontweight': 'normal',
                 'verticalalignment': 'center',
                 'horizontalalignment': 'center'}, minor=True)
            ax.set_yticks(np.arange(rdm.n_cond))
            ax.set_yticklabels(
                desc,
                {'fontsize': 'xx-small',
                 'fontweight': 'normal',
                 'verticalalignment': 'center',
                 'horizontalalignment': 'right'}, minor=True)
            # rotate if the string is over some reasonable limit
            # if isinstance(desc[0], str) and max([len(this_desc) for this_desc in desc]) > 5:
                # ax.tick_params(axis='x', rotation=45, ha='right')
        plt.ylim(rdm.n_cond - 0.5, -0.5)
        plt.xlim(-0.5, rdm.n_cond - 0.5)
        ax.set_xticks(gridlines)
        ax.set_yticks(gridlines)
        ax.yaxis.set_tick_params(length=0)
        ax.xaxis.set_tick_params(length=0)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(axis='both', color='w')
        plt.setp(ax.get_xticklabels(minor=True), rotation=90, ha="right",
                rotation_mode="anchor")
    else:
        ax.set_xticks([])
        ax.set_yticks([])
