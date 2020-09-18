#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Definition of pyrsa's colors

@author: iancharest
"""

import numpy as np
from skimage.color import rgb2hsv, hsv2rgb
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.interpolate import interp1d


def color_scale(n_cols, anchor_cols=None, monitor=False):
    """ linearly interpolates between a set of given
    anchor colours to give n_cols and displays them
    if monitor is set

    Args:
        n_cols (int): number of colors for the colormap
        anchor_cols (numpy.ndarray, optional): what color space to
            interpolate. Defaults to None.
        monitor (boolean, optional): quick visualisation of the
            resulting colormap. Defaults to False.

    Returns:
        numpy.ndarray: n_cols x 3 RGB array.

    """

    if anchor_cols is None:
        # if no anchor_cols provided, use red to blue
        anchor_cols = np.array([[1, 0, 0], [0, 0, 1]])

    # define color scale
    n_anchors = anchor_cols.shape[0]

    # simple 1D interpolation
    fn = interp1d(
        range(n_anchors),
        anchor_cols.T,
    )
    cols = fn(np.linspace(0, n_anchors - 1, n_cols)).T

    # optional visuals
    if monitor:
        reshaped_cols = cols.reshape((n_cols, 1, 3))
        width = int(n_cols / 2)
        mapping = np.tile(reshaped_cols, (width, 1))
        plt.imshow(mapping)
        plt.show()

    return cols


def rdm_colormap(n_cols=256, monitor=None):
    """this function provides a convenient colormap for visualizing
    dissimilarity matrices. it goes from blue to yellow and has grey for
    intermediate values.

    Args:
        n_cols (int, optional): precision of the colormap.
        Defaults to 256.

    Returns:
        [matplotlib ListedColormap]: this matplotlib color object can be
        used as a cmap in any plot.

    Example:
        .. code-block:: python

            import numpy as np
            import matplotlib.pyplot as plt
            from pyrsa.vis.colors import rdm_colormap
            plt.imshow(np.random.rand(10,10),cmap=rdm_colormap())
            plt.colorbar()
            plt.show()

    (ported from Niko Kriegeskorte's RDMcolormap.m)
    """

    # blue-cyan-gray-red-yellow with increasing V (BCGRYincV)
    anchor_cols = np.array([
        [0, 0, 1],
        [0, 1, 1],
        [.5, .5, .5],
        [1, 0, 0],
        [1, 1, 0],
    ])

    # skimage rgb2hsv is intended for 3d images (RGB)
    # here we add a new axis to our 2d anchorCols to satisfy
    # skimage, and then squeeze
    anchor_cols_hsv = rgb2hsv(anchor_cols[np.newaxis, :]).squeeze()

    inc_v_weight = 1
    anchor_cols_hsv[:, 2] = (1 - inc_v_weight) * anchor_cols_hsv[:, 2] + \
        inc_v_weight * np.linspace(0.5, 1, anchor_cols.shape[0]).T

    # anchorCols = brightness(anchorCols)
    anchor_cols = hsv2rgb(anchor_cols_hsv[np.newaxis, :]).squeeze()

    cols = color_scale(n_cols, anchor_cols, monitor)

    return ListedColormap(cols)
