#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
icon object which can be plotted into an axis
"""

import matplotlib.pyplot as plt
import numpy as np
import PIL
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, DrawingArea


class Icon:
    """ Icon object, i.e. an object which can be plotted into an axis or as
    an axis label.

    Args:
        image
        string
        col
        border_color
    """

    def __init__(self, image=None, string=None, col=None, border_color=None):
        self.image = image
        self.string = string
        self.col = col
        self.border_color = border_color

    def plot(self, x, y, ax=None, size=None, cmap=None):
        """
        plots the icon into an axis

        Args
        ax : TYPE
            DESCRIPTION.
        x : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.
        scale : float
            scale of the icon proportional to the axis shorter axis
            (default = 0.05)
        """
        if ax is None:
            ax = plt.gca()
        if size is None:
            imagebox = OffsetImage(self.image, cmap=cmap, zoom=1)
        else:
            imagebox = OffsetImage(self.image, cmap=cmap, zoom=size)
        if self.border_color is None:
            bboxprops = None
            frameon = False
        else:
            bboxprops = dict(boxstyle='round', color=self.border_color)
            frameon = True
        ab = AnnotationBbox(
            imagebox, (x, y),  frameon=frameon,
            bboxprops=bboxprops)
        ax.add_artist(ab)
        if self.border_color is not None:
            pass

    def x_tick_label(self, x, size, offset=7, ax=None):
        """
        uses the icon as a ticklabel at location x

        Parameters
        ----------

        Returns
        -------
        None.

        """
        if ax is None:
            ax = plt.gca()
        imagebox = OffsetImage(self.image, zoom=size)
        ab = AnnotationBbox(imagebox, (x, 0),
                            xybox=(0, -offset),
                            xycoords=('data', 'axes fraction'),
                            box_alignment=(.5, 1),
                            boxcoords='offset points',
                            bboxprops={'edgecolor': 'none'},
                            arrowprops={'arrowstyle': '-',
                                        'shrinkA': 0,
                                        'shrinkB': 1
                                        })
        ax.add_artist(ab)

    def y_tick_label(self, y, size, offset=7, ax=None):
        """
        uses the icon as a ticklabel at location y

        Parameters
        ----------

        Returns
        -------
        None.

        """
        if ax is None:
            ax = plt.gca()
        imagebox = OffsetImage(self.image, zoom=size)
        ab = AnnotationBbox(imagebox, (0, y),
                            xybox=(-offset, 0),
                            xycoords=('axes fraction', 'data'),
                            box_alignment=(1, .5),
                            boxcoords='offset points',
                            bboxprops={'edgecolor': 'none'},
                            arrowprops={'arrowstyle': '-',
                                        'shrinkA': 0,
                                        'shrinkB': 1
                                        })
        ax.add_artist(ab)

test_im = PIL.Image.fromarray(255 * np.random.rand(100, 100))
ic = Icon(image=test_im)
ax = plt.subplot(1, 1, 1)
ic.plot(0.5, 0.5, ax=ax)
ic2 = Icon(image=test_im, border_color='black')
ic2.plot(0.8, 0.2, ax=ax, size=0.4)
ic2.x_tick_label(0.5, 0.15, offset=7)
ic2.y_tick_label(0.5, 0.25, offset=7)
