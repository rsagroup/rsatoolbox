#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
icon object which can be plotted into an axis
"""

import matplotlib.pyplot as plt
import numpy as np
import PIL
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


class Icon:
    """ Icon object, i.e. an object which can be plotted into an axis or as
    an axis label.

    Args:
        image
        string
        col
        border_color
    """

    def __init__(self, image=None, string=None, col=None, border_color=None,
                 cmap=None, border_type='pad', border_width=2):
        self.image = image
        self.string = string
        self.col = col
        self.border_color = border_color
        self.border_type = border_type
        self.border_width = border_width
        self.cmap = cmap
        self.recompute_final_image()

    def set(self, image=None, string=None, col=None, border_color=None):
        if image is not None:
            self.image = image
        if string is not None:
            self.string = string
        if col is not None:
            self.col = col
        if border_color is not None:
            self.border_color = border_color
        self.recompute_final_image()

    def recompute_final_image(self):
        im = self.image
        if self.border_color is not None:
            if self.border_type == 'pad':
                if isinstance(im, np.ndarray):
                    im = np.pad(im, self.border_width,
                                padder=self.border_color)
                elif isinstance(im, PIL.Image.Image):
                    im = PIL.ImageOps.expand(
                        test_im,
                        border=self.border_width,
                        fill=self.border_color)
        self.final_image = im

    def plot(self, x, y, ax=None, size=None):
        """
        plots the icon into an axis

        Args
        ax : TYPE
            DESCRIPTION.
        x : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.
        size : float
            size of the icon 
        """
        if ax is None:
            ax = plt.gca()
        if size is None:
            imagebox = OffsetImage(self.final_image, cmap=self.cmap, zoom=1)
        else:
            imagebox = OffsetImage(self.final_image, cmap=self.cmap, zoom=size)
        ab = AnnotationBbox(
            imagebox, (x, y),  frameon=False,
            pad=0)
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
        imagebox = OffsetImage(self.final_image, zoom=size)
        ab = AnnotationBbox(
            imagebox, (x, 0),
            xybox=(0, -offset),
            xycoords=('data', 'axes fraction'),
            box_alignment=(.5, 1),
            boxcoords='offset points',
            bboxprops={'edgecolor': 'none'},
            arrowprops={
                'arrowstyle': '-',
                'shrinkA': 0,
                'shrinkB': 1
                },
            pad=0.1)
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
        imagebox = OffsetImage(self.final_image, zoom=size)
        ab = AnnotationBbox(
            imagebox, (0, y),
            xybox=(-offset, 0),
            xycoords=('axes fraction', 'data'),
            box_alignment=(1, .5),
            boxcoords='offset points',
            bboxprops={'edgecolor': 'none'},
            arrowprops={
                'arrowstyle': '-',
                'shrinkA': 0,
                'shrinkB': 1
                },
            pad=0.1)
        ax.add_artist(ab)


test_im = PIL.Image.fromarray(255 * np.random.rand(50, 100))
ic = Icon(image=test_im)
ax = plt.subplot(1, 1, 1)
ic.plot(0.5, 0.5, ax=ax)
ic2 = Icon(image=test_im, border_color='black', border_width=15)
ic2.plot(0.8, 0.2, ax=ax, size=0.4)
ic2.x_tick_label(0.5, 0.15, offset=7)
ic2.y_tick_label(0.5, 0.25, offset=7)
