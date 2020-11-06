#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
icon object which can be plotted into an axis
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, DrawingArea
import numpy as np
import PIL
import PIL.ImageOps
import PIL.ImageFilter
from PIL import UnidentifiedImageError
import os
from pyrsa.rdm import RDMs
from pyrsa.util.inference_util import pool_rdm


class Icon:
    """ Icon object, i.e. an object which can be plotted into an axis or as
    an axis label.

    Args:
        image (np.ndarray or PIL.Image or RDMs or Icon)
            the image to use as an icon
            arrays and images should give the image directly
            RDMs takes the average RDM from the object
            If an Icon is passed its image property is used
        string (String)
            string to place on the icon
        col (color definition)
            background / border color
            default: None -> no border or background
        marker (matplotlib markertype)
            sets what kind of symbol to plot
        cmap (color map)
            color map applied to the image
        border_type (String)
            None : default, puts the color as a background
                where the alpha of the image is not 0
            'pad' : pads the image with the border color -> square border
            'conv' : extends the area by convolving with a circle
        border_width (integer)
            width of the border
        make_square (bool)
            if set to true the image is first reshaped into a square
        circ_cut (flag)
            sets how the icon is cut into circular shape
            None : default, no cutting
            'cut' : sets alpha to 0 out of a circular aperture
            'cosine' : sets alpha to a raised cosine window
            a number between 0 and 1 : a tukey window with the flat proportion
                of the aperture given by the number. For 0 this corresponds
                to the cosine window, for 1 it corresponds to 'cut'.
        resolution (1 or two numbers):
            sets a resolution for the icon to which the image is resized
            prior to all processing. If only one number is provided,
            the image is resized to a square with that size
        marker_front (bool):
            switches whether the marker is plotted in front or behind the
            image. If True the marker is plotted unfilled in front
            If False the marker is plotted behind the image filled.
            default = True

    """

    def __init__(self, image=None, string=None, col=None, marker=None,
                 cmap=None, border_type=None, border_width=2,
                 make_square=False, circ_cut=None, resolution=None,
                 marker_front=True, markeredgewidth=2,
                 fontsize=None, fontname=None, fontcolor=None):
        self.fontsize = fontsize
        self.fontname = fontname
        self.string = string
        self.fontcolor = fontcolor
        self.marker = marker
        self.marker_front = marker_front
        self.markeredgewidth = markeredgewidth
        self._make_square = make_square
        self._border_width = border_width
        self._border_type = border_type
        self._cmap = cmap
        self._col = col
        self._circ_cut = None
        self._resolution = None
        self.image = image
        if resolution is not None:
            self.resolution = resolution
        self.circ_cut = circ_cut

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, image):
        """ interprets image/converts it into an image"""
        if isinstance(image, Icon):
            self._image = image.image
        elif isinstance(image, RDMs):
            avg_rdm = pool_rdm(image)
            image = avg_rdm.get_matrices()[0]
            self._image = image / np.max(image)
            if self.resolution is None:
                self._resolution = np.array(100)
        elif image is not None:
            self._image = image
        else:
            self._image = None
        self.recompute_final_image()

    @property
    def string(self):
        return self._string

    @string.setter
    def string(self, string):
        if string is None or isinstance(string, str):
            self._string = string
        else:
            raise ValueError('String must be a string')

    @property
    def col(self):
        return self._col

    @col.setter
    def col(self, col):
        self._col = col
        self.recompute_final_image()

    @property
    def cmap(self):
        return self._cmap

    @cmap.setter
    def cmap(self, cmap):
        self._cmap = cmap
        self.recompute_final_image()

    @property
    def make_square(self):
        return self._make_square

    @make_square.setter
    def make_square(self, make_square):
        self._make_square = make_square
        self.recompute_final_image()

    @property
    def border_width(self):
        return self._border_width

    @border_width.setter
    def border_width(self, border_width):
        self._border_width = border_width
        self.recompute_final_image()

    @property
    def border_type(self):
        return self._border_type

    @border_type.setter
    def border_type(self, border_type):
        self._border_type = border_type
        self.recompute_final_image()

    @property
    def resolution(self):
        return self._resolution

    @resolution.setter
    def resolution(self, resolution):
        if resolution is not None:
            self._resolution = np.array(resolution)
        else:
            self._resolution = None
        self.recompute_final_image()

    @property
    def circ_cut(self):
        return self._circ_cut

    @circ_cut.setter
    def circ_cut(self, circ_cut):
        if circ_cut is None:
            self._circ_cut = None
        elif circ_cut == 'cut':
            self._circ_cut = 1
        elif circ_cut == 'cosine':
            self._circ_cut = 0
        elif circ_cut <= 1 and circ_cut >= 0:
            self._circ_cut = circ_cut
        else:
            raise ValueError('circ_cut must be in [0,1]')
        self.recompute_final_image()

    def recompute_final_image(self):
        """ computes the icon image from the parameters

        This function handles most of the image processing and must be run
        again if any properties are changed. If you use set to change
        properties this is automatically run.
        """
        if self._image is None:
            self.final_image = None
            return
        if isinstance(self._image, np.ndarray):
            if self._image.dtype == np.uint8 or np.any(self._image > 1):
                # assume image is in uint8 0-255 range
                im = self._image / 255
            else:
                im = self._image
            if self.cmap is not None:
                im = cm.get_cmap(self.cmap)(im)
            im = PIL.Image.fromarray((im * 255).astype(np.uint8))
        else:  # we hope it is a PIL image or equivalent
            im = self._image
        im = im.convert('RGBA')
        if self.make_square:
            new_size = max(im.width, im.height)
            im = im.resize((new_size, new_size), PIL.Image.NEAREST)
        if self.resolution is not None:
            if self.resolution.size == 1:
                im = im.resize((self.resolution, self.resolution),
                               PIL.Image.NEAREST)
            else:
                im = im.resize(self.resolution,
                               PIL.Image.NEAREST)
        if self.circ_cut is not None:
            middle = np.array(im.size) / 2
            x = np.arange(im.size[0]) - middle[0] + 0.5
            x = x / np.max(np.abs(x))
            y = np.arange(im.size[1]) - middle[1] + 0.5
            y = y / np.max(np.abs(y))
            yy, xx = np.meshgrid(y, x)
            r = np.sqrt(xx ** 2 + yy ** 2)
            alpha = np.empty(r.shape)
            alpha[r > 1] = 0
            alpha[r <= self.circ_cut] = 1
            val = (r > self.circ_cut) & (r <= 1)
            alpha[val] = (
                0.5 + 0.5 * np.cos(
                    np.pi * (r[val] - self.circ_cut)
                    / (1 - self.circ_cut)))
            alpha = alpha.T * np.array(im.getchannel('A'))
            alpha = PIL.Image.fromarray(np.uint8(alpha))
            im.putalpha(alpha)
        if self.col is not None:
            if self.border_type is None:
                pass
            elif self.border_type == 'alpha':
                bg_alpha = np.array(im.getchannel('A'))
                bg_alpha = bg_alpha > 0
                bg_alpha = PIL.Image.fromarray(255 * np.uint8(bg_alpha))
                bg = PIL.Image.new('RGBA', im.size, color=tuple(
                    np.uint8(255 * self.col)))
                bg.putalpha(bg_alpha)
                im = PIL.Image.alpha_composite(bg, im)
            elif self.border_type == 'pad':
                im = PIL.ImageOps.expand(
                    im,
                    border=self.border_width,
                    fill=self.col)
            elif self.border_type == 'conv':
                im = PIL.ImageOps.expand(
                    im,
                    border=self.border_width,
                    fill=(0, 0, 0, 0))
                bg_alpha = im.getchannel('A')
                bg_alpha = bg_alpha.filter(PIL.ImageFilter.BoxBlur(
                    self.border_width))
                bg_alpha = np.array(bg_alpha)
                bg_alpha = 255 * np.uint8(bg_alpha > 0)
                bg_alpha = PIL.Image.fromarray(bg_alpha)
                bg = PIL.Image.new('RGBA', im.size, color=tuple(
                    np.uint8(255 * self.col)))
                bg.putalpha(bg_alpha)
                im = PIL.Image.alpha_composite(bg, im)
        self.final_image = im

    def plot(self, x, y, ax=None, size=None):
        """ plots the icon into an axis

        Args:
            x (float)
                x-position
            y (float)
                y-position
            ax (matplotlib axis)
                the axis to plot in
            size : float
                size of the icon scaling the image

        """
        if ax is None:
            ax = plt.gca()
        if size is None:
            size = 1
        if self.final_image is not None:
            imagebox = OffsetImage(self.final_image, zoom=size)
            ab = AnnotationBbox(
                imagebox, (x, y),  frameon=False,
                pad=0)
            ax.add_artist(ab)
            zorder = ab.zorder
        else:
            zorder = 0
        if self.marker:
            if self.final_image is not None:
                markersize = max(self.final_image.size)
            else:
                markersize = 50
            markersize = markersize * size
            if self.marker_front:
                plt.plot(x, y, marker=self.marker, markeredgecolor=self.col,
                         markerfacecolor=(0, 0, 0, 0), markersize=markersize,
                         zorder=zorder + 0.1,
                         markeredgewidth=self.markeredgewidth)
            else:
                plt.plot(x, y, marker=self.marker, markeredgecolor=self.col,
                         markerfacecolor=self.col, markersize=markersize,
                         zorder=zorder - 0.1,
                         markeredgewidth=self.markeredgewidth)
        if self.string is not None:
            ax.annotate(self.string, (x, y),
                        horizontalalignment='center',
                        verticalalignment='center',
                        zorder=zorder + 0.2,
                        fontsize=self.fontsize, fontname=self.fontname,
                        color=self.fontcolor)

    def _tick_label(self, x, y, size, offset=7, ax=None, linewidth=None,
            xybox=None, xycoords=None, box_alignment=None,
                horizontalalignment=None,
                verticalalignment=None):

        """
        uses the icon as a ticklabel at location x

        Args:
            x (float)
                the position of the tick
            size (float)
                scaling the size of the icon
            offset (integer)
                how far the icon should be from the axis in points
            ax (matplotlib axis)
                the axis to put the label on

        """
        if ax is None:
            ax = plt.gca()
        tickline_color = self.col
        if not np.any(tickline_color):
            tickline_color = [.8, .8, .8]
        if self.final_image is not None:
            imagebox = OffsetImage(self.final_image, zoom=size)
            ab = AnnotationBbox(
                imagebox, (x, y),
                xybox=xybox,
                xycoords=xycoords,
                box_alignment=box_alignment,
                boxcoords='offset points',
                bboxprops={'edgecolor': 'w', 'facecolor': 'w'},
                arrowprops={
                    'linewidth': linewidth,
                    'color': tickline_color,
                    'arrowstyle': '-',
                    'shrinkA': 0,
                    'shrinkB': 1
                    },
                pad=0.,
                annotation_clip=False)
            zorder = ab.zorder
            ax.add_artist(ab)
        else:
            zorder = 0
        if self.marker:
            if self.final_image is not None:
                markersize = max(self.final_image.size)
            else:
                markersize = 50
            markersize = markersize * size
            d = DrawingArea(markersize, markersize)
            if self.marker_front:
                zorder_marker = zorder + 0.1
            else:
                zorder_marker = zorder - 0.1
            d.set_zorder(zorder_marker)
            d.set_alpha(0)
            if self.marker_front:
                d.add_artist(plt.Line2D(
                    [markersize / 2], [markersize / 2],
                    marker=self.marker, markeredgecolor=self.col,
                    markerfacecolor=(0, 0, 0, 0), markersize=markersize,
                    markeredgewidth=self.markeredgewidth,
                    transform=d.get_transform(),
                    zorder=zorder_marker))
            else:
                d.add_artist(plt.Line2D(
                    [markersize / 2], [markersize / 2],
                    marker=self.marker, markeredgecolor=self.col,
                    markerfacecolor=self.col, markersize=markersize,
                    markeredgewidth=self.markeredgewidth,
                    transform=d.get_transform(),
                    zorder=zorder_marker))
            ab_marker = AnnotationBbox(
                d, (x, y),
                xybox=xybox,
                xycoords=xycoords,
                box_alignment=box_alignment,
                boxcoords='offset points',
                bboxprops={'edgecolor': 'none', 'facecolor': 'none'},
                arrowprops={
                    'linewidth': linewidth,
                    'arrowstyle': '-',
                    'shrinkA': 0,
                    'shrinkB': 1
                    },
                pad=0.,
                annotation_clip=False)
            ab_marker.set_zorder(zorder_marker)
            ab_marker.set_alpha(0)
            ax.add_artist(ab_marker)
        if self.string is not None:
            ax.annotate(
                self.string, (x, y),
                xytext=xybox,
                xycoords=xycoords,
                textcoords='offset points',
                horizontalalignment=horizontalalignment,
                verticalalignment=verticalalignment,
                arrowprops={
                    'arrowstyle': '-',
                    'shrinkA': 0,
                    'shrinkB': 1
                    },
                zorder=zorder + 0.2,
                fontsize=self.fontsize, fontname=self.fontname,
                color=self.fontcolor)

    def x_tick_label(self, x, size, offset, **kwarg):
        """
        uses the icon as a ticklabel at location x

        Args:
            x (float)
                the position of the tick
            size (float)
                scaling the size of the icon
            offset (integer)
                how far the icon should be from the axis in points
            ax (matplotlib axis)
                the axis to put the label on

        """
        self._tick_label(x=x, y=0, size=size, offset=offset, xybox=(0, -offset), xycoords=('data', 'axes fraction'),
                box_alignment=(.5, 1),
                horizontalalignment='center',
                verticalalignment='top',
                **kwarg)

    def y_tick_label(self, y, size, offset, **kwarg):
        """
        uses the icon as a ticklabel at location x

        Args:
            y (float)
                the position of the tick
            size (float)
                scaling the size of the icon
            offset (integer)
                how far the icon should be from the axis in points
            ax (matplotlib axis)
                the axis to put the label on

        """
        self._tick_label(x=0, y=y, size=size, offset=offset, xybox=(-offset, 0), xycoords=('axes fraction', 'data'),
                box_alignment=(1, .5),
                horizontalalignment='right',
                verticalalignment='center',
                **kwarg)


def icons_from_folder(folder, resolution=None, col=None,
                      cmap=None, border_type=None, border_width=2,
                      make_square=False, circ_cut=None):
    """ generates a dictionary of Icons for all images in a folder

    """
    icons = dict()
    for filename in os.listdir(folder):
        try:
            im = PIL.Image.open(filename)
            icons[filename] = Icon(
                image=im, col=col, resolution=resolution,
                cmap=cmap, border_type=border_type,
                border_width=border_width,
                make_square=make_square, circ_cut=circ_cut)
        except (FileNotFoundError, UnidentifiedImageError, IsADirectoryError,
                PermissionError):
            pass
    return icons
