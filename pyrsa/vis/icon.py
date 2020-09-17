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
        if isinstance(image, Icon):
            self.image = image.image
        elif isinstance(image, RDMs):
            avg_rdm = pool_rdm(image)
            image = avg_rdm.get_matrices()[0]
            self.image = image / np.max(image)
            if resolution is None:
                resolution = 100
        else:
            self.image = image
        self.fontsize = fontsize
        self.fontname = fontname
        self.marker_front = marker_front
        if resolution is not None:
            self.resolution = np.array(resolution)
        else:
            self.resolution = None
        self.markeredgewidth = markeredgewidth
        self.make_square = make_square
        self.border_width = border_width
        self.border_type = border_type
        self.cmap = cmap
        self.marker = marker
        self.col = col
        self.image = image
        self.string = string
        self.fontcolor = fontcolor
        if circ_cut is None:
            self.circ_cut = None
        elif circ_cut == 'cut':
            self.circ_cut = 1
        elif circ_cut == 'cosine':
            self.circ_cut = 0
        else:
            assert circ_cut <= 1 and circ_cut >= 0, \
                'a numeric circ_cut must be in [0,1]'
            self.circ_cut = circ_cut
        self.set(image, string, col, marker, cmap, border_type)

    def set(self, image=None, string=None, col=None, marker=None,
            cmap=None, border_type=None, border_width=None, make_square=None,
            circ_cut=None, resolution=None, marker_front=None,
            markeredgewidth=None, fontsize=None, fontname=None,
            fontcolor=None):
        """ sets individual parameters of the object and recomputes the
        icon image
        """
        if isinstance(image, Icon):
            self.image = image.image
        elif isinstance(image, RDMs):
            avg_rdm = pool_rdm(image)
            image = avg_rdm.get_matrices()[0]
            self.image = image / np.max(image)
            if resolution is None:
                resolution = 100
        elif image is not None:
            self.image = image
        if string is not None:
            self.string = string
        if col is not None:
            self.col = col
        if marker is not None:
            self.marker = marker
        if cmap is not None:
            self.cmap = cmap
        if border_type is not None:
            self.border_type = border_type
        if border_width is not None:
            self.border_width = border_width
        if make_square is not None:
            self.make_square = make_square
        if resolution is not None:
            self.resolution = np.array(resolution)
        if circ_cut is not None:
            if circ_cut == 'cut':
                self.circ_cut = 1
            elif circ_cut == 'cosine':
                self.circ_cut = 0
            else:
                assert circ_cut <= 1 and circ_cut >= 0, \
                    'a numeric circ_cut must be in [0,1]'
                self.circ_cut = circ_cut
        if marker_front is not None:
            self.marker_front = marker_front
        if markeredgewidth is not None:
            self.markeredgewidth = markeredgewidth
        if fontname is not None:
            self.fontname = fontname
        if fontsize is not None:
            self.fontsize = fontsize
        if fontcolor is not None:
            self.fontcolor = fontcolor
        self.recompute_final_image()

    def recompute_final_image(self):
        """ computes the icon image from the parameters

        This function handles most of the image processing and must be run
        again if any properties are changed. If you use set to change
        properties this is automatically run.
        """
        if self.image is None:
            self.final_image = None
            return
        elif isinstance(self.image, np.ndarray):
            if self.image.dtype == np.float and np.any(self.image > 1):
                im = self.image / 255
            else:
                im = self.image
            if self.cmap is not None:
                im = cm.get_cmap(self.cmap)(im)
            im = PIL.Image.fromarray((im * 255).astype(np.uint8))
        else:  # we hope it is a PIL image or equivalent
            im = self.image
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
                bg = PIL.Image.new('RGBA', im.size, color=self.col)
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
                bg = PIL.Image.new('RGBA', im.size, color=self.col)
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
        if self.final_image is not None:
            if size is None:
                imagebox = OffsetImage(self.final_image, zoom=1)
            else:
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

    def x_tick_label(self, x, size, offset=7, ax=None):
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
        if self.final_image is not None:
            imagebox = OffsetImage(self.final_image, zoom=size)
            ab = AnnotationBbox(
                imagebox, (x, 0),
                xybox=(0, -offset),
                xycoords=('data', 'axes fraction'),
                box_alignment=(.5, 1),
                boxcoords='offset points',
                bboxprops={'edgecolor': 'none', 'facecolor': 'none'},
                arrowprops={
                    'arrowstyle': '-',
                    'shrinkA': 0,
                    'shrinkB': 1
                    },
                pad=0.1)
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
                d, (x, 0),
                xybox=(0, -offset),
                xycoords=('data', 'axes fraction'),
                box_alignment=(.5, 1),
                boxcoords='offset points',
                bboxprops={'edgecolor': 'none', 'facecolor': 'none'},
                arrowprops={
                    'arrowstyle': '-',
                    'shrinkA': 0,
                    'shrinkB': 1
                    },
                pad=0.1)
            ab_marker.set_zorder(zorder_marker)
            ab_marker.set_alpha(0)
            ax.add_artist(ab_marker)
        if self.string is not None:
            ax.annotate(
                self.string, (x, 0),
                xytext=(0, -offset),
                xycoords=('data', 'axes fraction'),
                textcoords='offset points',
                horizontalalignment='center',
                verticalalignment='top',
                arrowprops={
                    'arrowstyle': '-',
                    'shrinkA': 0,
                    'shrinkB': 1
                    },
                zorder=zorder + 0.2,
                fontsize=self.fontsize, fontname=self.fontname,
                color=self.fontcolor)

    def y_tick_label(self, y, size, offset=7, ax=None):
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
        if ax is None:
            ax = plt.gca()
        if self.final_image is not None:
            imagebox = OffsetImage(self.final_image, zoom=size)
            ab = AnnotationBbox(
                imagebox, (0, y),
                xybox=(-offset, 0),
                xycoords=('axes fraction', 'data'),
                box_alignment=(1, .5),
                boxcoords='offset points',
                bboxprops={'edgecolor': 'none', 'facecolor': 'none'},
                arrowprops={
                    'arrowstyle': '-',
                    'shrinkA': 0,
                    'shrinkB': 1
                    },
                pad=0.1)
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
                d, (0, y),
                xybox=(-offset, 0),
                xycoords=('axes fraction', 'data'),
                box_alignment=(1, 0.5),
                boxcoords='offset points',
                bboxprops={'edgecolor': 'none', 'facecolor': 'none'},
                arrowprops={
                    'arrowstyle': '-',
                    'shrinkA': 0,
                    'shrinkB': 1
                    },
                pad=0.1)
            ab_marker.set_zorder(zorder_marker)
            ab_marker.set_alpha(0)
            ax.add_artist(ab_marker)
        if self.string is not None:
            ax.annotate(
                self.string, (0, y),
                xytext=(-offset, 0),
                xycoords=('axes fraction', 'data'),
                textcoords='offset points',
                horizontalalignment='right',
                verticalalignment='center',
                arrowprops={
                    'arrowstyle': '-',
                    'shrinkA': 0,
                    'shrinkB': 1
                    },
                zorder=zorder + 1,
                fontsize=self.fontsize, fontname=self.fontname,
                color=self.fontcolor)


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
