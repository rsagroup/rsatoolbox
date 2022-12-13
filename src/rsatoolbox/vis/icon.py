#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
icon object which can be plotted into an axis
"""

import os
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, DrawingArea
import numpy as np
import PIL
import PIL.ImageOps
import PIL.ImageFilter
from PIL import UnidentifiedImageError
from rsatoolbox.rdm import RDMs
from rsatoolbox.util.pooling import pool_rdm
if hasattr(matplotlib.colormaps, 'get_cmap'):
    mpl_get_cmap = matplotlib.colormaps.get_cmap
else:
    mpl_get_cmap = matplotlib.cm.get_cmap  # drop:py37


class Icon:
    """
    Icon object, i.e. an object which can be plotted into an axis or as
    an axis label.

    Args:
        image (np.ndarray or PIL.Image or RDMs or Icon)
            the image to use as an icon
            arrays and images should give the image directly
            RDMs takes the average RDM from the object
            If an Icon is passed its image property is used

        string (String)
            string to place on the icon

        color (color definition)
            background / border color
            default: None -> no border or background

        marker (matplotlib markertype)
            sets what kind of symbol to plot

        cmap (color map)
            color map applied to the image

        border_type (String)

            - None : default, puts the color as a background
              where the alpha of the image is not 0

            - 'pad' : pads the image with the border color -> square border

            - 'conv' : extends the area by convolving with a circle

        border_width (integer)
            width of the border

        make_square (bool)
            if set to true the image is first reshaped into a square

        circ_cut (flag)
            sets how the icon is cut into circular shape:

            - None : default, no cutting

            - 'cut' : sets alpha to 0 out of a circular aperture

            - 'cosine' : sets alpha to a raised cosine window

            - a number between 0 and 1 : a tukey window with the flat proportion
              of the aperture given by the number. For 0 this corresponds
              to the cosine window, for 1 it corresponds to 'cut'.

        resolution (one or two numbers):
            sets a resolution for the icon to which the image is resized
            prior to all processing. If only one number is provided,
            the image is resized to a square with that size

        marker_front (bool):
            switches whether the marker is plotted in front or behind the
            image. If True the marker is plotted unfilled in front
            If False the marker is plotted behind the image filled.

            default = True

        font_size (float)
            size of any annotation text

        font_name (str):
            annotation font

        font_color (np.ndarray)
            font color for annotations

    """

    def __init__(
            self, image=None, string=None, color=None, marker=None,
            cmap=None, border_type=None, border_width=2, make_square=False,
            circ_cut=None, resolution=None, marker_front=True,
            markeredgewidth=2, font_size=None, font_name=None,
            font_color=None):
        self.final_image = None
        self.font_size = font_size
        self.font_name = font_name
        self.string = string
        self.font_color = font_color
        self.marker = marker
        self.marker_front = marker_front
        self.markeredgewidth = markeredgewidth
        self._make_square = make_square
        self._border_width = border_width
        self._border_type = border_type
        self._cmap = cmap
        self._color = color
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
            raise ValueError("String must be a string")

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, color):
        self._color = color
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
        elif circ_cut == "cut":
            self._circ_cut = 1
        elif circ_cut == "cosine":
            self._circ_cut = 0
        elif 0 <= circ_cut <= 1:
            self._circ_cut = circ_cut
        else:
            raise ValueError("circ_cut must be in [0,1]")
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
                im = mpl_get_cmap(self.cmap)(im)
            im = PIL.Image.fromarray((im * 255).astype(np.uint8))
        else:  # we hope it is a PIL image or equivalent
            im = self._image
        im = im.convert("RGBA")
        if self.make_square:
            new_size = max(im.width, im.height)
            if int(PIL.__version__[0]) >= 9:
                im = im.resize((new_size, new_size), PIL.Image.Resampling.NEAREST)
            else:
                im = im.resize((new_size, new_size), PIL.Image.NEAREST)
        if self.resolution is not None:
            if self.resolution.size == 1:
                if int(PIL.__version__[0]) >= 9:
                    im = im.resize((self.resolution, self.resolution), PIL.Image.Resampling.NEAREST)
                else:
                    im = im.resize((self.resolution, self.resolution), PIL.Image.NEAREST)
            else:
                if int(PIL.__version__[0]) >= 9:
                    im = im.resize(self.resolution, PIL.Image.Resampling.NEAREST)
                else:
                    im = im.resize(self.resolution, PIL.Image.NEAREST)
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
            alpha[val] = 0.5 + 0.5 * np.cos(
                np.pi * (r[val] - self.circ_cut) / (1 - self.circ_cut)
            )
            alpha = alpha.T * np.array(im.getchannel("A"))
            alpha = PIL.Image.fromarray(np.uint8(alpha))
            im.putalpha(alpha)
        if self.color is not None:
            if self.border_type is None:
                pass
            elif self.border_type == "alpha":
                bg_alpha = np.array(im.getchannel("A"))
                bg_alpha = bg_alpha > 0
                bg_alpha = PIL.Image.fromarray(255 * np.uint8(bg_alpha))
                bg = PIL.Image.new(
                    "RGBA", im.size, color=tuple(np.uint8(255 * self.color))
                )
                bg.putalpha(bg_alpha)
                im = PIL.Image.alpha_composite(bg, im)
            elif self.border_type == "pad":
                im = PIL.ImageOps.expand(im, border=self.border_width, fill=self.color)
            elif self.border_type == "conv":
                im = PIL.ImageOps.expand(
                    im, border=self.border_width, fill=(0, 0, 0, 0)
                )
                bg_alpha = im.getchannel("A")
                bg_alpha = bg_alpha.filter(PIL.ImageFilter.BoxBlur(self.border_width))
                bg_alpha = np.array(bg_alpha)
                bg_alpha = 255 * np.uint8(bg_alpha > 0)
                bg_alpha = PIL.Image.fromarray(bg_alpha)
                bg = PIL.Image.new(
                    "RGBA", im.size, color=tuple(np.uint8(255 * self.color))
                )
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
            ab = AnnotationBbox(imagebox, (x, y), frameon=False, pad=0)
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
                plt.plot(
                    x,
                    y,
                    marker=self.marker,
                    markeredgecolor=self.color,
                    markerfacecolor=(0, 0, 0, 0),
                    markersize=markersize,
                    zorder=zorder + 0.1,
                    markeredgewidth=self.markeredgewidth,
                )
            else:
                plt.plot(
                    x,
                    y,
                    marker=self.marker,
                    markeredgecolor=self.color,
                    markerfacecolor=self.color,
                    markersize=markersize,
                    zorder=zorder - 0.1,
                    markeredgewidth=self.markeredgewidth,
                )
        if self.string is not None:
            ax.annotate(
                self.string,
                (x, y),
                horizontalalignment="center",
                verticalalignment="center",
                zorder=zorder + 0.2,
                fontsize=self.font_size,
                fontname=self.font_name,
                color=self.font_color,
            )

    def _tick_label(
        self,
        x,
        y,
        size,
        ax=None,
        linewidth=None,
        xybox=None,
        xycoords=None,
        box_alignment=None,
        horizontalalignment=None,
        verticalalignment=None,
        rotation=None,
    ):
        """
        uses the icon as a ticklabel at location x

        Args:
            x (float)
                the horizontal position of the tick
            y (float)
                the vertical position of the tick
            size (float)
                scaling the size of the icon
            ax (matplotlib axis)
                the axis to put the label on

        """
        ret_val = {}
        if ax is None:
            ax = plt.gca()
        tickline_color = self.color
        # np.any chokes on str input so need to test for this first
        if not (isinstance(tickline_color, str) or np.any(tickline_color)):
            tickline_color = [0.8, 0.8, 0.8]
        if self.final_image is not None:
            imagebox = OffsetImage(self.final_image, zoom=size, dpi_cor=True)
            ret_val['image'] = AnnotationBbox(
                imagebox,
                (x, y),
                xybox=xybox,
                xycoords=xycoords,
                box_alignment=box_alignment,
                boxcoords="offset points",
                bboxprops={"edgecolor": "none", "facecolor": "none"},
                arrowprops={
                    "linewidth": linewidth,
                    "color": tickline_color,
                    "arrowstyle": "-",
                    "shrinkA": 0,
                    "shrinkB": 1,
                },
                pad=0.0,
                annotation_clip=False,
            )
            zorder = ret_val['image'].zorder
            ax.add_artist(ret_val['image'])
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
                d.add_artist(
                    plt.Line2D(
                        [markersize / 2],
                        [markersize / 2],
                        marker=self.marker,
                        markeredgecolor=self.color,
                        markerfacecolor=(0, 0, 0, 0),
                        markersize=markersize,
                        markeredgewidth=self.markeredgewidth,
                        transform=d.get_transform(),
                        zorder=zorder_marker,
                    )
                )
            else:
                d.add_artist(
                    plt.Line2D(
                        [markersize / 2],
                        [markersize / 2],
                        marker=self.marker,
                        markeredgecolor=self.color,
                        markerfacecolor=self.color,
                        markersize=markersize,
                        markeredgewidth=self.markeredgewidth,
                        transform=d.get_transform(),
                        zorder=zorder_marker,
                    )
                )
            ret_val['marker'] = AnnotationBbox(
                d,
                (x, y),
                xybox=xybox,
                xycoords=xycoords,
                box_alignment=box_alignment,
                boxcoords="offset points",
                bboxprops={"edgecolor": "none", "facecolor": "none"},
                arrowprops={
                    "linewidth": linewidth,
                    "color": tickline_color,
                    "arrowstyle": "-",
                    "shrinkA": 0,
                    "shrinkB": 1,
                },
                pad=0.0,
                annotation_clip=False,
            )
            ret_val['marker'].set_zorder(zorder_marker)
            ret_val['marker'].set_alpha(0)
            ax.add_artist(ret_val['marker'])
        if self.string is not None:
            ret_val['string'] = ax.annotate(
                    self.string,
                    (x, y),
                    xytext=xybox,
                    xycoords=xycoords,
                    textcoords="offset points",
                    horizontalalignment=horizontalalignment,
                    verticalalignment=verticalalignment,
                    arrowprops={
                        "linewidth": linewidth,
                        "color": tickline_color,
                        "arrowstyle": "-",
                        "shrinkA": 0,
                        "shrinkB": 1,
                        },
                    zorder=zorder + 0.2,
                    fontsize=self.font_size,
                    fontname=self.font_name,
                    color=self.font_color,
                    rotation=rotation,
                    )
        return ret_val

    def x_tick_label(self, x, size, offset, **kwarg):
        """
        uses the icon as a ticklabel at location x

        Args:
            x (float)
                the position of the tick
            size (float)
                scaling the size of the icon
            offset (integer)
                how far the icon should be from the axis in axis units
            ax (matplotlib axis)
                the axis to put the label on

        """
        return self._tick_label(
            x=x,
            y=0,
            size=size,
            xybox=(0, -offset),
            xycoords=("data", "axes fraction"),
            box_alignment=(0.5, 1),
            horizontalalignment="center",
            verticalalignment="bottom",
            rotation=90,
            **kwarg
        )

    def y_tick_label(self, y, size, offset, **kwarg):
        """
        uses the icon as a ticklabel at location x

        Args:
            y (float)
                the position of the tick
            size (float)
                scaling the size of the icon
            offset (integer)
                how far the icon should be from the axis in axis units
            ax (matplotlib axis)
                the axis to put the label on

        """
        return self._tick_label(
            x=0,
            y=y,
            size=size,
            xybox=(-offset, 0),
            xycoords=("axes fraction", "data"),
            box_alignment=(1, 0.5),
            horizontalalignment="right",
            verticalalignment="center",
            rotation=0,
            **kwarg
        )


def icons_from_folder(
    folder,
    resolution=None,
    color=None,
    cmap=None,
    border_type=None,
    border_width=2,
    make_square=False,
    circ_cut=None,
):
    """ generates a dictionary of Icons for all images in a folder

    """
    icons = dict()
    for filename in os.listdir(folder):
        try:
            im = PIL.Image.open(filename)
            icons[filename] = Icon(
                image=im,
                color=color,
                resolution=resolution,
                cmap=cmap,
                border_type=border_type,
                border_width=border_width,
                make_square=make_square,
                circ_cut=circ_cut,
            )
        except (
            FileNotFoundError,
            UnidentifiedImageError,
            IsADirectoryError,
            PermissionError,
        ):
            pass
    return icons
