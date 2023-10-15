"""
Plot showing an RDMs object

public API:

- show_rdm()
- show_rdm_panel()
"""
from __future__ import annotations
import itertools
from pathlib import Path
from typing import TYPE_CHECKING, Union, Tuple, Optional, Literal, Dict, Any, List
from enum import Enum
from math import ceil
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
import rsatoolbox.rdm
from rsatoolbox.rdm.rdms import RDMs
from rsatoolbox import vis
from rsatoolbox.vis.colors import rdm_colormap_classic
from rsatoolbox.resources import get_style
if TYPE_CHECKING:
    import numpy.typing as npt
    from matplotlib.axes._axes import Axes
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Colormap
    from matplotlib.colorbar import Colorbar
    from matplotlib.figure import Figure
    from matplotlib.text import Text
    from matplotlib.image import AxesImage
    from matplotlib.axis import XAxis, YAxis
    from numpy.typing import NDArray, ArrayLike


class Axis(Enum):
    """X or Y axis Enum
    """
    X = 'x'
    Y = 'y'


def show_rdm(
    rdms: rsatoolbox.rdm.RDMs,
    pattern_descriptor: Optional[str] = None,
    cmap: Union[str, Colormap] = 'bone',
    rdm_descriptor: Optional[str] = None,
    n_column: Optional[int] = None,
    n_row: Optional[int] = None,
    show_colorbar: Optional[str] = None,
    gridlines: Optional[npt.ArrayLike] = None,
    num_pattern_groups: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None,
    nanmask: NDArray | str | None = "diagonal",
    style: Optional[Union[str, Path]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    icon_spacing: float = 1.0,
    linewidth: float = 0.5,
) -> Tuple[Figure, NDArray, Dict[int, Dict[str, Any]]]:
    """show_rdm. Heatmap figure for RDMs instance, with one panel per RDM.

    Args:
        rdm (rsatoolbox.rdm.RDMs): RDMs object to be plotted.
        pattern_descriptor (str): Key into rdm.pattern_descriptors to use for axis
            labels.
        cmap (str or Colormap): Colormap to be used.
            Either the name of a Matplotlib built-in colormap, a Matplotlib
            Colormap compatible object, or 'classic' for the matlab toolbox
            colormap. Defaults to 'bone'.
        rdm_descriptor (str): Key for rdm_descriptor to use as panel title, or
            str for direct labeling.
        n_column (int): Number of columns in subplot arrangement.
        n_row (int): Number of rows in subplot arrangement.
        show_colorbar (str): Set to 'panel' or 'figure' to display a colorbar. If
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
        nanmask (Union[npt.ArrayLike, str, None]): boolean mask defining RDM elements to suppress
            (by default, the diagonal).
            Use the string "diagonal" to suppress the diagonal.
        style (Union[str, Path]): Path to mplstyle file that controls
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
        Tuple[Figure, ArrayLike, Dict]:

        Tuple of

            - Handle to created figure.
            - Subplot axis handles from plt.subplots.
            - Nested dict containing handles to all other plotted
              objects (icon labels, colorbars, etc). The key at the first level is the axis index.

    """
    # create a plot "configuration" object which resolves all parameters
    conf = MultiRdmPlot.from_show_rdm_args(
        rdms, pattern_descriptor, cmap, rdm_descriptor, n_column, n_row,
        show_colorbar, gridlines, num_pattern_groups, figsize, nanmask,
        style, vmin, vmax, icon_spacing, linewidth,
    )
    # A dictionary of figure element handles
    handles = dict()
    handles[-1] = dict() # fig level handles
    # create a list of (row index, column index) tuples
    rc_tuples = list(itertools.product(range(conf.n_row), range(conf.n_column)))
    # number of empty panels at the top
    n_empty = (conf.n_row * conf.n_column) - rdms.n_rdm
    with plt.style.context(conf.style):
        fig, ax_array = plt.subplots(
            nrows=conf.n_row,
            ncols=conf.n_column,
            sharex=True,
            sharey=True,
            squeeze=False,
            figsize=conf.figsize,
        )
        p = 0
        for p, (r, c) in enumerate(rc_tuples):
            handles[p] = dict()
            rdm_index = p - n_empty ## rdm index
            if rdm_index < 0:
                ax_array[r, c].set_visible(False)
                continue

            handles[p]["image"] = _show_rdm_panel(conf.for_single(rdm_index), ax_array[r, c])

            if show_colorbar == "panel":
                # needs to happen before labels because it resizes the axis
                handles[p]["colorbar"] = _rdm_colorbar(
                    mappable=handles[p]["image"],
                    fig=fig,
                    ax=ax_array[r, c],
                    title=conf.dissimilarity_measure
                )
            if c == 0 and pattern_descriptor:
                handles[p]["y_labels"] = _add_descriptor_labels(Axis.Y, ax_array[r, c], conf)
            if r == 0 and pattern_descriptor:
                handles[p]["x_labels"] = _add_descriptor_labels(Axis.X, ax_array[r, c], conf)

        if show_colorbar == "figure":
            handles[-1]["colorbar"] = _rdm_colorbar(
                mappable=handles[p]["image"],
                fig=fig,
                ax=ax_array[0, 0],
                title=conf.dissimilarity_measure,
            )
            _adjust_colorbar_pos(handles[-1]["colorbar"], ax_array[0, 0])

    return fig, ax_array, handles


def _adjust_colorbar_pos(cb: Colorbar, parent: Axes) -> None:
    """Moves figure-level colorbar to the right position

    Args:
        cb (Colorbar): The matplotlib colorbar object
        parent (Axes): Parent object axes
    """
    # key challenge is to obtain a similarly-sized colorbar to the 'panel' case
    # BUT positioned centered on the reserved subplot axes
    #parent = ax_array[-1, -1]
    cbax_parent_orgpos = parent.get_position(original=True)
    # use last instance of 'image' (should all be yoked at this point)
    cbax_pos = cb.ax.get_position()
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
    cb.ax.set_position((x0, y0, cbax_pos.width, cbax_pos.height))


def _rdm_colorbar(mappable: ScalarMappable, fig: Figure, ax: Axes, title: str) -> Colorbar:
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
        ticks=LinearLocator(numticks=3),
    )
    cb.ax.set_title(title, loc="left", fontdict=dict(fontweight="normal"))
    return cb


def show_rdm_panel(
    rdms: rsatoolbox.rdm.RDMs,
    ax: Optional[Axes] = None,
    cmap: Union[str, Colormap] = 'bone',
    nanmask: Optional[NDArray] = None,
    rdm_descriptor: Optional[str] = None,
    gridlines: Optional[npt.ArrayLike] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> AxesImage:
    """show_rdm_panel. Add RDM heatmap to the axis ax.

    Args:
        rdm (rsatoolbox.rdm.RDMs): RDMs object to be plotted (n_rdm must be 1).
        ax (matplotlib.axes._axes.Axes): Matplotlib axis handle. plt.gca() by default.
        cmap (str or Colormap): Colormap to be used.
            Either the name of a Matplotlib built-in colormap, a Matplotlib
            Colormap compatible object, or 'classic' for the matlab toolbox
            colormap. Defaults to 'bone'.
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
    conf = SingleRdmPlot.from_show_rdm_panel_args(rdms, cmap, nanmask,
        rdm_descriptor, gridlines, vmin, vmax)
    return _show_rdm_panel(conf, ax or plt.gca())


def _show_rdm_panel(conf: SingleRdmPlot, ax: Axes) -> AxesImage:
    """Plot a single RDM based on a plot configuration object

    Args:
        conf (SingleRdmPlot): _description_
        ax (Axes): _description_

    Returns:
        AxesImage: _description_
    """
    rdmat = conf.rdms.get_matrices()[0, :, :]
    if np.any(conf.nanmask):
        rdmat[conf.nanmask] = np.nan
    image = ax.imshow(rdmat, cmap=conf.cmap, vmin=conf.vmin, vmax=conf.vmax,
        interpolation='none')
    ax.set_xlim(-0.5, conf.rdms.n_cond - 0.5)
    ax.set_ylim(conf.rdms.n_cond - 0.5, -0.5)
    ax.xaxis.set_ticks(conf.gridlines)
    ax.yaxis.set_ticks(conf.gridlines)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.xaxis.set_ticks(np.arange(conf.rdms.n_cond), minor=True)
    ax.yaxis.set_ticks(np.arange(conf.rdms.n_cond), minor=True)
    # hide minor ticks by default
    ax.xaxis.set_tick_params(length=0, which="minor")
    ax.yaxis.set_tick_params(length=0, which="minor")
    ax.set_title(conf.title)
    return image


def _add_descriptor_labels(which_axis: Axis, ax: Axes, conf: MultiRdmPlot) -> List:
    """_add_descriptor_labels.

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
    if which_axis == Axis.X:
        icon_method = "x_tick_label"
        axis = ax.xaxis
        horizontalalignment="center"
    else:
        icon_method = "y_tick_label"
        axis = ax.yaxis
        horizontalalignment="right"
    descriptor_arr = np.asarray(conf.rdms.pattern_descriptors[conf.pattern_descriptor])
    if isinstance(descriptor_arr[0], vis.Icon):
        return _add_descriptor_icons(
            descriptor_arr,
            icon_method,
            n_cond=conf.rdms.n_cond,
            ax=axis.axes,
            icon_spacing=conf.icon_spacing,
            num_pattern_groups=conf.num_pattern_groups,
            linewidth=conf.linewidth,
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
    axis: Union[XAxis, YAxis],
    horizontalalignment: str = "center",
    is_x_axis: bool = False,
) -> List[Text]:
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


class MultiRdmPlot:
    """Configuration for the multi-rdm plot
    """

    rdms: RDMs
    pattern_descriptor: Optional[str]
    cmap: Union[str, Colormap]
    rdm_descriptor: str
    n_column: int
    n_row: int
    show_colorbar: Optional[Literal["panel"] | Literal["figure"]]
    gridlines: NDArray
    num_pattern_groups: int
    figsize: Tuple[float, float]
    nanmask: NDArray
    style: Path
    vmin: Optional[float]
    vmax: Optional[float]
    icon_spacing: float
    linewidth: float
    n_panel: int
    dissimilarity_measure: str

    @classmethod
    def from_show_rdm_args(
        cls,
        rdm: RDMs,
        pattern_descriptor: Optional[str] = None,
        cmap: Union[str, Colormap] = 'bone',
        rdm_descriptor: Optional[str] = None,
        n_column: Optional[int] = None,
        n_row: Optional[int] = None,
        show_colorbar: Optional[str] = None,
        gridlines: Optional[npt.ArrayLike] = None,
        num_pattern_groups: Optional[int] = None,
        figsize: Optional[Tuple[float, float]] = None,
        nanmask: NDArray | str | None = "diagonal",
        style: Optional[Union[str, Path]] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        icon_spacing: float = 1.0,
        linewidth: float = 0.5,
    ) -> MultiRdmPlot:
        """Create an object from the original arguments to show_rdm()
        """
        conf = __class__()
        if show_colorbar not in (None, "panel", "figure"):
            raise ValueError(
                f"show_colorbar can be None, panel or figure, got: {show_colorbar}"
            )
        conf.show_colorbar = show_colorbar
        if nanmask is None:
            nanmask = np.zeros((rdm.n_cond, rdm.n_cond), dtype=bool)
        elif isinstance(nanmask, str):
            if nanmask == "diagonal":
                nanmask = np.eye(rdm.n_cond, dtype=bool)
            else:
                raise ValueError("Invalid nanmask value")
        conf.nanmask = nanmask
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
        conf.n_panel = n_panel
        conf.vmin = vmin
        conf.vmax = vmax
        if (n_column is None) and (n_row is None):
            n_column = ceil(np.sqrt(n_panel))
        if n_row is None:
            n_row = ceil(n_panel / n_column)
        if n_column is None:
            n_column = ceil(n_panel / n_row)
        conf.n_column = n_column
        conf.n_row = n_row
        if (n_column * n_row) < rdm.n_rdm:
            raise ValueError(
                f"invalid n_row*n_column specification for {n_panel} rdms: {n_row}*{n_column}"
            )
        if figsize is None:
            # scale with number of RDMs, up to a point (the intersection of A4 and us
            # letter)
            figsize = (min(2 * n_column, 8.3), min(2 * n_row, 11))
        conf.figsize = figsize
        gridlines = gridlines or list()
        if not np.any(gridlines):
            # empty list to disable gridlines
            gridlines = []
            if num_pattern_groups:
                # grid by pattern groups if they exist and explicit grid setting does not
                gridlines = np.arange(
                    num_pattern_groups - 0.5, rdm.n_cond + 0.5, num_pattern_groups
                )
        conf.gridlines = np.asarray(gridlines)
        if num_pattern_groups is None or num_pattern_groups == 0:
            num_pattern_groups = 1
        conf.num_pattern_groups = num_pattern_groups
        conf.n_panel = n_panel
        conf.style = Path(str(style)) if style is not None else get_style()
        conf.icon_spacing = icon_spacing
        conf.linewidth = linewidth
        if cmap == 'classic':
            cmap = rdm_colormap_classic()
        conf.cmap = cmap
        conf.rdms = rdm
        conf.pattern_descriptor = pattern_descriptor
        conf.rdm_descriptor = rdm_descriptor or ''
        conf.dissimilarity_measure = rdm.dissimilarity_measure or ''
        return conf

    def for_single(self, index: int) -> SingleRdmPlot:
        """Create a SingleRdmPlot object for the given rdm index

        Args:
            index (int): Index for the rdms

        Returns:
            SingleRdmPlot: _description_
        """
        conf = SingleRdmPlot()
        conf.rdms = self.rdms[index]
        conf.cmap = self.cmap
        conf.rdm_descriptor = self.rdm_descriptor
        conf.gridlines = self.gridlines
        conf.nanmask = self.nanmask
        conf.vmin = self.vmin
        conf.vmax = self.vmax
        if self.rdm_descriptor in conf.rdms.rdm_descriptors:
            conf.title = conf.rdms.rdm_descriptors[self.rdm_descriptor][0]
        else:
            conf.title = self.rdm_descriptor
        return conf

class SingleRdmPlot:
    """Configuration for the single-rdm plot
    """

    rdms: RDMs
    cmap: Union[str, Colormap]
    rdm_descriptor: str
    gridlines: ArrayLike
    nanmask: NDArray
    vmin: Optional[float]
    vmax: Optional[float]
    title: str

    @classmethod
    def from_show_rdm_panel_args(
        cls,
        rdms: RDMs,
        cmap: Union[str, Colormap] = 'bone',
        nanmask: Optional[NDArray] = None,
        rdm_descriptor: Optional[str] = None,
        gridlines: Optional[npt.ArrayLike] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> SingleRdmPlot:
        """Create an object from the original arguments to show_rdm_panel()
        """
        conf = __class__()
        if rdms.n_rdm > 1:
            raise ValueError("expected single rdm - use show_rdm for multi-panel figures")
        if cmap == 'classic':
            cmap = rdm_colormap_classic()
        conf.cmap = cmap
        if nanmask is None:
            nanmask = np.eye(rdms.n_cond, dtype=bool)
        conf.nanmask = nanmask
        gridlines = gridlines or list()
        if not np.any(gridlines):
            gridlines = []
        conf.gridlines = gridlines
        if rdm_descriptor in rdms.rdm_descriptors:
            conf.title = rdms.rdm_descriptors[rdm_descriptor][0]
        else:
            conf.title = rdm_descriptor or ''
        conf.vmin = vmin
        conf.vmax = vmax
        return conf
