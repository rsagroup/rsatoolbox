"""
Plot showing an RDMs object

public API:

- show_rdm()
- show_rdm_panel()
"""
from __future__ import annotations
import itertools
from pathlib import Path
from typing import TYPE_CHECKING, Union, Tuple, Optional, Literal, Dict, Any, List, Iterator
from enum import Enum
from math import ceil
import numpy as np
from scipy.spatial.distance import squareform
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon
import rsatoolbox.rdm
from rsatoolbox.rdm.rdms import RDMs
from rsatoolbox import vis
from rsatoolbox.vis.colors import rdm_colormap_classic
from rsatoolbox.resources import get_style
if TYPE_CHECKING:
    from matplotlib.axes._axes import Axes
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Colormap
    from matplotlib.colorbar import Colorbar
    from matplotlib.figure import Figure
    from matplotlib.text import Text
    from matplotlib.image import AxesImage
    from matplotlib.axis import XAxis, YAxis
    from numpy.typing import NDArray, ArrayLike
    ArrayOrRdmDescriptor = NDArray | Tuple[str, str]


class Axis(Enum):
    """X or Y axis Enum
    """
    X = 'x'
    Y = 'y'


class Symmetry(Enum):
    """RDM Triangle Enum: both, upper or lower
    """
    BOTH = 'both'
    UPPER = 'upper'
    LOWER = 'lower'


def show_rdm(
    rdms: rsatoolbox.rdm.RDMs,
    pattern_descriptor: Optional[str] = None,
    cmap: Union[str, Colormap] = 'bone_r',
    rdm_descriptor: Optional[str] = None,
    n_column: Optional[int] = None,
    n_row: Optional[int] = None,
    show_colorbar: Optional[str] = None,
    gridlines: Optional[ArrayLike] = None,
    num_pattern_groups: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None,
    nanmask: NDArray | str | None = "diagonal",
    style: Optional[Union[str, Path]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    icon_spacing: float = 1.0,
    linewidth: float = 0.5,
    overlay: Optional[ArrayOrRdmDescriptor] = None,
    overlay_color: str = '#00ff0050',
    overlay_symmetry: Symmetry = Symmetry.BOTH,
    contour: Optional[ArrayOrRdmDescriptor] = None,
    contour_color: str = 'red',
    contour_symmetry: Symmetry = Symmetry.BOTH,
) -> Tuple[Figure, NDArray, Dict[int, Dict[str, Any]]]:
    """show_rdm. Heatmap figure for RDMs instance, with one panel per RDM.

    Args:
        rdm (rsatoolbox.rdm.RDMs): RDMs object to be plotted.
        pattern_descriptor (str): Key into rdm.pattern_descriptors to use for axis
            labels.
        cmap (str or Colormap): Colormap to be used.
            Either the name of a Matplotlib built-in colormap, a Matplotlib
            Colormap compatible object, or 'classic' for the matlab toolbox
            colormap. Defaults to 'bone_r'.
        rdm_descriptor (str): Key for rdm_descriptor to use as panel title, or
            str for direct labeling.
        n_column (int): Number of columns in subplot arrangement.
        n_row (int): Number of rows in subplot arrangement.
        show_colorbar (str): Set to 'panel' or 'figure' to display a colorbar. If
            'panel' a colorbar is added next to each RDM. If 'figure' a shared colorbar
            (and scale) is used across panels.
        gridlines (ArrayLike): Set to add gridlines at these positions. If
            num_pattern_groups is defined this is used to infer gridlines.
        num_pattern_groups (int): Number of rows/columns for any image labels. Also
            determines gridlines frequency by default (so e.g., num_pattern_groups=3
            with results in gridlines every 3 rows/columns).
        figsize (Tuple[float, float]): mpl.Figure argument. By default we
            auto-scale to achieve a figure that fits on a standard A4 / US Letter page
            in portrait orientation.
        nanmask (Union[ArrayLike, str, None]): boolean mask defining RDM elements to suppress
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
        overlay ((str, str) or NDArray): RDM descriptor name-value tuple,
            or vector (one value per pair) which indicates whether to highlight the given cells
        overlay_color (str): Color to use to highlight the pairs in the overlay argument.
            Use RGBA to specify transparency. Default is 50% opaque green.
        contour ((str, str) or NDArray): RDM descriptor name-value tuple,
            or vector (one value per pair) which indicates whether to add a border
            to the given cells
        contour_color (str): Color to use for a border around pairs in the contour argument.
            Use RGBA to specify transparency. Default is red.

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
        style, vmin, vmax, icon_spacing, linewidth, overlay, overlay_color,
        overlay_symmetry, contour, contour_color, contour_symmetry
    )
    return _plot_multi_rdm(conf)


def _plot_multi_rdm(conf: MultiRdmPlot) -> Tuple[Figure, NDArray, Dict[int, Dict[str, Any]]]:
    # A dictionary of figure element handles
    handles = dict()
    handles[-1] = dict()  # fig level handles
    # create a list of (row index, column index) tuples
    rc_tuples = list(itertools.product(range(conf.n_row), range(conf.n_column)))
    # number of empty panels at the top
    n_empty = (conf.n_row * conf.n_column) - conf.rdms.n_rdm
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
            rdm_index = p - n_empty  # rdm index
            if rdm_index < 0:
                ax_array[r, c].set_visible(False)
                continue

            handles[p]["image"] = _show_rdm_panel(conf.for_single(rdm_index), ax_array[r, c])

            if conf.show_colorbar == "panel":
                # needs to happen before labels because it resizes the axis
                handles[p]["colorbar"] = _rdm_colorbar(
                    mappable=handles[p]["image"],
                    fig=fig,
                    ax=ax_array[r, c],
                    title=conf.dissimilarity_measure
                )
            if c == 0 and conf.pattern_descriptor:
                handles[p]["y_labels"] = _add_descriptor_labels(Axis.Y, ax_array[r, c], conf)
            if r == 0 and conf.pattern_descriptor:
                handles[p]["x_labels"] = _add_descriptor_labels(Axis.X, ax_array[r, c], conf)

        if conf.show_colorbar == "figure":
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
    # parent = ax_array[-1, -1]
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
    cmap: Union[str, Colormap] = 'bone_r',
    nanmask: Optional[NDArray] = None,
    rdm_descriptor: Optional[str] = None,
    gridlines: Optional[ArrayLike] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    overlay: Optional[NDArray] = None,
    overlay_color: str = '#00ff0050',
    overlay_symmetry: Symmetry = Symmetry.BOTH,
    contour: Optional[NDArray] = None,
    contour_color: str = 'red',
    contour_symmetry: Symmetry = Symmetry.BOTH
) -> AxesImage:
    """show_rdm_panel. Add RDM heatmap to the axis ax.

    Args:
        rdm (rsatoolbox.rdm.RDMs): RDMs object to be plotted (n_rdm must be 1).
        ax (matplotlib.axes._axes.Axes): Matplotlib axis handle. plt.gca() by default.
        cmap (str or Colormap): Colormap to be used.
            Either the name of a Matplotlib built-in colormap, a Matplotlib
            Colormap compatible object, or 'classic' for the matlab toolbox
            colormap. Defaults to 'bone_r'.
        nanmask (ArrayLike): boolean mask defining RDM elements to suppress
            (by default, the diagonals).
        rdm_descriptor (str): Key for rdm_descriptor to use as panel title, or
            str for direct labeling.
        gridlines (ArrayLike): Set to add gridlines at these positions.
        vmin (float): Minimum intensity for colorbar mapping. matplotlib imshow
            argument.
        vmax (float): Maximum intensity for colorbar mapping. matplotlib imshow
            argument.
        overlay ((str, str) or NDArray): RDM descriptor name-value tuple, or vector
            (one value per pair) which indicates whether to highlight the given cells
        overlay_color (str): Color to use to highlight the pairs in the overlay argument.
            Use RGBA to specify transparency. Default is 50% opaque green.
        contour ((str, str) or NDArray): RDM descriptor name-value tuple, or vector
            (one value per pair) which indicates whether to add a border to the given cells
        contour_color (str): Color to use for a border around pairs in the contour argument.
            Use RGBA to specify transparency. Default is red.

    Returns:
        matplotlib.image.AxesImage: Matplotlib handle.
    """
    conf = SingleRdmPlot.from_show_rdm_panel_args(
        rdms, cmap, nanmask,
        rdm_descriptor, gridlines, vmin, vmax, overlay, overlay_color,
        overlay_symmetry, contour, contour_color, contour_symmetry
    )
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
    image = ax.imshow(
        rdmat, cmap=conf.cmap, vmin=conf.vmin, vmax=conf.vmax,
        interpolation='none')
    _overlay(conf, ax)
    _contour(conf, ax)
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


def _overlay(conf: SingleRdmPlot, ax: Axes) -> None:
    """Add overlay image to axes

    Args:
        conf (SingleRdmPlot): Plot configuration
        ax (Axes): axes object for this plot
    """
    if not np.any(conf.overlay_mask):
        return
    cmap = ListedColormap(['none', conf.overlay_color])
    ax.imshow(conf.overlay_mask, cmap=cmap, interpolation='none')


def _contour(conf: SingleRdmPlot, ax: Axes) -> None:
    """Add contour outline to axes

    Args:
        conf (SingleRdmPlot): Plot configuration
        ax (Axes): axes object for this plot
    """
    if not np.any(conf.contour_mask):
        return
    for (x1, y1, x2, y2) in _contour_coords(conf.contour_mask, -0.5):
        ax.add_patch(
            Polygon(
                [(x1, y1),  (x2, y2)],
                facecolor='none',
                edgecolor=conf.contour_color,
                linewidth=3,
                closed=True,
                joinstyle='round'
            )
        )


def _mask_from_vector(vector: NDArray, triangles: Symmetry) -> NDArray:
    """Turn a triangular vector into a matrix mask, with given symmetry

    Returns:
        NDArray: 2-D boolean matrix
    """
    mask = squareform(vector)
    if triangles == Symmetry.BOTH:
        return mask
    elif triangles == Symmetry.LOWER:
        return np.tril(mask)
    elif triangles == Symmetry.UPPER:
        return np.triu(mask)


def _contour_coords(mask: NDArray, offset: float) -> Iterator[Tuple[float, float, float, float]]:
    """Determine filled edges for the given mask

    Returns a tuple of x1, y1, x2, y2 coordinates for each line.

    Args:
        mask (NDArray): nconds x nconds mask
        offset (float): value to add for matplotlib indexing

    Yields:
        Iterator[Tuple[float, float, float, float]]: coordinates
    """
    mask_t = mask.T
    mask_idx = np.where(mask_t)
    sides = [
        ((0, -1), (0, 0, 1, 0)),  # top
        ((1,  0), (1, 0, 1, 1)),  # right
        ((0,  1), (1, 1, 0, 1)),  # bottom
        ((-1,  0), (0, 1, 0, 0)),  # left
    ]
    for x, y in np.vstack(mask_idx).T:
        for neighbor, edge in sides:
            if not mask_t[(x+neighbor[0], y+neighbor[1])]:
                x1, y1, x2, y2 = edge
                yield (x+x1+offset, y+y1+offset, x+x2+offset, y+y2+offset)


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
        horizontalalignment = "center"
    else:
        icon_method = "y_tick_label"
        axis = ax.yaxis
        horizontalalignment = "right"
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
    descriptor_arr: ArrayLike,
    axis: Union[XAxis, YAxis],
    horizontalalignment: str = "center",
    is_x_axis: bool = False,
) -> List[Text]:
    """_add_descriptor_text. Used internally by _add_descriptor_labels to add vanilla
    Matplotlib-based text labels to the X or Y axis.

    Args:
        descriptor_arr (ArrayLike): np.Array-like version of the labels.
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
    descriptor_arr: ArrayLike,
    icon_method: str,
    n_cond: int,
    ax: Optional[Axes] = None,
    num_pattern_groups: Optional[int] = None,
    icon_spacing: float = 1.0,
    linewidth: float = 0.5,
) -> list:
    """_add_descriptor_icons. Used internally by _add_descriptor_labels to add
    Icon-based labels to the X or Y axis.

    Args:
        descriptor_arr (ArrayLike): np.Array-like version of the labels.
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
    offset = (im_max_pix / icon_spacing) * size
    label_handles = []
    for group_ind in range(num_pattern_groups - 1, -1, -1):  # e.g. 2->1->0 for npg = 3
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
    overlay: NDArray
    overlay_color: str
    overlay_symmetry: Symmetry
    contour: NDArray
    contour_color: str
    contour_symmetry: Symmetry
    overlay_mask: NDArray
    contour_mask: NDArray

    fig: Optional[Figure]
    ax: Optional[NDArray]
    handles: Optional[Dict[int, Dict[str, Any]]]

    @classmethod
    def from_show_rdm_args(
        cls,
        rdm: RDMs,
        pattern_descriptor: Optional[str],
        cmap: Union[str, Colormap],
        rdm_descriptor: Optional[str],
        n_column: Optional[int],
        n_row: Optional[int],
        show_colorbar: Optional[str],
        gridlines: Optional[ArrayLike],
        num_pattern_groups: Optional[int],
        figsize: Optional[Tuple[float, float]],
        nanmask: NDArray | str | None,
        style: Optional[Union[str, Path]],
        vmin: Optional[float],
        vmax: Optional[float],
        icon_spacing: float,
        linewidth: float,
        overlay: Optional[Tuple[str, str] | NDArray],
        overlay_color: str,
        overlay_symmetry: Symmetry,
        contour: Optional[Tuple[str, str] | NDArray],
        contour_color: str,
        contour_symmetry: Symmetry
    ) -> MultiRdmPlot:
        """Create an object from the original arguments to show_rdm()
        """
        conf = __class__(rdm)
        if show_colorbar not in (None, "panel", "figure"):
            raise ValueError(
                f"show_colorbar can be None, panel or figure, got: {show_colorbar}"
            )
        conf.show_colorbar = show_colorbar
        conf.nanmask = cls.init_nan_mask(nanmask, rdm)
        conf.n_panel = rdm.n_rdm + int(show_colorbar == "figure")
        if show_colorbar == "figure":
            rdmat = rdm.get_matrices()
            vmin = vmin or rdmat[:, ~conf.nanmask].min()
            vmax = vmax or rdmat[:, ~conf.nanmask].max()
        conf.vmin = vmin
        conf.vmax = vmax
        conf.n_row, conf.n_column = cls.determine_rows_cols_panels(
            n_row, n_column, conf.n_panel)

        conf.figsize = figsize or cls.calc_figsize(conf.n_column, conf.n_row)
        gridlines = np.asarray(gridlines or list())
        if num_pattern_groups and (not np.any(gridlines)):
            # grid by pattern groups if they exist and explicit grid setting does not
            gridlines = np.arange(
                num_pattern_groups - 0.5, rdm.n_cond + 0.5, num_pattern_groups
            )
        conf.gridlines = np.asarray(gridlines)
        if num_pattern_groups is None or num_pattern_groups == 0:
            num_pattern_groups = 1
        conf.num_pattern_groups = num_pattern_groups
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
        conf.overlay = conf.interpret_rdm_arg(overlay, rdm)
        conf.overlay_color = overlay_color
        conf.overlay_symmetry = overlay_symmetry
        conf.overlay_mask = _mask_from_vector(conf.overlay, conf.overlay_symmetry)
        conf.contour = conf.interpret_rdm_arg(contour, rdm)
        conf.contour_color = contour_color
        conf.contour_symmetry = contour_symmetry
        conf.contour_mask = _mask_from_vector(conf.contour, conf.contour_symmetry)
        return conf

    def interpret_rdm_arg(self, val: Optional[ArrayOrRdmDescriptor], rdms: RDMs) -> NDArray:
        """Resolve argument that can be an rdm descriptor key/value pair or a utv
        """
        if val is None:
            n_pairs = rdms.dissimilarities.shape[1]
            return np.zeros(n_pairs)
        if isinstance(val, np.ndarray):
            return val
        else:
            return rdms.subset(*val).dissimilarities[0, :]

    @classmethod
    def determine_rows_cols_panels(
            cls,
            n_row: Optional[int],
            n_column: Optional[int],
            n_panel: int
    ) -> Tuple[int, int]:
        """Choose the number of rows and columns of panels
        """
        if (n_column is None) and (n_row is None):
            n_column = ceil(np.sqrt(n_panel))
        if n_row is None:
            n_row = ceil(n_panel / n_column)
        if n_column is None:
            n_column = ceil(n_panel / n_row)
        return n_row, n_column

    @classmethod
    def init_nan_mask(
            cls,
            nanmask: NDArray | str | None,
            rdms: RDMs,
    ) -> NDArray:
        """Interpret user's choice of nanmask
        """
        if nanmask is None:
            nanmask = np.zeros((rdms.n_cond, rdms.n_cond), dtype=bool)
        elif isinstance(nanmask, str):
            if nanmask == "diagonal":
                nanmask = np.eye(rdms.n_cond, dtype=bool)
            else:
                raise ValueError("Invalid nanmask value")
        return nanmask

    @classmethod
    def calc_figsize(cls, n_column: int, n_row: int) -> Tuple[float, float]:
        """"
         scale with number of RDMs, up to (intersection of A4 and us letter)
        """
        return (
            min(2 * n_column, 8.3), min(2 * n_row, 11)
        )

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
        conf.overlay = self.overlay
        conf.overlay_mask = self.overlay_mask
        conf.overlay_color = self.overlay_color
        conf.overlay_symmetry = self.overlay_symmetry
        conf.contour = self.contour
        conf.contour_mask = self.contour_mask
        conf.contour_color = self.contour_color
        conf.contour_symmetry = self.contour_symmetry
        if self.rdm_descriptor in conf.rdms.rdm_descriptors:
            conf.title = conf.rdms.rdm_descriptors[self.rdm_descriptor][0]
        else:
            conf.title = self.rdm_descriptor
        return conf

    def __init__(self, rdms: RDMs):
        self.rdms = rdms
        self.pattern_descriptor = None
        self.cmap = 'bone_r'
        self.rdm_descriptor = ''
        self.gridlines = np.array([])
        self.num_pattern_groups = 1
        self.show_colorbar = None
        self.n_row, self.n_column = self.determine_rows_cols_panels(
            None, None, self.rdms.n_rdm)
        self.figsize = self.calc_figsize(self.n_column, self.n_row)
        self.nanmask = self.init_nan_mask('diagonal', self.rdms)
        self.style = get_style()
        self.vmin = None
        self.vmax = None
        self.icon_spacing = 1.0
        self.linewidth = 0.5
        n_pairs = rdms.dissimilarities.shape[1]
        self.overlay = np.zeros(n_pairs)
        self.overlay_color = '#00ff0050'
        self.overlay_symmetry = Symmetry.BOTH
        self.contour = np.zeros(n_pairs)
        self.contour_color = 'red'
        self.contour_symmetry = Symmetry.BOTH

    def addOverlay(self, mask: ArrayOrRdmDescriptor, color: str, triangles: Symmetry):
        self.overlay = self.interpret_rdm_arg(mask, self.rdms)
        self.overlay_color = color
        self.overlay_symmetry = triangles
        self.overlay_mask = _mask_from_vector(self.overlay, triangles)

    def addContour(self, mask: ArrayOrRdmDescriptor, color: str, triangles: Symmetry):
        self.contour = self.interpret_rdm_arg(mask, self.rdms)
        self.contour_color = color
        self.contour_symmetry = triangles
        self.contour_mask = _mask_from_vector(self.contour, triangles)

    def plot(self):
        self.fig, self.ax, self.handles = _plot_multi_rdm(self)
        return self.fig


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
    overlay: NDArray
    overlay_color: str
    overlay_symmetry: Symmetry
    contour: NDArray
    contour_color: str
    contour_symmetry: Symmetry
    overlay_mask: NDArray
    contour_mask: NDArray

    fig: Optional[Figure]
    ax: Optional[NDArray]
    handles: Optional[Dict[int, Dict[str, Any]]]

    @classmethod
    def from_show_rdm_panel_args(
        cls,
        rdms: RDMs,
        cmap: Union[str, Colormap],
        nanmask: Optional[NDArray],
        rdm_descriptor: Optional[str],
        gridlines: Optional[ArrayLike],
        vmin: Optional[float],
        vmax: Optional[float],
        overlay: Optional[NDArray],
        overlay_color: str,
        overlay_symmetry: Symmetry,
        contour: Optional[NDArray],
        contour_color: str,
        contour_symmetry: Symmetry
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
        conf.overlay = conf.interpret_rdm_arg(overlay, rdms)
        conf.overlay_color = overlay_color
        conf.overlay_symmetry = overlay_symmetry
        conf.overlay_mask = _mask_from_vector(conf.overlay, conf.overlay_symmetry)
        conf.contour = conf.interpret_rdm_arg(contour, rdms)
        conf.contour_color = contour_color
        conf.contour_symmetry = contour_symmetry
        conf.contour_mask = _mask_from_vector(conf.contour, conf.contour_symmetry)
        return conf

    def interpret_rdm_arg(self, val: Optional[ArrayOrRdmDescriptor], rdms: RDMs) -> NDArray:
        """Resolve argument that can be an rdm descriptor key/value pair or a utv
        """
        if val is None:
            n_pairs = rdms.dissimilarities.shape[1]
            return np.zeros(n_pairs)
        if isinstance(val, np.ndarray):
            return val
        else:
            return rdms.subset(*val).dissimilarities[0, :]
