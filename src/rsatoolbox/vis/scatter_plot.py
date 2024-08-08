from __future__ import annotations
from typing import TYPE_CHECKING, Optional
import math
import matplotlib.pyplot
import sklearn.manifold
import numpy
from rsatoolbox.util.vis_utils import weight_to_matrices, Weighted_MDS
from rsatoolbox.vis.icon import Icon
if TYPE_CHECKING:
    from rsatoolbox.rdm import RDMs
    from numpy.typing import NDArray
    from matplotlib.figure import Figure
seed = numpy.random.RandomState(seed=1)


def show_scatter(
        rdms: RDMs,
        coords: NDArray,
        rdm_descriptor: Optional[str]=None,
        pattern_descriptor: Optional[str]=None,
        icon_size: float=0.1
    ) -> Figure:
    """Draw a 2-dimensional scatter plot based on the provided coordinates

    Args:
        rdms (RDMs): The RDMs object to display
        coords (NDArray): Array of x and y coordinates for each
            pattern (patterns x 2)
        rdm_descriptor: (Optional[str]): If provided, this will be used as
            title for each individual RDM.
        pattern_descriptor (Optional[str]): If provided, the chosen pattern
            descriptor will be printed adjacent to each point in the plot
        icon_size: relative size of icons if the pattern descriptor chosen
            is of type Icon

    Returns:
        Figure: A matplotlib figure in which the plot is drawn
    """
    frac, n = math.modf(math.sqrt(rdms.n_rdm))
    nrows, ncols = math.floor(n), math.floor(n)
    if frac > 0:
        nrows += 1
    if frac > 0.5:
        ncols += 1
    fig, axes = matplotlib.pyplot.subplots(nrows=nrows, ncols=ncols)
    axes = numpy.array(axes)  ## it's now an array even if there's only one
    for r, ax in enumerate(axes.ravel()):

        if r > (rdms.n_rdm - 1):
            ## fewer rdms than rows x cols, hide the remaining axes
            ax.axis('off')
            break

        ax.scatter(coords[r, :, 0], coords[r, :, 1])
        ax.set_xlim(coords.min()*0.95, coords.max()*1.05)
        ax.set_ylim(coords.min()*0.95, coords.max()*1.05)

        ## RDM names
        if rdm_descriptor is not None:
            ax.set_title(rdms.rdm_descriptors[rdm_descriptor][r])

        ## print labels next to dots
        if pattern_descriptor is not None:
            for p in range(coords.shape[1]):
                pat_desc = rdms.pattern_descriptors[pattern_descriptor][p]
                pat_coords = (coords[r, p, 0], coords[r, p, 1])
                if isinstance(pat_desc, Icon):
                    pat_desc.plot(pat_coords[0], pat_coords[1], ax=ax, size=icon_size)
                else:
                    label = ax.annotate(pat_desc, pat_coords)
                    label.set_alpha(.6)

        ## turn off all axis ticks and labels
        ax.tick_params(axis='both', which='both', bottom=False, top=False,
            right=False, left=False, labelbottom=False, labeltop=False,
            labelleft=False, labelright=False)
    return fig

def show_2d(
        rdms: RDMs,
        method: str,
        weights: Optional[NDArray]=None,
        rdm_descriptor: Optional[str]=None,
        pattern_descriptor: Optional[str]=None,
        icon_size: float=0.1
    ) -> Figure:
    """Draw a scatter plot of the RDMs reduced to two dimensions

    Args:
        rdms (RDMs): The RDMs object to display
        method (str): One of 'MDS', 't-SNE', 'Isomap'.
        weights: Optional array of weights (vector per RDM)
        rdm_descriptor: (Optional[str]): If provided, this will be used as
            title for each individual RDM.
        pattern_descriptor (Optional[str]): If provided, the chosen pattern
            descriptor will be printed adjacent to each point in the plot
        icon_size: relative size of icons if the pattern descriptor chosen
            is of type Icon

    Returns:
        Figure: A matplotlib figure in which the plot is drawn
    """
    if method == 'MDS':
        MDS = sklearn.manifold.MDS if weights is None else Weighted_MDS
        embedding = MDS(
            n_components=2,
            random_state=seed,
            dissimilarity='precomputed',
            normalized_stress='auto',
        )
    elif method == 't-SNE':
        embedding = sklearn.manifold.TSNE(n_components=2)
    elif method == 'Isomap':
        embedding = sklearn.manifold.Isomap(n_components=2)
    else:
        raise NotImplementedError('Unknown method: ' + str(method))
    rdm_mats = rdms.get_matrices()
    coords = numpy.full((rdms.n_rdm, rdms.n_cond, 2), numpy.nan)
    for r in range(rdms.n_rdm):
        fitKwargs = dict()
        if weights is not None:
            fitKwargs['weight'] = weight_to_matrices(weights)[r, :, :]
        coords[r, :, :] = embedding.fit_transform(rdm_mats[r, :, :], **fitKwargs)
    return show_scatter(
        rdms,
        coords,
        rdm_descriptor=rdm_descriptor,
        pattern_descriptor=pattern_descriptor,
        icon_size=icon_size
    )

def show_MDS(
        rdms: RDMs,
        weights: Optional[NDArray]=None,
        rdm_descriptor: Optional[str]=None,
        pattern_descriptor: Optional[str]=None,
        icon_size: float=0.1
    ) -> Figure:
    """Draw a scatter plot based on Multidimensional Scaling dimensionality reduction

    Args:
        rdms (RDMs): The RDMs object to display
        weights: Optional array of weights (vector per RDM)
        rdm_descriptor: (Optional[str]): If provided, this will be used as
            title for each individual RDM.
        pattern_descriptor (Optional[str]): If provided, the chosen pattern
            descriptor will be printed adjacent to each point in the plot
        icon_size: relative size of icons if the pattern descriptor chosen
            is of type Icon

    Returns:
        Figure: A matplotlib figure in which the plot is drawn
    """
    return show_2d(
        rdms,
        method='MDS',
        weights=weights,
        rdm_descriptor=rdm_descriptor,
        pattern_descriptor=pattern_descriptor,
        icon_size=icon_size
    )

def show_tSNE(
        rdms: RDMs,
        rdm_descriptor: Optional[str]=None,
        pattern_descriptor: Optional[str]=None,
        icon_size: float=0.1
    ) -> Figure:
    """Draw a scatter plot based on t-SNE dimensionality reduction

    Args:
        rdms (RDMs): The RDMs object to display
        rdm_descriptor: (Optional[str]): If provided, this will be used as
            title for each individual RDM.
        pattern_descriptor (Optional[str]): If provided, the chosen pattern
            descriptor will be printed adjacent to each point in the plot
        icon_size: relative size of icons if the pattern descriptor chosen
            is of type Icon

    Returns:
        Figure: A matplotlib figure in which the plot is drawn
    """
    return show_2d(
        rdms,
        method='t-SNE',
        rdm_descriptor=rdm_descriptor,
        pattern_descriptor=pattern_descriptor,
        icon_size=icon_size
    )

def show_iso(
        rdms: RDMs,
        rdm_descriptor: Optional[str]=None,
        pattern_descriptor: Optional[str]=None,
        icon_size: float=0.1
    ) -> Figure:
    """Draw a scatter plot based on Isomap dimensionality reduction

    Args:
        rdms (RDMs): The RDMs object to display
        rdm_descriptor: (Optional[str]): If provided, this will be used as
            title for each individual RDM.
        pattern_descriptor (Optional[str]): If provided, the chosen pattern
            descriptor will be printed adjacent to each point in the plot
        icon_size: relative size of icons if the pattern descriptor chosen
            is of type Icon

    Returns:
        Figure: A matplotlib figure in which the plot is drawn
    """
    return show_2d(
        rdms,
        method='Isomap',
        rdm_descriptor=rdm_descriptor,
        pattern_descriptor=pattern_descriptor,
        icon_size=icon_size
    )
