from __future__ import annotations
from typing import TYPE_CHECKING, Optional
import matplotlib.pyplot
import sklearn.manifold
import numpy
from rsatoolbox.util.vis_utils import weight_to_matrices, Weighted_MDS
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

    Returns:
        Figure: A matplotlib figure in which the plot is drawn
    """
    fig, ax = matplotlib.pyplot.subplots()
    ax.scatter(coords[0, :, 0], coords[0, :, 1])

    ## RDM names
    if rdm_descriptor is not None:
        ax.set_title(rdms.rdm_descriptors[rdm_descriptor][0])

    ## print labels next to dots
    if pattern_descriptor is not None:
        for p in range(coords.shape[1]):
            label = ax.annotate(
                rdms.pattern_descriptors[pattern_descriptor][p],
                (coords[0, p, 0], coords[0, p, 1])
            )
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
        pattern_descriptor: Optional[str]=None
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

    Returns:
        Figure: A matplotlib figure in which the plot is drawn
    """
    if method == 'MDS':
        MDS = sklearn.manifold.MDS if weights is None else Weighted_MDS
        embedding = MDS(n_components=2, random_state=seed, dissimilarity='precomputed')
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
        pattern_descriptor=pattern_descriptor
    )
    
def show_MDS(
        rdms: RDMs,
        weights: Optional[NDArray]=None,
        rdm_descriptor: Optional[str]=None,
        pattern_descriptor: Optional[str]=None
    ) -> Figure:
    """Draw a scatter plot based on Multidimensional Scaling dimensionality reduction

    Args:
        rdms (RDMs): The RDMs object to display
        weights: Optional array of weights (vector per RDM)
        rdm_descriptor: (Optional[str]): If provided, this will be used as
            title for each individual RDM.
        pattern_descriptor (Optional[str]): If provided, the chosen pattern
            descriptor will be printed adjacent to each point in the plot

    Returns:
        Figure: A matplotlib figure in which the plot is drawn
    """
    return show_2d(
        rdms,
        method='MDS',
        weights=weights,
        rdm_descriptor=rdm_descriptor,
        pattern_descriptor=pattern_descriptor
    )

def show_tSNE(
        rdms: RDMs,
        rdm_descriptor: Optional[str]=None,
        pattern_descriptor: Optional[str]=None
    ) -> Figure:
    """Draw a scatter plot based on t-SNE dimensionality reduction

    Args:
        rdms (RDMs): The RDMs object to display
        rdm_descriptor: (Optional[str]): If provided, this will be used as
            title for each individual RDM.
        pattern_descriptor (Optional[str]): If provided, the chosen pattern
            descriptor will be printed adjacent to each point in the plot

    Returns:
        Figure: A matplotlib figure in which the plot is drawn
    """
    return show_2d(
        rdms,
        method='t-SNE',
        rdm_descriptor=rdm_descriptor,
        pattern_descriptor=pattern_descriptor
    )

def show_iso(
        rdms: RDMs,
        rdm_descriptor: Optional[str]=None,
        pattern_descriptor: Optional[str]=None
    ) -> Figure:
    """Draw a scatter plot based on Isomap dimensionality reduction

    Args:
        rdms (RDMs): The RDMs object to display
        rdm_descriptor: (Optional[str]): If provided, this will be used as
            title for each individual RDM.
        pattern_descriptor (Optional[str]): If provided, the chosen pattern
            descriptor will be printed adjacent to each point in the plot

    Returns:
        Figure: A matplotlib figure in which the plot is drawn
    """
    return show_2d(
        rdms,
        method='Isomap',
        rdm_descriptor=rdm_descriptor,
        pattern_descriptor=pattern_descriptor
    )
