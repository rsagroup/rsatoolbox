from __future__ import annotations
from typing import TYPE_CHECKING, Optional
import matplotlib.pyplot as plt
if TYPE_CHECKING:
    from rsatoolbox.rdm import RDMs
    from numpy.typing import NDArray
    from matplotlib.figure import Figure


def show_scatter(rdms: RDMs, coords: NDArray, pattern_descriptor: Optional[str]) -> Figure:
    """Draw a 2-dimensional scatter plot based on the provided coordinates

    Args:
        rdms (RDMs): The RDMs object to display
        coords (NDArray): Array of x and y coordinates for each
            pattern (patterns x 2)
        pattern_descriptor (Optional[str]): If provided, the chosen pattern 
            descriptor will be printed adjacent to each point in the plot

    Returns:
        Figure: A matplotlib figure in which the plot is drawn
    """
    fig, ax = plt.subplots()
    ax.scatter(coords[0, :, 0], coords[0, :, 1])

    if pattern_descriptor is not None:
        for p in range(coords.shape[1]):
            ax.annotate(
                rdms.pattern_descriptors[pattern_descriptor][p],
                (coords[0, p, 0], coords[0, p, 1])
            )

    ## turn off all axis ticks and labels
    ax.tick_params(axis='both', which='both', bottom=False, top=False,
        right=False, left=False, labelbottom=False, labeltop=False, 
        labelleft=False, labelright=False)
    return fig
