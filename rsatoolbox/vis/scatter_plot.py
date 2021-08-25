from __future__ import annotations
from typing import TYPE_CHECKING, Optional
import matplotlib.pyplot as plt
if TYPE_CHECKING:
    from rsatoolbox.rdm import RDMs
    from numpy.typing import NDArray
    from matplotlib.figure import Figure


def show_scatter(rdms: RDMs, coords: NDArray, pattern_descriptor: Optional[str]) -> Figure:
    fig, ax = plt.subplots()
    ax.scatter(coords[0, :, 0], coords[0, :, 1])

    if pattern_descriptor is not None:
        for p in range(coords.shape[1]):
            ax.annotate(
                rdms.pattern_descriptors[pattern_descriptor][p],
                (coords[0, p, 0], coords[0, p, 1])
            )
    return fig
