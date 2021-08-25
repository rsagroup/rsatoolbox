from __future__ import annotations
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
if TYPE_CHECKING:
    from rsatoolbox.rdm import RDMs
    from numpy.typing import NDArray
    from matplotlib.figure import Figure


def show_scatter(rdms: RDMs, coords: NDArray) -> Figure:
    fig, ax = plt.subplots()
    ax.scatter(coords[0, :, 0], coords[0, :, 1])
    return fig