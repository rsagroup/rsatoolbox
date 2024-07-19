"""Lineplot of dissimilarity over time

See demo_meg_mne for an example.
"""
# pylint: disable=too-many-statements,unused-argument,too-many-locals
from __future__ import annotations
from typing import TYPE_CHECKING, Tuple, List, Optional, Dict
import matplotlib.pyplot as plt
import numpy as np
if TYPE_CHECKING:
    from rsatoolbox.rdm.rdms import RDMs
    from matplotlib.axes._axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


def plot_timecourse(
        rdms_data: RDMs,
        descriptor: str,
        n_t_display: int = 20,
        fig_width: Optional[int] = None,
        timecourse_plot_rel_height: Optional[int] = None,
        time_formatted: Optional[List[str]] = None,
        colored_conditions: Optional[list] = None,
        plot_individual_dissimilarities: Optional[bool] = None,
) -> Tuple[Figure, List[Axes]]:
    """ plots the RDM movie for a given descriptor

    Args:
        rdms_data (rsatoolbox.rdm.RDMs): rdm movie
        descriptor (str): name of the descriptor that created the rdm movie
        n_t_display (int, optional): number of RDM time points to display. Defaults to 20.
        fig_width (int, optional):  width of the figure (in inches). Defaults to None.
        timecourse_plot_rel_height (int, optional): height of the timecourse plot (relative to
            the rdm movie row).
        time_formatted (List[str], optional): time points formatted as strings.
            Defaults to None (i.e., rdms_data.time_descriptors['time'] is considered to
            be in seconds)
        colored_condiitons (list, optional): vector of pattern condition names to dissimilarities
        according to a categorical model on colored_conditions Defaults to None.
        plot_individual_dissimilarities (bool, optional): whether to plot the individual
            dissimilarities. Defaults to None (i.e., False if colored_conditions is not
            None, True otherwise).

    Returns:
        Tuple[matplotlib.figure.Figure, npt.ArrayLike, collections.defaultdict]:

        Tuple of
            - Handle to created figure
            - Subplot axis handles from plt.subplots.
    """
    # create labels
    time = rdms_data.rdm_descriptors['time']
    unique_time = np.unique(time)
    time_formatted = time_formatted or [f'{np.round(x*1000,2):0.0f} ms' for x in unique_time]

    n_dissimilarity_elements = rdms_data.dissimilarities.shape[1]

    # color mapping from colored conditions
    plot_individual_dissimilarities, color_index = _map_colors(
        colored_conditions, plot_individual_dissimilarities, rdms_data)

    colors = plt.get_cmap('turbo')(np.linspace(0, 1, len(color_index)+1))

    # how many rdms to display
    n_times = len(unique_time)
    t_display_idx = (np.round(np.linspace(0, n_times-1, min(n_times, n_t_display)))).astype(int)
    t_display_idx = np.unique(t_display_idx)
    n_t_display = len(t_display_idx)

    # auto determine relative sizes of axis
    timecourse_plot_rel_height = timecourse_plot_rel_height or n_t_display // 3
    base_size = 40 / n_t_display if fig_width is None else fig_width / n_t_display

    # figure layout
    fig = plt.figure(
        constrained_layout=True,
        figsize=(base_size * n_t_display, base_size * timecourse_plot_rel_height)
    )
    gs = fig.add_gridspec(timecourse_plot_rel_height+1, n_t_display)
    tc_ax = fig.add_subplot(gs[:-1, :])
    rdm_axes = [fig.add_subplot(gs[-1, i]) for i in range(n_t_display)]

    # plot dissimilarity timecourses
    dissimilarities_mean = np.zeros((rdms_data.dissimilarities.shape[1], len(unique_time)))
    for i, t in enumerate(unique_time):
        dissimilarities_mean[:, i] = np.mean(rdms_data.dissimilarities[t == time, :], axis=0)

    def _plot_mean_dissimilarities(labels=False):
        for i, (pairwise_name, idx) in enumerate(color_index.items()):
            mn = np.mean(dissimilarities_mean[idx, :], axis=0)
            n = np.sqrt(dissimilarities_mean.shape[0])
            # se is over dissimilarities, not over subjects
            se = np.std(dissimilarities_mean[idx, :], axis=0)/n
            tc_ax.fill_between(unique_time, mn-se, mn+se, color=colors[i], alpha=.3)
            label = pairwise_name if labels else None
            tc_ax.plot(unique_time, mn, color=colors[i], linewidth=2, label=label)

    def _plot_individual_dissimilarities():
        for i, (_, idx) in enumerate(color_index.items()):
            a = max(1/255., 1/n_dissimilarity_elements)
            tc_ax.plot(unique_time, dissimilarities_mean[idx, :].T, color=colors[i], alpha=a)

    if plot_individual_dissimilarities:
        if colored_conditions is not None:
            _plot_mean_dissimilarities()
            yl = tc_ax.get_ylim()
            _plot_individual_dissimilarities()
            tc_ax.set_ylim(yl)
        else:
            _plot_individual_dissimilarities()

    if colored_conditions is not None:
        _plot_mean_dissimilarities(True)

    yl = tc_ax.get_ylim()
    for t in unique_time[t_display_idx]:
        tc_ax.plot([t, t], yl, linestyle=':', color='b', alpha=0.3)
    tc_ax.set_ylabel(f'Dissimilarity\n({rdms_data.dissimilarity_measure})')
    tc_ax.set_xticks(unique_time)
    tc_ax.set_xticklabels([
        time_formatted[idx] if idx in t_display_idx else '' for idx in range(len(unique_time))
    ])
    dt = np.diff(unique_time[t_display_idx])[0]
    tc_ax.set_xlim(unique_time[t_display_idx[0]] - dt / 2, unique_time[t_display_idx[-1]]+dt/2)

    tc_ax.legend()

    # display (selected) rdms
    vmax = np.std(rdms_data.dissimilarities) * 2
    for i, (tidx, a) in enumerate(zip(t_display_idx, rdm_axes)):
        mean_dissim = np.mean(rdms_data.subset('time', unique_time[tidx]).get_matrices(), axis=0)
        a.imshow(mean_dissim, vmin=0, vmax=vmax)  # pyright: ignore reportArgumentType
        a.set_title(f'{np.round(unique_time[tidx]*1000,2):0.0f} ms')
        a.set_yticklabels([])
        a.set_yticks([])
        a.set_xticklabels([])
        a.set_xticks([])

    return fig, [tc_ax] + rdm_axes


def unsquareform(a: NDArray) -> NDArray:
    """Helper function; convert squareform to vector
    """
    return a[np.nonzero(np.triu(a, k=1))]


def _map_colors(
        colored_conditions: Optional[list],
        plot_individual_dissimilarities: Optional[bool],
        rdms: RDMs
) -> Tuple[bool, Dict[str, NDArray]]:
    n_dissimilarity_elements = rdms.dissimilarities.shape[1]
    # color mapping from colored conditions
    if colored_conditions is not None:
        if plot_individual_dissimilarities is None:
            plot_individual_dissimilarities = False
        sf_conds = [[{c1, c2} for c1 in colored_conditions] for c2 in colored_conditions]
        pairwise_conds = unsquareform(np.array(sf_conds))
        pairwise_conds_unique = np.unique(pairwise_conds)
        color_index = {}
        for x in pairwise_conds_unique:
            if len(list(x)) == 2:
                key = f'{list(x)[0]} vs {list(x)[1]}'
            else:
                key = f'{list(x)[0]} vs {list(x)[0]}'
            color_index[key] = pairwise_conds == x
    else:
        color_index = {'': np.array([True]*n_dissimilarity_elements)}
        plot_individual_dissimilarities = True
    return plot_individual_dissimilarities, color_index
