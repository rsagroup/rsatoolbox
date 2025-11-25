#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Barplot for model comparison based on a results file
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.path import Path
from matplotlib import transforms
from matplotlib import cm
import networkx as nx
from networkx.algorithms.clique import find_cliques as maximal_cliques
from scipy.spatial.distance import squareform
from rsatoolbox.util.inference_util import all_tests, get_errorbars
from rsatoolbox.util.rdm_utils import batch_to_vectors


def plot_model_comparison(result, sort=False, colors=None,
                          alpha=0.01, test_pair_comparisons=True,
                          multiple_pair_testing='fdr',
                          test_above_0=True,
                          test_below_noise_ceil=True,
                          error_bars='sem',
                          test_type='t-test'):
    """ Plots the results of RSA inference on a set of models as a bar graph
    with one bar for each model indicating its predictive performance. The
    function also shows the noise ceiling whose upper edge is an upper bound
    on the performance the true model could achieve (given noise and inter-
    subject variability) and whose lower edge is an estimate of a lower bound
    on the performance of the true model. In addition, all pairwise inferential
    model comparisons are shown in the upper part of the figure.
    The only mandatory input is a "result" object containing  model evaluations
    for bootstrap samples and crossvalidation folds. These are used here to
    construct confidence intervals and perform the significance tests.

    All string inputs are case insensitive.

    Args:
        result (rsatoolbox.inference.result.Result):
            model evaluation result

        sort (Boolean or string):

            False (default): plot bars in the order passed

            'descend[ing]': plot bars in descending order of model performance

            'ascend[ing]': plot bars in ascending order of model performance

        colors (list of lists, numpy array, matplotlib colormap):

            None (default):
                default blue for all bars

            single color:
                list or numpy array of 3 or 4 values (RGB, RGBA)
                specifying the color for all bars

            multiple colors:
                list of lists or numpy array (number of colors by
                3 or 4 channels -- RGB, RGBA). If the number of colors
                matches the number of models, each color is used for the
                bar corresponding to one model (in the order of the models
                as passed). If the number of colors does not match the
                number of models, the list is linearly interpolated to
                assign a color to each model (in the order of the models as
                passed). For example, two colors will become a gradation,
                unless there are exactly two model. Instead of a list of
                lists or numpy array, a matplotlib colormap object may also
                be passed (e.g. colors = cm.coolwarm).

        alpha (float):
            significance threshold (p threshold or FDR q threshold)

        test_pair_comparisons (Boolean or string):

            False or None:
                do not plot pairwise model comparison results

            True (default):
                plot pairwise model comparison results using
                default settings

            'arrows':
                plot results in arrows style, indicating pairs of sets
                between which all differences are significant

            'nili':
                plot results as Nili bars (Nili et al. 2014), indicating
                each significant difference by a horizontal line (or each
                nonsignificant difference if the string contains a '2', e.g.
                'nili2')

            'golan':
                plot results as Golan wings (Golan et al. 2020), with one
                wing (graphical element) indicating all dominance relationships
                for one model.

            'cliques': plot results as cliques of insignificant differences

        multiple_pair_testing (Boolean or string):

            False or 'none':
                do not adjust for multiple testing for the
                pairwise model comparisons

            'FDR' or 'fdr' (default):
                control the false-discorvery rate at
                q = alpha

            'FWER',' fwer', or 'Bonferroni':
                control the familywise error rate
                using the Bonferroni method

        test_above_0 (Boolean or string):

            False or None:
                do not plot results of statistical comparison of
                each model performance against 0

            True (default):
                plot results of statistical comparison of each
                model performance against 0 using default settings ('dewdrops')

            'dewdrops':
                place circular "dewdrops" at the baseline to indicate
                models whose performance is significantly greater than 0

            'icicles':
                place triangular "icicles" at the baseline to indicate
                models whose performance is significantly greater than 0

            Tests are one-sided, use the global alpha threshold and are
            automatically Bonferroni-corrected for the number of models tested.

        test_below_noise_ceil (Boolean or string):

            False or None:
                do not plot results of statistical comparison of
                each model performance against the lower-bound estimate of the
                noise ceiling

            True (default):
                plot results of statistical comparison of each
                model performance against the lower-bound estimate of the noise
                ceiling using default settings ('dewdrops')

            'dewdrops':
                use circular "dewdrops" at the lower bound of the
                noise ceiling to indicate models whose performance is
                significantly below the lower-bound estimate of the noise
                ceiling

            'icicles':
                use triangular "icicles" at the lower bound of the noise
                ceiling to indicate models whose performance is significantly
                below the lower-bound estimate of the noise ceiling

            Tests are one-sided, use the global alpha threshold and are
            automatically Bonferroni-corrected for the number of models tested.

        error_bars (Boolean or string):

            False or None:
                do not plot error bars

            True (default) or 'SEM':
                plot the standard error of the mean

            'CI':
                plot 95%-confidence intervals (exluding 2.5% on each side)

            'CI[x]':
                plot x%-confidence intervals
                (exluding (100-x)/2% on each side)
                i.e. 'CI' has the same effect as 'CI95'

            Confidence intervals are based on the bootstrap procedure,
            reflecting variability of the estimate across subjects and/or
            experimental conditions.

            'dots':
                Draws dots for each data-point, i.e. first dimension of
                the evaluation tensor. This is primarily sensible for
                fixed evaluation where this dimension
                corresponds to the subjects in the experiment.

        test_type (string):
            which tests to perform:

            't-test':
                performs a t-test based on the variance estimates
                in the result structs

            'bootstrap':
                performs a bootstrap test, i.e. checks based
                on the number of samples defying H0

            'ranksum':
                performs wilcoxon signed rank sum tests

    Returns:
        (matplotlib.pyplot.Figure, matplotlib.pyplot.Axis,
            matplotlib.pyplot.Axis):
            the figure and axes the plots were made into.
            This allows further modification, saving and printing.

    """

    # Prepare and sort data
    evaluations = result.evaluations
    models = result.models
    noise_ceiling = result.noise_ceiling
    method = result.method
    model_var = result.model_var
    diff_var = result.diff_var
    noise_ceil_var = result.noise_ceil_var
    dof = result.dof
    if result.cv_method == 'fixed':
        n_bootstraps, n_models, _ = evaluations.shape
        perf = np.mean(evaluations, axis=0)
        perf = np.nanmean(perf, axis=-1)
    elif result.cv_method == 'crossvalidation':
        n_bootstraps, n_models, _ = evaluations.shape
        perf = np.mean(evaluations, axis=0)
        perf = np.nanmean(perf, axis=-1)
        if any([test_pair_comparisons,
                test_above_0, test_below_noise_ceil]):
            warnings.warn('tests deactivated as crossvalidation does not'
                          + 'provide uncertainty estimate')
            test_pair_comparisons = False
            test_above_0 = False
            test_below_noise_ceil = False
        if error_bars and error_bars.lower() != 'dots':
            warnings.warn('errorbars deactivated as crossvalidation does not'
                          + 'provide uncertainty estimate')
            error_bars = False
    else:
        while len(evaluations.shape) > 2:
            evaluations = np.nanmean(evaluations, axis=-1)
        evaluations = evaluations[~np.isnan(evaluations[:, 0])]
        n_bootstraps, n_models = evaluations.shape
        perf = np.mean(evaluations, axis=0)
    noise_ceiling = np.array(noise_ceiling)
    if sort is True:
        sort = 'descending'  # descending by default if sort is True
    elif sort is False:
        sort = 'unsorted'
    if sort != 'unsorted':  # 'descending' or 'ascending'
        idx = np.argsort(perf)
        if 'descend' in sort.lower():
            idx = np.flip(idx)
        perf = perf[idx]
        evaluations = evaluations[:, idx]
        if model_var is not None:
            model_var = model_var[idx]
        if noise_ceil_var is not None:
            noise_ceil_var = noise_ceil_var[idx]
        if diff_var is not None:
            diff_var = squareform(squareform(diff_var)[idx][:, idx])
        models = [models[i] for i in idx]
        if not ('descend' in sort.lower() or
                'ascend' in sort.lower()):
            raise ValueError(
                'plot_model_comparison: Argument ' +
                'sort is incorrectly defined as ' +
                sort + '.')

    # run tests
    if any([test_pair_comparisons,
            test_above_0, test_below_noise_ceil]):
        p_pairwise, p_zero, p_noise = all_tests(
            evaluations, noise_ceiling, test_type,
            model_var=model_var, diff_var=diff_var,
            noise_ceil_var=noise_ceil_var, dof=dof)

    # Prepare axes for bars and pairwise comparisons
    fs, fs2 = 18, 14  # axis label font sizes
    l, b, w, h = 0.15, 0.15, 0.8, 0.8  # noqa: E741
    fig = plt.figure(figsize=(12.5, 10))
    if test_pair_comparisons is True:
        test_pair_comparisons = 'arrows'
    if test_pair_comparisons:
        if test_pair_comparisons.lower() in ['arrows', 'cliques']:
            h_pair_tests = 0.25
        elif 'golan' in test_pair_comparisons.lower():
            h_pair_tests = 0.3
        elif 'nili' in test_pair_comparisons.lower():
            h_pair_tests = 0.4
        else:
            raise ValueError(
                'plot_model_comparison: Argument ' +
                'test_pair_comparisons is incorrectly defined as ' +
                test_pair_comparisons + '.')
        ax = plt.axes((l, b, w, h*(1-h_pair_tests)))
        axbar = plt.axes((l, b + h * (1 - h_pair_tests), w,
                          h * h_pair_tests * 0.7))
    else:
        ax = plt.axes((l, b, w, h))
        axbar = None

    # Define the model colors
    if colors is None:  # no color passed...
        colors = [0, 0.4, 0.9, 1]  # use default blue
    elif isinstance(colors, cm.colors.LinearSegmentedColormap):
        cmap = cm.get_cmap(colors)
        colors = cmap(np.linspace(0, 1, 100))[np.newaxis, :, :3].squeeze()
    colors = np.array([np.array(col) for col in colors])
    if len(colors.shape) == 1:  # one color passed...
        n_col, n_chan = 1, colors.shape[0]
        colors.shape = (n_col, n_chan)
    else:  # multiple colors passed...
        n_col, n_chan = colors.shape
        if n_col == n_models:  # one color passed for each model...
            cols2 = colors
        else:  # number of colors passed does not match number of models...
            # interpolate colors to define a color for each model
            cols2 = np.empty((n_models, n_chan))
            for c in range(n_chan):
                cols2[:, c] = np.interp(np.array(range(n_models)),
                                        np.array(range(n_col))/n_col*n_models,
                                        colors[:, c])
        if sort != 'unsorted':
            colors = cols2[idx, :]
        else:
            colors = cols2
    if colors.shape[1] == 3:
        colors = np.concatenate((colors, np.ones((colors.shape[0], 1))),
                                axis=1)

    # Plot bars and error bars
    if method == 'neg_riem_dist':
        ax.bar(np.arange(evaluations.shape[1]), perf-np.min(perf),
               color=colors, bottom=np.min(perf))
    else:
        ax.bar(np.arange(evaluations.shape[1]), perf, color=colors)
    if error_bars:
        limits = get_errorbars(model_var, evaluations, dof, error_bars,
                               test_type)
        ax.errorbar(np.arange(evaluations.shape[1]), perf,
                    yerr=limits, fmt='none', ecolor='k',
                    capsize=0, linewidth=3)

    # Test whether model performance exceeds 0 (one sided)
    if test_above_0 is True:
        test_above_0 = 'dewdrops'
    if test_above_0:
        model_significant = p_zero < alpha / n_models
        half_sym_size = 9
        if test_above_0.lower() == 'dewdrops':
            halfmoonup = Path.wedge(0, 180)
            ax.plot(model_significant.nonzero()[0],
                    np.tile(0, model_significant.sum()), 'w',
                    marker=halfmoonup, markersize=half_sym_size,
                    linewidth=0)
        elif test_above_0.lower() == 'icicles':
            ax.plot(model_significant.nonzero()[0],
                    np.tile(0, model_significant.sum()), 'w',
                    marker=10, markersize=half_sym_size,
                    linewidth=0)
        else:
            raise ValueError(
                'plot_model_comparison: Argument test_above_0' +
                ' is incorrectly defined as ' + test_above_0 + '.')

    # Plot noise ceiling
    noise_ceil_col = [0.5, 0.5, 0.5, 0.2]
    if noise_ceiling is not None:
        noise_lower = np.nanmean(noise_ceiling[0])
        noise_upper = np.nanmean(noise_ceiling[1])
        noiserect = patches.Rectangle((-0.5, noise_lower), len(perf),
                                      noise_upper-noise_lower, linewidth=0,
                                      facecolor=noise_ceil_col, zorder=1e6)
        ax.add_patch(noiserect)

    # Test whether model performance is below the noise ceiling's lower bound
    # (one sided)
    if test_below_noise_ceil is True:
        test_below_noise_ceil = 'dewdrops'
    if test_below_noise_ceil:
        model_below_lower_bound = p_noise < alpha / n_models

        if test_below_noise_ceil.lower() == 'dewdrops':
            halfmoondown = Path.wedge(180, 360)
            ax.plot(model_below_lower_bound.nonzero()[0],
                    np.tile(noise_lower+0.0000, model_below_lower_bound.sum()),
                    color='none',
                    marker=halfmoondown, markersize=half_sym_size,
                    markerfacecolor=noise_ceil_col,
                    markeredgecolor='none', linewidth=0)
        elif test_below_noise_ceil.lower() == 'icicles':
            ax.plot(model_below_lower_bound.nonzero()[0],
                    np.tile(noise_lower+0.0007, model_below_lower_bound.sum()),
                    color='none',
                    marker=11, markersize=half_sym_size,
                    markerfacecolor=noise_ceil_col,
                    markeredgecolor='none', linewidth=0)
        else:
            raise ValueError(
                'plot_model_comparison: Argument ' +
                'test_below_noise_ceil is incorrectly defined as ' +
                test_below_noise_ceil + '.')

    # Pairwise model comparisons
    if test_pair_comparisons:
        if test_type == 'bootstrap':
            model_comp_descr = 'Model comparisons: two-tailed bootstrap, '
        elif test_type == 't-test':
            model_comp_descr = 'Model comparisons: two-tailed t-test, '
        elif test_type == 'ranksum':
            model_comp_descr = 'Model comparisons: two-tailed Wilcoxon-test, '
        n_tests = int((n_models ** 2 - n_models) / 2)
        if multiple_pair_testing is None:
            multiple_pair_testing = 'uncorrected'
        if multiple_pair_testing.lower() == 'bonferroni' or \
           multiple_pair_testing.lower() == 'fwer':
            significant = p_pairwise < (alpha / n_tests)
        elif multiple_pair_testing.lower() == 'fdr':
            ps = batch_to_vectors(np.array([p_pairwise]))[0][0]
            ps = np.sort(ps)
            criterion = alpha * (np.arange(ps.shape[0]) + 1) / ps.shape[0]
            k_ok = ps < criterion
            if np.any(k_ok):
                k_max = np.max(np.where(ps < criterion)[0])
                crit = criterion[k_max]
            else:
                crit = 0
            significant = p_pairwise < crit
        else:
            if 'uncorrected' not in multiple_pair_testing.lower():
                raise ValueError(
                    'plot_model_comparison: Argument ' +
                    'multiple_pair_testing is incorrectly defined as ' +
                    multiple_pair_testing + '.')
            significant = p_pairwise < alpha
        model_comp_descr = _get_model_comp_descr(
            test_type, n_models, multiple_pair_testing, alpha,
            n_bootstraps, result.cv_method, error_bars,
            test_above_0, test_below_noise_ceil)
        fig.suptitle(model_comp_descr, fontsize=fs2/2)
        axbar.set_xlim(ax.get_xlim())
        digits = [d for d in list(test_pair_comparisons) if d.isdigit()]
        if len(digits) > 0:
            v = int(digits[0])
        else:
            v = None
        if 'nili' in test_pair_comparisons.lower():
            if v:
                plot_nili_bars(axbar, significant, version=v)
            else:
                plot_nili_bars(axbar, significant)
        elif 'golan' in test_pair_comparisons.lower():
            if v:
                plot_golan_wings(axbar, significant, perf, sort, colors,
                                 version=v)
            else:
                plot_golan_wings(axbar, significant, perf, sort, colors)
        elif 'arrows' in test_pair_comparisons.lower():
            plot_arrows(axbar, significant)
        elif 'cliques' in test_pair_comparisons.lower():
            plot_cliques(axbar, significant)

    # Floating axes
    if method == 'neg_riem_dist':
        ytoptick = noise_upper + 0.1
        ymin = np.min(perf)
    else:
        ytoptick = np.floor(min(1, noise_upper) * 10) / 10
        ymin = 0
    ax.set_yticks(np.arange(ymin, ytoptick + 1e-6, step=0.1))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks(np.arange(n_models))
    ax.spines['left'].set_bounds(ymin, ytoptick)
    ax.spines['bottom'].set_bounds(0, n_models - 1)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.rc('ytick', labelsize=fs2)

    # Axis labels
    y_label_string = _get_y_label(method)
    ylabel_fig_x, ysublabel_fig_x = 0.07, 0.095
    trans = transforms.blended_transform_factory(fig.transFigure,
                                                 ax.get_yaxis_transform())
    ax.text(ylabel_fig_x, (ymin + ytoptick) / 2, 'RDM prediction accuracy',
            horizontalalignment='center', verticalalignment='center',
            rotation='vertical', fontsize=fs, fontweight='bold',
            transform=trans)
    ax.text(ysublabel_fig_x, (ymin+ytoptick)/2,
            y_label_string,
            horizontalalignment='center', verticalalignment='center',
            rotation='vertical', fontsize=fs2, fontweight='normal',
            transform=trans)

    if models is not None:
        ax.set_xticklabels([m.name for m in models], fontsize=fs2,
                           rotation=45)
    return fig, ax, axbar


def plot_nili_bars(axbar, significant, version=1):
    """ plots the results of the pairwise inferential model comparisons in the
    form of a set of black horizontal bars connecting significantly different
    models as in the 2014 RSA Toolbox (Nili et al. 2014).

    Args:
        axbar: Matplotlib axes handle to plot in

        significant: Boolean matrix of model comparisons

        version:

            - 1 (Normal Nili bars, indicating significant differences)
            - 2 (Negative Nili bars in gray, indicating nonsignificant
              comparison results)

    Returns:
        ---

    """

    k = 1
    ns_col = [0.5, 0.5, 0.5]
    w = 0.2
    for i in range(significant.shape[0]):
        drawn1 = False
        for j in range(i + 1, significant.shape[0]):
            if version == 1:
                if significant[i, j]:
                    axbar.plot((i, j), (k, k), 'k-', linewidth=2)
                    k += 1
                    drawn1 = True
            elif version == 2:
                if not significant[i, j]:
                    axbar.plot((i, j), (k, k), '-', linewidth=2,
                               color=ns_col)
                    axbar.plot(((i+j)/2-w/2, (i+j)/2+w/2), (k, k), '-',
                               linewidth=3, color='w')
                    axbar.text((i+j)/2, k, 'n.s.',
                               horizontalalignment='center',
                               verticalalignment='center',
                               fontsize=8, fontweight='normal', color=ns_col)
                    k += 1
                    drawn1 = True
        if drawn1:
            k += 1
    axbar.set_axis_off()
    axbar.set_ylim((0, k))


def plot_golan_wings(axbar, significant, perf, sort, colors=None,
                     always_black=False, version=3):
    """ Plots the results of the pairwise inferential model comparisons in the
    form of black horizontal bars with a tick mark at the reference model and
    a circular bulge at each significantly different model similar to the
    visualization in Golan, Raju, Kriegeskorte (2020).

    Args:
        axbar: Matplotlib axes handle to plot in
        significant: Boolean matrix of model comparisons
        version:

            - 0 (single wing: solid circle anchor and open circles),
            - 1 (single wing: tick anchor and circles),
            - 2 (single wing: circle anchor and up and down feathers)
            - 3 (double wings: circle anchor,
              downward dominance-indicating feathers,
              from bottom to top in model order)
            - 4 (double wings: circle anchor,
              downward dominance-indicating feathers,
              from bottom to top in performance order)

    Returns:
        ---

    """

    # Define wing order
    n_models = significant.shape[0]
    wing_order = np.array(range(n_models))  # to the right by default
    if 'ascend' in sort.lower():
        wing_order = np.flip(wing_order)  # to the left if bars are ascending
    if version == 4:
        wing_order = np.argsort(-perf)

    # Define vertical spacing
    bbox = axbar.get_window_extent().transformed(
        plt.gcf().dpi_scale_trans.inverted())
    h_inch = bbox.height
    h = 1
    for wo_i, i in enumerate(wing_order):
        if version in [3, 4]:
            js = np.concatenate((wing_order[0:wo_i],
                                 wing_order[wo_i+1:])).astype('int')
            js = js[np.logical_and(significant[i, js], perf[i] > perf[js])]
        else:
            js = wing_order[wo_i+1:][significant[i, wing_order[wo_i+1:]]]
        js = js[significant[i, js]]
        if len(js) > 0:
            h += 1
    axbar.set_axis_off()
    axbar.set_ylim((0, h))

    # Draw the wings
    if always_black or colors is None or colors.shape[0] == 1:
        colors = np.tile([0, 0, 0, 1], (n_models, 1))
    tick_length_inch = 0.08
    k = 1
    for wo_i, i in enumerate(wing_order):
        if version in [3, 4]:
            js = np.concatenate((wing_order[0:wo_i],
                                 wing_order[wo_i+1:])).astype('int')
            js = js[np.logical_and(significant[i, js], perf[i] > perf[js])]
        else:
            js = wing_order[wo_i+1:][significant[i, wing_order[wo_i+1:]]]
        js = js[significant[i, js]]
        if len(js) > 0:
            if version != 1:
                # circle anchor
                axbar.plot(i, k, markersize=8, marker='o',
                           markeredgecolor=colors[i, :],
                           markerfacecolor=colors[i, :])
            elif version == 1:
                # tick anchor
                axbar.plot((i, i), (k - tick_length_inch/h_inch*h, k), '-',
                           linewidth=2, color=colors[i, :])  # tick
            for j in js:
                if version == 0:
                    axbar.plot(j, k, markersize=8, marker='o',
                               markeredgecolor=colors[i, :],
                               markerfacecolor='w')
                elif version == 1:
                    axbar.plot(j, k, markersize=8, marker='o',
                               markeredgecolor=colors[i, :],
                               markerfacecolor=colors[i, :])
                elif version in [2, 3, 4]:
                    if perf[i] > perf[j]:
                        tick_ver_end = k - tick_length_inch/h_inch*h
                    elif perf[i] < perf[j]:
                        tick_ver_end = k + tick_length_inch/h_inch*h
                    axbar.plot((j, j), (k, tick_ver_end), '-', linewidth=2,
                               color=colors[i, :])
            # wing line
            axbar.plot((min(i, js.min()), max(i, js.max())), (k, k), 'k-',
                       linewidth=2, color=colors[i, :], zorder=-1)
            k += 1


def plot_arrows(axbar, significant):
    """ Summarizes the significances with arrows. The argument significant is
    a binary matrix of pairwise model comparisons. A nonzero value (or True)
    indicates that the model specified by the row index beats the model
    specified by the column index. Only the lower triangular part of compMat is
    used, so the upper triangular part need not be filled in symmetrically. The
    summary will be most concise if models are ordered by performance (using
    the sort argument of model_plot.py).
    """

    # Preparations
    n = significant.shape[0]
    remaining = significant.copy()

    # make arrowheads
    verts_R = [(0, 0), (0, 1), (2, 0), (0, -1), (0, 0)]
    verts_L = [(-x, y) for (x, y) in verts_R]
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO,
             Path.CLOSEPOLY]
    ah_R = Path(verts_R, codes)
    ah_L = Path(verts_L, codes)

    # Capture as many comparisons as possible with double arrows
    double_arrows = []
    for ambiguity_span in range(0, n-1):
        # consider short double arrows first (these cover many comparisons)
        for i in range(n-1, ambiguity_span, -1):
            if significant[i:n, 0:i-ambiguity_span].all() and \
                    remaining[i:n, 0:i-ambiguity_span].any():
                # add double arrow
                double_arrows.append((i-ambiguity_span-1, i))
                remaining[i:n, 0:i-ambiguity_span] = 0

    # Capture as many of the remaining comparisons as possible with arrows
    arrows = []
    for dist2diag in range(1, n):
        for i in range(n-1, dist2diag-1, -1):
            if significant[i, 0:i-dist2diag+1].all() and \
                    remaining[i, 0:i-dist2diag+1].any():
                arrows.append((i, i-dist2diag))  # add left arrow
                remaining[i, 0:i-dist2diag+1] = 0
            if significant[i:n, i-dist2diag].all() and \
                    remaining[i:n, i-dist2diag].any():
                arrows.append((i-dist2diag, i))  # add right arrow
                remaining[i:n, i-dist2diag] = 0

    # Capture the remaining comparisons with lines
    lines = []
    for i in range(1, n):
        for j in range(0, i-1):
            if remaining[i, j]:
                lines.append((i, j))  # add line

    # Plot
    expected_n_lines = 6
    axbar.set_ylim((0, expected_n_lines))
    axbar.set_axis_off()
    n_elements = len(double_arrows)+len(arrows)+len(lines)
    if n_elements == 0:
        return
    occupied = np.zeros((n_elements, 3*n))
    for m in range(0, int(np.ceil(n/2))):
        double_arrows_left = [(i, j) for (i, j) in double_arrows if i == m]
        if len(double_arrows_left) > 0:
            i, j = double_arrows_left[0]
            double_arrows.remove((i, j))
            if j < i:
                i, j = j, i
            k = 1
            while occupied[k-1, i*3+2:j*3+1].any():
                k += 1
            if i == 0:
                draw_hor_arrow(axbar, i, j, k, '->', ah_L, ah_R)
            elif j == n-1:
                draw_hor_arrow(axbar, i, j, k, '<-', ah_L, ah_R)
            else:
                draw_hor_arrow(axbar, i, j, k, '<->', ah_L, ah_R)
            occupied[k-1, i*3+2:j*3+1] = 1

        double_arrows_right = \
            [(i, j) for (i, j) in double_arrows if j == n-1-m]
        if len(double_arrows_right) > 0:
            i, j = double_arrows_right[0]
            double_arrows.remove((i, j))
            k = 1
            while occupied[k-1, i*3+2:j*3+1].any():
                k += 1
            if i == 0:
                draw_hor_arrow(axbar, i, j, k, '->', ah_L, ah_R)
            elif j == n-1:
                draw_hor_arrow(axbar, i, j, k, '<-', ah_L, ah_R)
            else:
                draw_hor_arrow(axbar, i, j, k, '<->', ah_L, ah_R)
            occupied[k-1, i*3+2:j*3+1] = 1

    for m in range(0, int(np.ceil(n/2))):
        arrows_left = [(i, j) for (i, j) in arrows if (i < j and i == m) or
                       (j < i and j == m)]
        while len(arrows_left) > 0:
            i, j = arrows_left.pop()
            arrows.remove((i, j))
            k = 1
            while occupied[k-1, i*3+2:j*3+1].any() or \
                    occupied[k-1, j*3+2:i*3+1].any():
                k += 1
            draw_hor_arrow(axbar, i, j, k, '->', ah_L, ah_R)
            occupied[k-1, i*3+2:j*3+1] = 1
            occupied[k-1, j*3+2:i*3+1] = 1

        arrows_right = [(i, j) for (i, j) in arrows if (i < j and j == n-1-m)
                        or (j < i and i == n-1-m)]
        while len(arrows_right) > 0:
            i, j = arrows_right.pop()
            arrows.remove((i, j))
            k = 1
            while occupied[k-1, i*3+2:j*3+1].any() or \
                    occupied[k-1, j*3+2:i*3+1].any():
                k += 1
            draw_hor_arrow(axbar, i, j, k, '->', ah_L, ah_R)
            occupied[k-1, i*3+2:j*3+1] = 1
            occupied[k-1, j*3+2:i*3+1] = 1

    for m in range(0, int(np.ceil(n/2))):
        lines_left = [(i, j) for (i, j) in lines if i == m]
        while len(lines_left) > 0:
            i, j = lines_left.pop()
            lines.remove((i, j))
            if j < i:
                i, j = j, i
            k = 1
            while occupied[k-1, i*3+2:j*3+1].any():
                k += 1
            axbar.plot((i, j), (k, k), 'k-', linewidth=2)
            occupied[k-1, i*3+2:j*3+1] = 1

        lines_right = [(i, j) for (i, j) in lines if j == n-1-m]
        while len(lines_right) > 0:
            i, j = lines_right.pop()
            lines.remove((i, j))
            if j < i:
                i, j = j, i
            k = 1
            while occupied[k-1, i*3+2:j*3+1].any():
                k += 1
            axbar.plot((i, j), (k, k), 'k-', linewidth=2)
            occupied[k-1, i*3+2:j*3+1] = 1
    h = occupied.sum(axis=1)
    if np.any(h > 0):
        h = h.nonzero()[0].max()+1
    else:
        h = 1
    axbar.set_ylim((0, max(expected_n_lines, h)))


def draw_hor_arrow(ax, x1, x2, y, style, ah_L, ah_R):
    """
    Draws a horizontal arrow from (x1, y) to (x2, y) if style is '->' and
    in the reverse direction if style is '<-'. If style is '<->', this
    function draws a double arrow.
    """
    lw, s = 1.6, 0.45
    ms, ms_a = 8, 18
    if (x1 < x2 and style == '->') or (x2 < x1 and style == '<-'):
        mr = ah_R  # arrow points right
    else:
        mr = ah_L  # arrow points left
    if style == '<-':
        x1, x2 = x2, x1  # arrow from x1 to x2 now
    d = (x2-x1)/abs(x2-x1)
    if style == '<->':
        ax.plot(x1+d*s, y, 'k', markersize=ms_a, marker=ah_L)
        ax.plot((x1+d*s, x2-d*s), (y, y), 'k-', linewidth=lw)
        ax.plot(x2-d*s, y, 'k', markersize=ms_a, marker=ah_R)
    else:
        ax.plot(x1, y, 'k', markersize=ms, marker='o')
        ax.plot((x1, x2-d*s), (y, y), 'k-', linewidth=lw)
        ax.plot(x2-d*s, y, 'k', markersize=ms_a, marker=mr)


def plot_cliques(axbar, significant):
    """ plots the results of the pairwise inferential model comparisons in the
    form of a set of maximal cliques of models that are not significantly
    different in performance. One bar is drawn for each clique with open
    circles indicating the clique members. Within a clique of models, no
    pair comparison is significant. All pair comparisons not indicated as
    insignificant are significant.

    Args:
        axbar: Matplotlib axes handle to plot in
        significant: Boolean matrix of model comparisons

    Returns:
        ---

    """

    G = nx.Graph(np.logical_not(significant))
    cliques = list(maximal_cliques(G))
    n = significant.shape[0]
    ns_col = [0.6, 0.6, 0.6]
    expected_n_lines = 6
    axbar.set_ylim((0, expected_n_lines))
    axbar.set_axis_off()
    occupied = np.zeros((len(cliques), 3*n))
    for c in cliques:
        if len(c) > 1:
            i, j = min(c), max(c)
            k = 1
            while occupied[k-1, i*3+1:j*3+2].any():
                k += 1
            occupied[k-1, i*3+1:j*3+2] = 1
            axbar.plot((i, j), (k, k), '-', linewidth=2, color=ns_col)
            for i in c:
                axbar.plot(i, k, markersize=8, marker='o',
                           markeredgecolor=ns_col, markerfacecolor='w')
    h = occupied.sum(axis=1).nonzero()[0].max()+1
    axbar.set_ylim((0, max(expected_n_lines, h)))


def _get_model_comp_descr(test_type, n_models, multiple_pair_testing, alpha,
                          n_bootstraps, cv_method, error_bars,
                          test_above_0, test_below_noise_ceil):
    """constructs the statistics description from the parts

    Args:
        test_type : String
        n_models : integer
        multiple_pair_testing : String
        alpha : float
        n_bootstraps : integer
        cv_method : String
        error_bars : String
        test_above_0 : Bool
        test_below_noise_ceil : Bool

    Returns:
        model

    """
    if test_type == 'bootstrap':
        model_comp_descr = 'Model comparisons: two-tailed bootstrap, '
    elif test_type == 't-test':
        model_comp_descr = 'Model comparisons: two-tailed t-test, '
    elif test_type == 'ranksum':
        model_comp_descr = 'Model comparisons: two-tailed Wilcoxon-test, '
    n_tests = int((n_models ** 2 - n_models) / 2)
    if multiple_pair_testing is None:
        multiple_pair_testing = 'uncorrected'
    if multiple_pair_testing.lower() == 'bonferroni' or \
       multiple_pair_testing.lower() == 'fwer':
        model_comp_descr = (model_comp_descr
                            + 'p < {:<.5g}'.format(alpha)
                            + ', Bonferroni-corrected for '
                            + str(n_tests)
                            + ' model-pair comparisons')
    elif multiple_pair_testing.lower() == 'fdr':
        model_comp_descr = (model_comp_descr +
                            'FDR q < {:<.5g}'.format(alpha) +
                            ' (' + str(n_tests) +
                            ' model-pair comparisons)')
    else:
        if 'uncorrected' not in multiple_pair_testing.lower():
            raise ValueError(
                'plot_model_comparison: Argument ' +
                'multiple_pair_testing is incorrectly defined as ' +
                multiple_pair_testing + '.')
        model_comp_descr = (model_comp_descr +
                            'p < {:<.5g}'.format(alpha) +
                            ', uncorrected (' + str(n_tests) +
                            ' model-pair comparisons)')
    if cv_method in ['bootstrap_rdm', 'bootstrap_pattern',
                     'bootstrap_crossval']:
        model_comp_descr = model_comp_descr + \
            '\nInference by bootstrap resampling ' + \
            '({:<,.0f}'.format(n_bootstraps) + ' bootstrap samples) of '
    if cv_method == 'bootstrap_rdm':
        model_comp_descr = model_comp_descr + 'subjects. '
    elif cv_method == 'bootstrap_pattern':
        model_comp_descr = model_comp_descr + 'experimental conditions. '
    elif cv_method in ['bootstrap', 'bootstrap_crossval']:
        model_comp_descr = model_comp_descr + \
            'subjects and experimental conditions. '
    if error_bars[0:2].lower() == 'ci':
        model_comp_descr = model_comp_descr + 'Error bars indicate the'
        if len(error_bars) == 2:
            CI_percent = 95.0
        else:
            CI_percent = float(error_bars[2:])
        model_comp_descr = (model_comp_descr + ' ' +
                            str(CI_percent) + '% confidence interval.')
    elif error_bars.lower() == 'sem':
        model_comp_descr = (
            model_comp_descr +
            'Error bars indicate the standard error of the mean.')
    elif error_bars.lower() == 'sem':
        model_comp_descr = (model_comp_descr +
                            'Dots represent the individual model evaluations.')
    if test_above_0 or test_below_noise_ceil:
        model_comp_descr = (
            model_comp_descr +
            '\nOne-sided comparisons of each model performance ')
    if test_above_0:
        model_comp_descr = model_comp_descr + 'against 0 '
    if test_above_0 and test_below_noise_ceil:
        model_comp_descr = model_comp_descr + 'and '
    if test_below_noise_ceil:
        model_comp_descr = (
            model_comp_descr +
            'against the lower-bound estimate of the noise ceiling ')
    if test_above_0 or test_below_noise_ceil:
        model_comp_descr = (model_comp_descr +
                            'are Bonferroni-corrected for ' +
                            str(n_models) + ' models.')
    return model_comp_descr


def _get_y_label(method) -> str:
    """ generates y-label string

    Args:
        method : String
            Method for model evaluation used

    Returns:
        y_label : String

    """
    if method.lower() == 'cosine':
        y_label = '[across-subject mean of cosine similarity]'
    elif method.lower() in ['cosine_cov', 'whitened cosine']:
        y_label = '[across-subject mean of whitened-RDM cosine]'
    elif method.lower() == 'spearman':
        y_label = '[across-subject mean of Spearman r rank correlation]'
    elif method.lower() in ['corr', 'pearson']:
        y_label = '[across-subject mean of Pearson r correlation]'
    elif method.lower() in ['whitened pearson', 'corr_cov']:
        y_label = '[across-subject mean of whitened-RDM Pearson r correlation]'
    elif method.lower() in ['kendall', 'tau-b']:
        y_label = '[across-subject mean of Kendall tau-b rank correlation]'
    elif method.lower() == 'tau-a':
        y_label = '[across-subject mean of ' \
            + 'Kendall tau-a rank correlation]'
    elif method.lower() == 'neg_riem_dist':
        y_label = '[across-subject mean of ' \
            + 'negative riemannian distance]'
    elif method.lower() == 'rho-a':
        y_label = '[across-subject mean of ' \
            + 'Spearman r rank correlation with random tie-breaking]'
    else:
        raise ValueError(f'Unsupported method: {method}')
    return y_label
