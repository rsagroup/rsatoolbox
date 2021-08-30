# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 17:09:00 2020
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.spatial.distance as ssd
import scipy.stats as sst
from tqdm import tqdm
import sys

import pyrsa
from pyrsa.util.inference_util import pair_tests
from pyrsa.util.rdm_utils import batch_to_vectors

fs_small, fs, fs_large = 12, 18, 22
fig_width, dpi = 10, 300

# from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc('font',**{'family':'serif','serif':['Times']})
# rc('text', usetex=True)


def map_model_comparison(result, rdms_data=None, RDM_dist_measure='corr',
                         colors=None, alpha=0.01, test_pair_comparisons=True,
                         multiple_pair_testing='fdr', test_above_0=True,
                         test_below_noise_ceil=True, error_bars='ci99',
                         label_orientation='tangential', fliplr=False):
    """ Maps the model RDMs around the data RDM to reveal the pairwise
    similarities among the model-predicted RDMs along with each
    model-predicted RDM's similarity to the data RDM. The model map
    additionally contains inferential information, indicating significant
    differences among models in terms of their ability to explain the data and
    significant deviations of the models from the data (noise floor).

    Args (All strings case insensitive):
        result (pyrsa.inference.result.Result):
            model evaluation result
            result.models (list of pyrsa.model objects): the models in the
                order in which they were passed to the inference function to
                produce the result argument. These need to be fixed RDM models.
                (Flexible models will be handled in the near future.)
        rdms_data (pyrsa.rdm.rdms.RDMs): single-subject data RDMs
        RDM_dist_measure (string): measure used to express the 2nd-order
            distances among RDMs
            'cosine': cosine distance
            'Euclidean': Euclidean distance between normalized RDMs
                (proportional to the square root of the cosine distance)
        colors (list of lists, numpy array, matplotlib colormap):
            None (default): default blue for all bars
            single color: list or numpy array of 3 or 4 values (RGB, RGBA)
                specifying the color for all bars
            multiple colors: list of lists or numpy array (number of colors by
                3 or 4 channels -- RGB, RGBA). If the number of colors matches
                the number of models, each color is used for the bar
                corresponding to one model (in the order of the models as
                passed). If the number of colors does not match the number of
                models, the list is linearly interpolated to assign a color to
                each model (in the order of the models as passed). For example,
                two colors will become a gradation, unless there are exactly
                two model. Instead of a list of lists or numpy array, a
                matplotlib colormap object may also be passed (e.g. colors =
                cm.coolwarm).
        alpha (float):
            significance threshold (p threshold or FDR q threshold)
        test_pair_comparisons (Boolean or string):
            False or None: do not plot pairwise model comparison results
            True (default): plot pairwise model comparison results using
                default settings
        multiple_pair_testing (Boolean or string):
            False or 'none': do not adjust for multiple testing for the
                pairwise model comparisons
            'FDR' or 'fdr' (default): control the false-discorvery rate at
                q = alpha
            'FWER',' fwer', or 'Bonferroni': control the familywise error rate
            using the Bonferroni method
        test_above_0 (Boolean or string):
            False or None: do not plot results of statistical comparison of
                each model performance against 0
            True (default): plot results of statistical comparison of each
                model performance against 0 using default settings ('dewdrops')
            Tests are one-sided, use the global alpha threshold and are
            automatically Bonferroni-corrected for the number of models tested.
        test_below_noise_ceil (Boolean or string):
            False or None: do not plot results of statistical comparison of
                each model performance against the lower-bound estimate of the
                noise ceiling
            True (default): plot results of statistical comparison of each
                model performance against the lower-bound estimate of the noise
                ceiling using default settings
            Tests are one-sided, use the global alpha threshold and are
            automatically Bonferroni-corrected for the number of models tested.
        error_bars (Boolean or string):
            False or None: do not plot error bars
            True (default) or 'SEM': plot the standard error of the mean
            'CI': plot 95%-confidence intervals (exluding 2.5% on each side)
            'CI[x]': plot x%-confidence intervals (exluding 2.5% on each side)
            Confidence intervals are based on the bootstrap procedure,
            reflecting variability of the estimate across subjects and/or
            experimental conditions.
        label_orientation (string): 'tangential' (default) for tangential model
            labels, 'radial' for radial model labels.
        fliplr (Boolean): If this is False (default) the arrangement will map
            the first model to the left of the second model (in reading
            direction). If this argument is True, the map will left-right
            flipped about the vertical axis.

    Returns:
        ---

    """

    # %% Prepare and sort data
    evaluations = result.evaluations
    models = result.models
    noise_ceiling = result.noise_ceiling
    inference_descr = ''

    # average the bootstrap evaluations
    while len(evaluations.shape) > 2:
        # average across trailing dimensions
        evaluations = np.nanmean(evaluations, axis=-1)
    evaluations = evaluations[~np.isnan(evaluations[:, 0])]
    noise_ceiling = np.array(noise_ceiling)
    perf = np.mean(evaluations, axis=0)  # average across bootstrap samples
    n_bootstraps, n_models = evaluations.shape

    # %% Test each model RDM for relatedness to and distinctness from the data RDM
    # test if model RDMs are significantly related to data RDM
    # (above 0)
    if test_above_0:  # one-sided test
        p = ((evaluations < 0).sum(axis=0) + 1) / n_bootstraps
        model_significant = p < alpha / n_models  # Bonferroni-corrected

    # test if model RDMs are significantly distinct from the data RDM
    # (below the noise ceiling's lower bound)
    if test_below_noise_ceil:  # one-sided test
        if len(noise_ceiling.shape) > 1:
            noise_lower_bs = noise_ceiling[0]
            noise_lower_bs = noise_lower_bs.reshape(noise_lower_bs.shape[0], 1)
        else:
            noise_lower_bs = noise_ceiling[0].reshape(1, 1)
        diffs = noise_lower_bs - evaluations  # positive if below lower bound
        p = ((diffs < 0).sum(axis=0) + 1) / n_bootstraps
        model_below_lower_bound = p < alpha / n_models  # Bonferroni-corrected

    # %% Perform pairwise model comparisons
    if test_pair_comparisons:
        inference_descr += 'Model comparisons: two-tailed, '
        p_values = pair_tests(evaluations)
        n_tests = int((n_models**2 - n_models) / 2)
        if multiple_pair_testing is None:
            multiple_pair_testing = 'uncorrected'
        if multiple_pair_testing.lower() == 'bonferroni' or \
           multiple_pair_testing.lower() == 'fwer':
            significant = p_values < (alpha / n_tests)
            inference_descr += ('p < {:<.5g}'.format(alpha) +
                                ', Bonferroni-corrected for ' +
                                str(n_tests) +
                                ' model-pair comparisons')
        elif multiple_pair_testing.lower() == 'fdr':
            ps = batch_to_vectors(np.array([p_values]))[0][0]
            ps = np.sort(ps)
            criterion = alpha * (np.arange(ps.shape[0]) + 1) / ps.shape[0]
            k_ok = ps < criterion
            if np.any(k_ok):
                k_max = np.max(np.where(ps < criterion)[0])
                crit = criterion[k_max]
            else:
                crit = 0
            significant = p_values < crit
            inference_descr += ('FDR q < {:<.5g}'.format(alpha) +
                                ' (' + str(n_tests) +
                                ' model-pair comparisons)')
        else:
            if 'uncorrected' not in multiple_pair_testing.lower():
                raise Exception(
                    'plot_model_comparison: Argument ' +
                    'multiple_pair_testing is incorrectly defined as ' +
                    multiple_pair_testing + '.')
            significant = p_values < alpha
            inference_descr = (inference_descr +
                               'p < {:<.5g}'.format(alpha) +
                               ', uncorrected (' + str(n_tests) +
                               ' model-pair comparisons)')
        if result.cv_method in ['bootstrap_rdm', 'bootstrap_pattern',
                                'bootstrap_crossval']:
            inference_descr = inference_descr + \
                '\nInference by bootstrap resampling ' + \
                '({:<,.0f}'.format(n_bootstraps) + ' bootstrap samples) of '
        if result.cv_method == 'bootstrap_rdm':
            inference_descr = inference_descr + 'subjects. '
        elif result.cv_method == 'bootstrap_pattern':
            inference_descr = inference_descr + 'experimental conditions. '
        elif result.cv_method in ['bootstrap', 'bootstrap_crossval']:
            inference_descr = inference_descr + \
                'subjects and experimental conditions. '

    # %% Compute the errorbars
    if error_bars is True:
        error_bars = 'sem'
    if error_bars.lower() == 'sem':
        errorbar_low = np.std(evaluations, axis=0)
        errorbar_high = np.std(evaluations, axis=0)
    elif error_bars[0:2].lower() == 'ci':
        if len(error_bars) == 2:
            CI_percent = 95
        else:
            CI_percent = int(error_bars[2:])
        prop_cut = (1 - CI_percent / 100) / 2
        framed_evals = np.concatenate(
            (np.tile(np.array((-np.inf, np.inf)).reshape(2, 1), (1, n_models)),
             evaluations), axis=0)
        errorbar_low = -(np.quantile(framed_evals, prop_cut, axis=0)
                         - perf)
        errorbar_high = (np.quantile(framed_evals, 1 - prop_cut, axis=0)
                         - perf)
        limits = np.concatenate((errorbar_low, errorbar_high))
        if np.isnan(limits).any() or (abs(limits) == np.inf).any():
            raise Exception(
                'plot_model_comparison: Too few bootstrap samples for the ' +
                'requested confidence interval: ' + error_bars + '.')
    elif error_bars:
        raise Exception('plot_model_comparison: Argument ' +
                        'error_bars is incorrectly defined as '
                        + error_bars + '.')

    # %% Print description of inferential methods
    inference_descr += 'Error bars indicate the'
    if error_bars[0:2].lower() == 'ci':
        inference_descr += ' ' + str(CI_percent) + '% confidence interval.'
    elif error_bars.lower() == 'sem':
        inference_descr += ' standard error of the mean.'
    if test_above_0 or test_below_noise_ceil:
        inference_descr += '\nOne-sided comparisons of each model performance '
    if test_above_0:
        inference_descr += 'against 0 '
    if test_above_0 and test_below_noise_ceil:
        inference_descr += 'and '
    if test_below_noise_ceil:
        inference_descr += 'against the lower-bound estimate of the noise ceiling '
    if test_above_0 or test_below_noise_ceil:
        inference_descr += ('are Bonferroni-corrected for ' +
                            str(n_models) + ' models.\n')
    inference_descr += 'Inter-RDM distances were measured by the '
    if result.method == 'corr':
        inference_descr += (
            'Pearson correlation distance '
            + '(proportional to squared Euclidean distance'
            + ' after RDM centering and divisive normalization).')
    elif result.method == 'cosine':
        inference_descr += (
            'cosine distance '
            + '(proportional to squared Euclidean distance after RDM divisive normalizaton). ')
    else:
        raise Exception('pyrsa.vis.map_model_comparison: result.method ' +
                        result.method + ' not yet handled.')
    inference_descr += 'Inter-RDM distances are mapped as '
    if RDM_dist_measure.lower() == 'corr':
        inference_descr += (
            'Pearson correlation distance '
            + '(proportional to squared Euclidean distance'
            + ' after RDM centering and divisive normalization).')
    elif RDM_dist_measure.lower() == 'cosine':
        inference_descr += (
            'cosine distance '
            + '(proportional to squared Euclidean distance after RDM divisive normalizaton). ')
    elif RDM_dist_measure.lower() == 'euclidean':
        inference_descr += (
            'Euclidean distance '
            + '(proportional to the square root of correlation or cosine distance'
            + ' if RDMs were appropriately normalized).')

    print(inference_descr)

    # Define the model colors
    if colors is None:  # no color passed...
        colors = np.array([0, 0.4, 0.9, 1])[None, :]  # use default blue
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
                                        np.array(range(n_col)) /
                                        (n_col - 1) * (n_models - 1),
                                        colors[:, c])
        colors = cols2
    # if there is no alpha channel, make opaque
    if colors.shape[1] == 3:
        colors = np.concatenate((colors, np.ones((colors.shape[0], 1))),
                                axis=1)
    if colors.shape[0] == 1:
        colors = np.tile(colors, (n_models, 1))

    # %% Compute data-model distance estimates
    # We assume performance estimates p are correlations or cosines.
    # We convert to correlation distance or cosine distance: 1-p.
    # Correlation distance and cosine distance are porportional to squared
    # Euclidean distances after the respective approriate pattern
    # normalizations.
    # For each model, we estimate the data-model distance as
    # d = sqrt( (1-p) - (1-u) ) = sqrt(u-p),
    # where p is the performance and u the upper bound of the noise ceiling.

    noise_lower = np.nanmean(noise_ceiling[0])
    noise_upper = np.nanmean(noise_ceiling[1])

    # use upper noise ceiling as reference
    # (conservative: distances will still be positively biased)
    # data_model_dists = np.sqrt( 2 * (noise_upper - perf) )
    # noise_halo_rad = np.sqrt( 2 * (noise_upper - noise_lower) )

    # estimate true model-data distances
    # (assuming noise and intersubject variation displace RDMs orthogonally to
    # the model-brain RDM-space axis)
    print('\nEstimating the bias due to noise of the inter-RDM distance estimates...')
    model_data_dist_bias_corr = 'bootstrap'  # 'SE' or 'bootstrap'
    if rdms_data is None:
        print('No data RDMs passed. Omitting noise correction.'
              + ' Data-model RDM distances will be positively biased.')
        bias_of_sq_data_model_dist = 0
    else:
        N = rdms_data.n_rdm
        if result.method in ['corr', 'cosine']:
            bias_of_sq_data_model_dist = 2 * (1 - noise_upper) / (N - 1)
            print(
                'SE-based bias_of_sq_data_model_dist: {:.4f}'.format(bias_of_sq_data_model_dist))
            if model_data_dist_bias_corr == 'bootstrap':
                n_bootstrap = 500
                rdms = rdms_data.dissimilarities
                if result.method == 'corr':
                    rdms -= rdms.mean(axis=1, keepdims=True)
                rdms /= np.sqrt(np.einsum('ij,ij->i', rdms, rdms))[:, None]
                mean_rdm = rdms.mean(axis=0, keepdims=True)
                if result.method == 'corr':
                    mean_rdm -= mean_rdm.mean(axis=1)
                mean_rdm /= np.sqrt(np.sum(mean_rdm**2))
                index_vals = np.array(range(N))
                SSQD = 0
                print(
                    'Bootstrapping to estimate the bias of the squared data-model RDM distances...',
                    flush=True)
                sys.stdout.flush()
                for bootstrap_i in tqdm(range(n_bootstrap)):
                    indices = np.random.choice(index_vals, N)
                    rdms_bs = rdms_data.dissimilarities[indices, :]
                    mean_rdm_bs = rdms_bs.mean(axis=0, keepdims=True)
                    if result.method == 'corr':
                        mean_rdm_bs -= mean_rdm_bs.mean(axis=1)
                    mean_rdm_bs /= np.sqrt(np.sum(mean_rdm_bs**2))
                    SSQD += np.sum((mean_rdm_bs - mean_rdm)**2)
                bias_of_sq_data_model_dist = SSQD / n_bootstrap
                print(
                    '\nBootstrap-based bias_of_sq_data_model_dist: {:.4f}'.format(
                        bias_of_sq_data_model_dist))
        else:
            raise Exception(
                'pyrsa.vis.map_model_comparison:'
                + ' RDM comparison method must be "corr" or "cosine" for current implementation.')

    # DEBUG
    # bias_of_sq_data_model_dist = 0  # no correction: upper bound is at the center (0)
    # bias_of_sq_data_model_dist = 2*(noise_upper - noise_lower)  # collapse
    # the noise ceiling: lower bound is at the center (0)

    if RDM_dist_measure.lower() in ['corr', 'cosine']:
        dist_pow = 1
    elif RDM_dist_measure.lower() == 'euclidean':
        dist_pow = 1 / 2
    data_model_dists = (2 * (noise_upper - perf) -
                        bias_of_sq_data_model_dist)**dist_pow
    noise_halo_rad = (2 * (noise_upper - noise_lower) -
                      bias_of_sq_data_model_dist)**dist_pow

    # %% Compute intermodel distances
    n_dissim = models[0].rdm.shape[0]
    modelRDMs = np.empty((n_models, n_dissim))
    for model_i in range(n_models):
        modelRDMs[model_i, :] = models[model_i].rdm

    if result.method == 'corr':
        modelRDMs = modelRDMs - modelRDMs.mean(axis=1, keepdims=True)
    modelRDMs /= np.sqrt(np.einsum('ij,ij->i', modelRDMs, modelRDMs))[:, None]
    intermodelDists = ssd.squareform(
        ssd.pdist(modelRDMs, metric='euclidean'))**(2 * dist_pow)
    # the below yield identical results...
    # intermodelDists2 = np.sqrt(2*(1 - np.einsum('ik,jk', modelRDMs, modelRDMs)))
    # intermodelDists3 = ssd.squareform(np.sqrt(2*ssd.pdist(modelRDMs, metric='correlation')))

    # %% Assemble second-order distance matrix (distances among RDMs)
    rdm_dists = np.zeros((n_models + 1, n_models + 1))
    rdm_dists[1:, 0] = data_model_dists
    rdm_dists[0, 1:] = data_model_dists
    rdm_dists[1:, 1:] = intermodelDists
    rdm_dists[np.eye(n_models + 1) == 1] = 0

    # optionally use squared distances (e.g. correlation distances, rather than their square roots)
    # rdm_dists = rdm_dists**2

    plt.figure(figsize=(fig_width, fig_width), dpi=dpi)
    plt.imshow(rdm_dists, cmap='Greys')
    plt.title(
        'Matrix of inter-RDM distances\n(0: data RDM, 1..n: model RDMs)', fontsize=fs_large)
    plt.colorbar()
    plt.xticks(fontsize=fs_small)
    plt.yticks(fontsize=fs_small)
    plt.show()

    # %% Perform MDS to map model RDMs around the data RDM
    MDS_method = 'custom'  # 'custom', 'weighted'
    print('\nPerforming MDS to map model RDMs around the data RDM, using ' +
          MDS_method + ' MDS...', flush=True)
    sys.stdout.flush()
    if MDS_method == 'custom':
        locs2d = custom_MDS(rdm_dists, n_init=10, n_iter=500)
    elif MDS_method == 'weighted':
        locs2d = weighted_MDS(rdm_dists, n_weightings=1, n_MDS_runs=400)

    # ensure canonical reflection
    if bool(locs2d[1, 0] < locs2d[2, 0]) == fliplr:
        locs2d[:, 0] *= -1

    # show Shepard plot and rubberband plot of mapping distortions
    r, r_model_data, Spearman_r_model_data = show_Shepard_plot(
        locs2d, rdm_dists, models, colors)
    print('Pearson r(RDM dist, 2d-map dist): {:.4f}'.format(r))
    print(
        'Pearson r(RDM dist, 2d-map dist) for model-data dists: {:.4f}'.format(r_model_data))
    print(
        'Spearman r(RDM dist, 2d-map dist) for model-data dists: {:.4f}'.format(
            Spearman_r_model_data))

    # %% Draw the map with error bars
    rdm_dot_size = 200  # size of dots representing RDMs [pt]
    noise_halo_col = [0.85, 0.85, 0.85, 1]  # color of the noise halo
    orbit_col = [0.85, 0.85, 0.85, 0.4]   # color of the orbits
    ns_lw = 6  # line width for non-significance cords and rays
    ns_col = [0, 0, 0, 0.15]  # color for non-significance cords and rays

    # prepare axes for map
    l, b, w, h = 0.15, 0.15, 0.85, 0.85
    plt.figure(figsize=(fig_width, fig_width), dpi=dpi)
    ax = plt.axes((l, b, w, h))
    r_max = np.max(np.sqrt(np.sum(np.array(locs2d)**2, axis=1)))
    clearance_fac = 1.08  # factor by which the maximum radius is multiplied
    # to determine the inner edge of the model labels...
    rng = r_max * clearance_fac  # ...and minimum extent shown in the axes.
    # plt.axis([-rng, rng, -rng, rng])
    # ensure axes shows the largest orbit completely
    plt.scatter([-rng, 0, rng, 0], [0, rng, 0, -rng], c='none')

    # plot non-significance arches
    for i in range(n_models - 1):
        for j in range(i + 1, n_models):
            if not significant[i, j]:
                # draw non-significance arch
                xi, yi = np.array(locs2d[i + 1]).squeeze()
                xj, yj = np.array(locs2d[j + 1]).squeeze()
                rad_i, rad_j = np.sqrt(xi**2 + yi**2), np.sqrt(xj**2 + yj**2)
                angle_i, angle_j = np.arctan2(xi, yi), np.arctan2(xj, yj)
                angle_diff = angle_j - angle_i  # pos if angle_j is greater
                if (angle_diff > 0 and abs(angle_diff) <= np.pi) or \
                   (angle_diff <= 0 and abs(angle_diff) > np.pi):
                    # clockwise from i to j is shorter
                    angles = np.linspace(
                        angle_i, angle_i + min(abs(angle_diff), 2 * np.pi - abs(angle_diff)), 360)
                    radii = np.linspace(rad_i, rad_j, 360)
                else:
                    # clockwise from j to i is shorter
                    angles = np.linspace(
                        angle_j, angle_j + min(abs(angle_diff), 2 * np.pi - abs(angle_diff)), 360)
                    radii = np.linspace(rad_j, rad_i, 360)
                xx, yy = np.sin(angles) * radii, np.cos(angles) * radii
                plt.plot(xx, yy, color=ns_col, linewidth=ns_lw)

    # plot orbits, relatedness and distinctness tests, error bars, and model
    # labels
    for model_i in range(n_models):
        # orbits
        v = locs2d[model_i + 1]
        rad = np.sqrt(v * v.T)
        orbit = plt.Circle((0, 0), rad, color='none', ec=orbit_col, lw=1)
        ax.add_artist(orbit)
        # plt.plot(v[0, 0], v[0, 1], ms=200, color='r')

        # indicate whether model RDM is significantly related to data RDM
        if not model_significant[model_i]:
            x0, y0 = locs2d[model_i + 1, 0], locs2d[model_i + 1, 1]
            x1, y1 = np.array((x0, y0)) / np.sqrt(np.sum(x0 **
                                                         2 + y0**2)) * r_max * clearance_fac * 0.98
            # plot outer non-significance ray
            plt.plot([x0, x1], [y0, y1], color=ns_col, linewidth=ns_lw)

        # indicate whether model RDM is significantly distinct from data RDM
        if not model_below_lower_bound[model_i]:
            x, y = locs2d[model_i + 1, 0], locs2d[model_i + 1, 1]
            # plot outer non-significance ray
            plt.plot([0, x], [0, y], color=ns_col, linewidth=ns_lw)

        # error bars
        vn = v / rad
        errbar_dist_low = (2 * (noise_upper - (perf[model_i] - errorbar_low[model_i]))
                           - bias_of_sq_data_model_dist)**dist_pow
        errbar_dist_high = (2 * (noise_upper - (perf[model_i] + errorbar_high[model_i]))
                            - bias_of_sq_data_model_dist)**dist_pow
        low_high = np.array((errbar_dist_low, errbar_dist_high))

        edlx = vn[0, 0] * low_high
        edly = vn[0, 1] * low_high
        plt.plot(edlx, edly, color=colors[model_i])

        # model labels
        tx = vn[0, 0] * r_max * clearance_fac
        ty = vn[0, 1] * r_max * clearance_fac
        if label_orientation == 'tangential':
            angle_deg = (np.arctan2(vn[0, 1], vn[0, 0]) /
                         np.pi * 180) % 180 - 90  # tangential
            horAlign = 'center'
            verAlign = 'bottom' if ty > 0 else 'top'
        elif label_orientation == 'radial':
            angle_deg = (np.arctan2(vn[0, 1], vn[0, 0]) /
                         np.pi * 180 - 90) % 180 - 90  # radial
            horAlign = 'right' if tx <= 1e-5 else 'left'
            verAlign = 'center'
        plt.text(tx, ty, models[model_i].name, va=verAlign, ha=horAlign,
                 rotation_mode='anchor', family='sans-serif', size=fs, rotation=angle_deg)
        # plt.plot(tx, ty, 'r.')

    # plot the data RDM with its noise halo
    noise_halo = plt.Circle((0, 0), noise_halo_rad,
                            color=noise_halo_col, zorder=10)
    ax.add_artist(noise_halo)
    plt.scatter(0, 0, s=rdm_dot_size, c='k', zorder=20)

    # plot the model RDMs
    x, y = np.array(locs2d[1:, 0]).squeeze(), np.array(locs2d[1:, 1]).squeeze()
    plt.scatter(x, y, s=rdm_dot_size, c=colors, zorder=10)

    # add a scalebar
    approx_frac_of_max = 0.15
    dists_2d_vec = ssd.pdist(locs2d)
    rdm_dists_vec = ssd.squareform(rdm_dists)
    distortion_facs = dists_2d_vec / rdm_dists_vec
    percent_covered = 100
    prop_cut = (1 - percent_covered / 100) / 2
    qnts = np.quantile(distortion_facs, [prop_cut, 1 - prop_cut], axis=0)
    np.max(distortion_facs)
    dist_2d_max = np.max(dists_2d_vec)
    scalebar_length = round(approx_frac_of_max * dist_2d_max * 10) / 10
    plt.plot([-rng, -rng + scalebar_length],
             [-rng, -rng], color='k', linewidth=6)
    plt.plot([-rng + scalebar_length * qnts[0], -rng + scalebar_length * qnts[1]], [-rng, -rng],
             color=[0.5, 0.5, 0.5], linewidth=2)
    descr = '{:.1f} [a.u.]\n'.format(scalebar_length)
    if RDM_dist_measure.lower() == 'corr':
        descr += ('Pearson corr. dist.')
    elif RDM_dist_measure.lower() == 'cosine':
        descr += ('cosine dist.')
    elif RDM_dist_measure.lower() == 'euclidean':
        descr += ('normalized-pattern Eucl. dist.')
    plt.text(-rng + 0.5 * scalebar_length, -rng * 1.02, descr, va='top', ha='center',
             family='sans-serif', size=fs_small)
    # ax.autoscale()
    plt.axis('equal')
    plt.axis('off')
    plt.show()


# Custom multidimensional scaling
def custom_MDS(rdm_dists, n_init=100, n_iter=500):
    """ Performs multidimensional scaling (MDS) using the metric stress cost
    function (sum of squared distance deviations) for the intermodel RDM
    distances while exactly preserving the model-data RDM distances.
    Assumes that the data RDM has index 0 in the passed vectorized matrix of
    RDM distances (second-order distances). The data RDM is placed at the
    origin. The best fitting model is placed straight above it at the exact
    RDM distance. Since the remaining models must be placed at radii exactly
    matching the model-data RDM distances, only the angles are free parameters.
    The remaining model-RDMs are placed in random order. Each is placed
    by line search to minimize the sum of squared deviations from the RDM-
    distance to the already placed models. After this initial round of
    placements, each model (except the best one, which remains in its initial
    position) is re-placed in random order. Models are adjusted until
    convergence in random order. The entire process, including the
    initialization, is repeated n_init times. The best arrangement (the one
    with minimum sum of squared deviations) is returned."""

    # Preparations
    n_models = rdm_dists.shape[0] - 1
    best_model_i = np.argmin(rdm_dists[0, 1:])
    other_model_is = [i for i in range(0, n_models) if i != best_model_i]
    rdm_dists_vec = ssd.squareform(rdm_dists)
    ssqd_min = np.inf
    two = 2  # minimizes sum(abs(errors)**two)

    # Run repeatedly with random initialization
    for init_i in tqdm(range(n_init)):
        locs2d = np.full((n_models + 1, 2), np.NaN)
        # place data RDM at the origin and the best model straight above it
        locs2d[0] = 0, 0
        # ...and best model straight above it
        locs2d[best_model_i + 1] = 0, rdm_dists[0, best_model_i + 1]
        # Initialize by locally optimal placing of each of the remaining models
        # in random order
        rand_perm_other_model_is = np.random.permutation(other_model_is)
        for model_i in rand_perm_other_model_is:
            locs2d[model_i + 1,
                   :] = place_model(model_i, locs2d, rdm_dists, 1, two)
        # Initialize by placing the remaining models at random angles
        # (This random-angle approach is less reliable than the random-order initialization above.)
        # for model_i in other_model_is:
        #     angle = np.random.rand() * 2 * np.pi
        #     rad = rdm_dists[0, model_i+1]
        #     locs2d[model_i+1, :] = np.sin(angle) * rad, np.cos(angle) * rad

        # Adjust each model (except the best) in random order
        n_scales, n_scales_max = 1, 2
        for iter_i in range(n_iter):
            locs2d_prev = locs2d.copy()
            rand_perm_other_model_is = np.random.permutation(other_model_is)
            for model_i in rand_perm_other_model_is:
                locs2d[model_i + 1,
                       :] = place_model(model_i, locs2d, rdm_dists, n_scales, two)
            if np.max(abs(locs2d_prev - locs2d)) < 1e-8:
                print('MDS converged after {:.0f} iterations using {:.0f} scales.'.format(
                    iter_i, n_scales))
                n_scales += 1
                if n_scales > n_scales_max:
                    break
        dists_2d = ssd.pdist(locs2d, metric='euclidean')
        ssqd = np.sum(abs(rdm_dists_vec - dists_2d)**two)
        if ssqd < ssqd_min:
            ssqd_min = ssqd
            # r = np.corrcoef(rdm_dists_vec, dists_2d)[0, 1]
            r_0fixed = np.sum(rdm_dists_vec * dists_2d) \
                / (np.sum(rdm_dists_vec ** 2) + np.sum(dists_2d ** 2))
            locs2d_best = locs2d.copy()
            print(' SSQD: {:.4f}, corr_0fixed(RDM dist, map dist): {:.4f}'.format(
                ssqd, r_0fixed))
        if iter_i == n_iter - 1:
            print(' MDS did not converge. Doubling number of iterations.')
            n_iter *= 2
    return np.matrix(locs2d_best)


def place_model(model_i, locs2d, rdm_dists, n_scales=3, two=2):
    """ place a model

    Args:
        model_i : TYPE
            DESCRIPTION.
        locs2d : TYPE
            DESCRIPTION.
        rdm_dists : TYPE
            DESCRIPTION.
        n_scales : TYPE, optional
            DESCRIPTION. The default is 3.
        two : TYPE, optional
            DESCRIPTION. The default is 2.

    """
    n_angles = 180
    n_models = rdm_dists.shape[0] - 1
    radius = rdm_dists[0, model_i + 1]
    start, stop = 0, 2 * np.pi

    for scale_i in range(n_scales):
        angles = np.linspace(start, stop, n_angles)
        cand_locs = np.concatenate([np.sin(angles)[:, None], np.cos(angles)[:, None]], axis=1) \
            * radius
        dists_2d = np.sqrt(np.sum(
            (cand_locs.reshape(n_angles, 1, 2) -
             locs2d.reshape(1, n_models + 1, 2)) ** 2,
            axis=2))
        ssqd = np.nansum(abs(
            dists_2d - rdm_dists[model_i + 1, :].reshape(1, n_models + 1)) ** two, axis=1)
        best_angle_i = np.argmin(ssqd)
        start, stop = angles[(best_angle_i + np.array([-1, 1])) % n_angles]
        n_angles = int(n_angles / 3)

    return cand_locs[best_angle_i, :]


def weighted_MDS(rdm_dists, n_weightings=1, n_MDS_runs=100):
    """Perform MDS using the general function for weighted MDS with very high
    weight assigned to the model-data RDM distances"""
    n_rdms = rdm_dists.shape[0]
    n_models = n_rdms - 1
    n_rdm_pairs = int((n_rdms**2 - n_rdms) / 2)
    pyrsa_rdm_dists = pyrsa.rdm.RDMs(ssd.squareform(rdm_dists))
    rdm_dists_vec = ssd.squareform(rdm_dists)

    h = 1e2   # high weight assigned to model-data RDM distances
    l = 1  # low weight assigned to model-model RDM distances
    W = np.zeros((n_models + 1, n_models + 1))
    W[1:, 0] = h
    W[0, 1:] = h
    W[1:, 1:] = l
    W[np.eye(n_models + 1) == 1] = 0
    w = ssd.squareform(W).reshape((1, n_rdm_pairs))
    # plt.imshow(W)
    # plt.colorbar()
    # plt.show()

    r = 0
    n_weightings = 1
    print('Optimizing the mapping with weighted MDS...')
    for weighting_i in range(n_weightings):
        for mds_run_i in tqdm(range(n_MDS_runs),
                              desc='{:.0f} of {:.0f}'.format(weighting_i + 1, n_weightings)):
            # perform weighted MDS
            xy_try = pyrsa.vis.mds(pyrsa_rdm_dists, dim=2, weight=w)
            xy_try = np.matrix(xy_try.squeeze())
            r_try = np.corrcoef(rdm_dists_vec, ssd.pdist(
                xy_try, metric='euclidean'))[0, 1]
            Spearman_r_model_data_try = sst.spearmanr(
                rdm_dists_vec[:n_models],
                ssd.pdist(xy_try, metric='euclidean')[:n_models]).correlation
            # print(r, Spearman_r_model_data)
            if r_try > r:
                print('  r(dist RDM, dists_2d) = {:.3f}'.format(r_try))
                if Spearman_r_model_data_try == 1.0:
                    r = r_try
                    # Spearman_r_model_data = Spearman_r_model_data_try
                    xy = xy_try
                    print('Improved map preserves model ranks:'
                          + ' r(dist RDM, dists_2d) = {:.3f}'.format(r))
        w = w ** 0.9

    if r == 0:
        raise Exception('pyrsa.vis.map_model_comparison:'
                        + ' Could not find map that preserves model performance ranks.')

    # Center arrangement on the data RDM and orient the best model upward
    xy = xy - xy[0]  # center on the data RDM
    best_model_i = np.argmin(rdm_dists[1:, 0])
    xy_best = xy[best_model_i + 1]

    baseVec1 = np.matrix(xy_best / np.sqrt(xy_best * xy_best.T))
    baseVec2 = np.matrix([baseVec1[0, 1], -baseVec1[0, 0]])
    baseVec2 *= -1
    basis = np.concatenate((baseVec2.T, baseVec1.T), axis=1)
    locs2d = xy * basis  # new basis

    # checks out: identical geometry
    # ssd.pdist(xy, metric='euclidean')
    # ssd.pdist(locs2d, metric='euclidean')

    return locs2d


def show_Shepard_plot(locs2d, rdm_dists, models, colors):
    """ Show shepard plot """
    rdm_dists_vec = ssd.squareform(rdm_dists)
    n_models = locs2d.shape[0] - 1
    dists_2d_vec = ssd.pdist(locs2d, metric='euclidean')
    dists_2d = ssd.squareform(dists_2d_vec)
    mx = max(max(dists_2d_vec), max(rdm_dists_vec))

    r = np.corrcoef(rdm_dists_vec, dists_2d_vec)[0, 1]
    r_model_data = np.corrcoef(
        rdm_dists_vec[:n_models], dists_2d_vec[:n_models])[0, 1]
    Spearman_r_model_data = sst.spearmanr(rdm_dists_vec[:n_models],
                                          dists_2d_vec[:n_models]).correlation

    # Shepard plot
    plt.figure(figsize=(fig_width, fig_width), dpi=dpi)
    plt.plot((0, mx), (0, mx), color=[0.5, 0.5, 0.5, 0.5])
    plt.plot(rdm_dists_vec, dists_2d_vec, '.', color=[0, 0, 0, 0.8],
             mec=[1, 1, 1], ms=20, label='model-model distances')
    plt.plot(rdm_dists_vec[:n_models], dists_2d_vec[:n_models], '.',
             color=[0.8, 0, 0], mec=[1, 1, 1, 1], ms=20, label='model-data distances')
    plt.title('Shepard plot\n(Pearson r: {:.3f}, Pearson r for model-data dists: {:.1f})'.format(
        r, r_model_data), fontsize=fs_large)
    plt.xlabel('distance between RDMs', fontsize=fs)
    plt.ylabel('2d distance', fontsize=fs)
    plt.xticks(fontsize=fs_small)
    plt.yticks(fontsize=fs_small)
    plt.axis([0, mx, 0, mx])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(fontsize=fs)

    # plot distortion factor range
    approx_frac_of_max = 0.15
    distortion_facs = dists_2d_vec / rdm_dists_vec
    percent_covered = 100
    prop_cut = (1 - percent_covered / 100) / 2
    qnts = np.quantile(distortion_facs, [prop_cut, 1 - prop_cut], axis=0)
    dist_2d_max = np.max(dists_2d_vec)
    scalebar_length = round(approx_frac_of_max * dist_2d_max * 10) / 10
    plt.plot([0, mx], [0, mx * qnts[0]],
             color=[0.5, 0.5, 0.5, 0.2], linewidth=1)
    plt.plot([0, mx], [0, mx * qnts[1]],
             color=[0.5, 0.5, 0.5, 0.2], linewidth=1)
    plt.plot([scalebar_length, scalebar_length], [0, scalebar_length],
             color=[0, 0, 0, 0.7], linewidth=6)
    plt.plot([scalebar_length, scalebar_length],
             [scalebar_length * qnts[0], scalebar_length * qnts[1]],
             color=[0.5, 0.5, 0.5], linewidth=2)

    # distance-distortion plot
    plt.figure(figsize=(fig_width, fig_width), dpi=dpi)
    r_max = np.max(np.sqrt(np.sum(np.array(locs2d)**2, axis=1)))
    rng = r_max * 1.2  # ...and minimum extent shown in the axes.
    # plt.axis([-rng, rng, -rng, rng])
    # ensure axes shows the largest orbit completely
    plt.scatter([-rng, 0, rng, 0], [0, rng, 0, -rng], c='none')

    # model labels
    r_max = np.max(np.sqrt(np.sum(np.array(locs2d)**2, axis=1)))
    clearance_fac = 1.05  # factor by which the maximum radius is multiplied
    for model_i in range(n_models):
        v = locs2d[model_i + 1]
        rad = np.sqrt(v * v.T)
        vn = v / rad
        tx = vn[0, 0] * r_max * clearance_fac
        ty = vn[0, 1] * r_max * clearance_fac
        angle_deg = (np.arctan2(vn[0, 1], vn[0, 0]) /
                     np.pi * 180) % 180 - 90  # tangential
        horAlign = 'center'
        verAlign = 'bottom' if ty > 0 else 'top'
        plt.text(tx, ty, models[model_i].name, va=verAlign, ha=horAlign,
                 rotation_mode='anchor', family='sans-serif',
                 size=fs, rotation=angle_deg)

    # plot the data RDM
    rdm_dot_size = 50
    plt.scatter(0, 0, s=rdm_dot_size, c='k', zorder=20)

    # plot the distortion-indicating rubberbands
    stretched_col = np.array([0, 0.5, 1])
    squeezed_col = np.array([0.8, 0, 0])
    correct_col = np.array([0.5, 0.5, 0.5])
    lw_undistorted = 3

    # slackstringplot(rdm_dists)
    for i in range(n_models - 1):
        for j in range(i + 1, n_models):
            distortion = dists_2d[i + 1, j + 1] - rdm_dists[i + 1, j + 1]
            w = abs(distortion) / np.max(abs(dists_2d_vec - rdm_dists_vec))
            if distortion <= 0:
                col = w * squeezed_col + (1 - w) * correct_col
            else:
                col = w * stretched_col + (1 - w) * correct_col
            # if abs(distortion) < 0.1:
            #     col = correct_col
            # else:
            #     col = stretched_col if distortion > 0 else squeezed_col
            lw = lw_undistorted / dists_2d[i +
                                           1, j + 1] * rdm_dists[i + 1, j + 1]
            plt.plot(locs2d[[i + 1, j + 1], 0], locs2d[[i + 1, j + 1], 1],
                     color=col, linewidth=lw, solid_capstyle='round')

    # plot the model RDMs
    x, y = np.array(locs2d[1:, 0]).squeeze(), np.array(locs2d[1:, 1]).squeeze()
    plt.scatter(x, y, s=rdm_dot_size, c=colors, zorder=10)

    plt.title('Rubberband plot of inter-RDM-distance distortions\n' +
              '(thin & blue: stretched, thick & red: squeezed)', fontsize=fs_large)
    plt.axis('equal')
    plt.axis('off')
    plt.show()

    return r, r_model_data, Spearman_r_model_data