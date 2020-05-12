#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 14:04:52 2020
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pyrsa.util.inference_util import pair_tests
from pyrsa.util.rdm_utils import batch_to_vectors


def plot_model_comparison(result, alpha=0.05, plot_pair_tests='arrows',
                          multiple_testing='FDR', sort=None, colors=None,
                          error_bars='SEM', eb_alpha=0.05):
    """ plots the results of a model comparison
    Input should be a results object with model evaluations
    evaluations, which uses the bootstrap samples for confidence intervals
    and significance tests and averages over all trailing dimensions
    like cross-validation folds
    
    Args (case insensitive):
        result:
            model evaluation result (pyrsa.inference.result.Result)
        alpha:
            significance threshold (p threshold or FDR q threshold)
        plot_pair_tests: 
            False or None: do not plot pairwise model comparison results
            'arrows': plot results in arrows style
            'nili': plot results as Nili bars (Nili et al. 2014)
            'golan': plot results as Golan wings (Golan et al. 2020)
        multiple_testing:
            False or 'none': do not adjust for multiple testing
            'FDR' or 'fdr': control the false-discorvery rate at q = alpha
            'FWER',' fwer', or 'Bonferroni': control the familywise error rate 
            using the Bonferroni method
        sort:
            False or 'none': plot bars in the order passed
            'descend[ing]': plot bars in descending order of model performance
            'ascend[ing]': plot bars in ascending order of model performance
        colors:
            'none': default blue for all bars
            single color: list or numpy array of 3 or 4 values (RGB, RGBA)
                    specifying the color for all bars
            multiple colors: numpy array (number of colors by 3 or 4 channels 
                    -- RGB, RGBA). If the number of colors matches the number
                    of models, each color is used for the bar corresponding
                    to that model (in the order of the models as passed). 
                    If the number of colors does not match the number of
                    models, the list is exanded to match by linear 
                    interpolation (e.g. 2 colors will become a gradation).
        error_bars:
            'SEM': plot the standard error of the mean
            'CI': plot confidence intervals covering (1-eb_alpha)*100%
        eb_alpha: error-bar alpha, i.e. proportion of bootstrap samples outside
            the confidence interval
    
    Returns:
        ---
    
    """
    # Preparations
    evaluations = result.evaluations
    models = result.models
    noise_ceiling = result.noise_ceiling
    method = result.method
    while len(evaluations.shape) > 2:
        evaluations = np.nanmean(evaluations, axis=-1)
    evaluations = evaluations[~np.isnan(evaluations[:, 0])]
    evaluations = 1 - evaluations
    perf = np.mean(evaluations, axis=0)
    n_models = evaluations.shape[1]
    if sort==True:
        sort = 'descending'  # descending by default
    if sort and not sort.lower()=='none': # 'descending' or 'ascending'
        idx = np.argsort(perf)
        if 'descend' in sort.lower():
            idx = np.flip(idx)
        perf = perf[idx]
        evaluations = evaluations[:, idx]
        models = [models[i] for i in idx]
    if error_bars == 'CI':
        errorbar_low = -(np.quantile(evaluations, eb_alpha / 2, axis=0)
                         - perf)
        errorbar_high = (np.quantile(evaluations, 1 - (eb_alpha / 2), axis=0)
                         - perf)
    elif error_bars == 'SEM':
        errorbar_low = np.std(evaluations, axis=0)
        errorbar_high = np.std(evaluations, axis=0)
    noise_ceiling = 1 - np.array(noise_ceiling)
    # Plot bars
    l, b, w, h = 0.15, 0.15, 0.8, 0.8
    if plot_pair_tests:
        if plot_pair_tests.lower()=='arrows':        
            h_pairTests = 0.3
        else:
            h_pairTests = 0.4
        plt.figure(figsize=(12.5, 10))
        ax = plt.axes((l, b, w, h*(1-h_pairTests)))
        axbar = plt.axes((l, b + h * (1 - h_pairTests), w,
                          h * h_pairTests * 0.7))
    else:
        plt.figure(figsize=(12.5, 10))
        ax = plt.axes((l, b, w, h))
    # Define the colors for the bars
    if colors is None: # no color passed...
        colors=[0, 0.4, 0.9, 1]  # use default blue
    colors = np.array([np.array(col) for col in colors])
    if len(colors.shape)==1: # one color passed...
        n_col, n_chan = 1, colors.shape[0]
        colors.shape = (n_col, n_chan)
    else: # multiple colors passed...
        n_col, n_chan = colors.shape
        if n_col == n_models: # one color passed for each model...
            cols2 = colors
        else: # number of colors passed does not match number of models...
            # interpolate colors to define a color for each model
            cols2 = np.empty((n_models,n_chan))
            for c in range(n_chan):
                cols2[:,c] = np.interp(np.array(range(n_models)),
                               np.array(range(n_col))/n_col*n_models,
                               colors[:,c])
        if sort and not sort.lower()=='none':
            colors = cols2[idx,:]
        else:
            colors = cols2
    if colors.shape[1]==3:
        colors=np.concatenate((colors,np.ones((colors.shape[0],1))),axis=1)
    # Plot bars and error bars
    ax.bar(np.arange(evaluations.shape[1]), perf, color=colors)
    ax.errorbar(np.arange(evaluations.shape[1]), perf,
                yerr=[errorbar_low, errorbar_high], fmt='none', ecolor='k',
                capsize=0, linewidth=3)
    # Plot noise ceiling
    if noise_ceiling is not None:
        noise_min = np.nanmean(noise_ceiling[0])
        noise_max = np.nanmean(noise_ceiling[1])
        noiserect = patches.Rectangle((-0.5, noise_min), len(perf),
                                      noise_max - noise_min, linewidth=1,
                                      edgecolor=[0.5, 0.5, 0.5, 0.3],
                                      facecolor=[0.5, 0.5, 0.5, 0.3], 
                                      zorder=10e6)
        ax.add_patch(noiserect)
    # Floating axes
    fs, fs2 = 18, 14 # axis label font sizes
    ytoptick = np.ceil(min(1,ax.get_ylim()[1]) * 10) / 10
    ax.set_yticks(np.arange(0, ytoptick + 1e-6, step=0.1))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks(np.arange(n_models))
    ax.spines['left'].set_bounds(0, ytoptick)
    ax.spines['bottom'].set_bounds(0, n_models - 1)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.rc('ytick', labelsize=fs2)
    # Axis labels
    ax.text(-1.8, ytoptick/2, 'RDM prediction accuracy', 
            horizontalalignment='center', verticalalignment='center', 
            rotation='vertical', fontsize=fs, fontweight='bold') 
    if method == 'cosine':
        ax.set_ylabel('[across-subject mean of cosine similarity]', 
                      fontsize=fs2)
    if method == 'cosine_cov' or method == 'whitened cosine':
        ax.set_ylabel('[across-subject mean of whitened-RDM cosine]',
                      fontsize=fs2)
    elif method == 'Spearman' or method == 'spearman':
        ax.set_ylabel('[across-subject mean of Spearman r rank correlation]', 
                      fontsize=fs2)
    elif method == 'corr' or method == 'Pearson' or method == 'pearson':
        ax.set_ylabel('[across-subject mean of Pearson r correlation]', 
                      fontsize=fs2)
    elif method == 'corr_cov':
        ax.set_ylabel('[across-subject mean of whitened-RDM Pearson r correlation]', 
                      fontsize=fs2)
    elif method == 'kendall' or method == 'tau-b':
        ax.set_ylabel('[across-subject mean of Kendall tau-b rank correlation]', 
                      fontsize=f2s)
    elif method == 'tau-a':
        ax.set_ylabel('[across-subject mean of Kendall tau-a rank correlation]', 
                      fontsize=fs2)
    if models is not None:
        ax.set_xticklabels([m.name for m in models], fontsize=fs2,
                           rotation=45)
    # Pairwise model comparisons
    if plot_pair_tests and not plot_pair_tests=='none':
        model_comp_descr = 'Model comparisons: two-tailed, '
        res = pair_tests(evaluations)
        n_tests = int((n_models**2-n_models)/2)
        if multiple_testing.lower() == 'bonferroni' or \
           multiple_testing.lower() == 'fwer':
            significant = res < (alpha / n_tests)
            model_comp_descr = (model_comp_descr
                                + 'p < {:3.3f}'.format(alpha)
                                + ', Bonferroni-corrected for '
                                + str(n_tests)
                                + ' model-pair comparisons')
        elif multiple_testing.lower() == 'fdr':
            ps = batch_to_vectors(np.array([res]))[0][0]
            ps = np.sort(ps)
            criterion = alpha * (np.arange(ps.shape[0]) + 1) / ps.shape[0]
            k_ok = ps < criterion
            if np.any(k_ok):
                k_max = np.max(np.where(ps < criterion)[0])
                crit = criterion[k_max]
            else:
                crit = 0
            significant = res < crit
            model_comp_descr = (model_comp_descr +
                                'FDR q < {:3.3f}'.format(alpha) +
                                ' (' + str(n_tests) +
                                ' model-pair comparisons)')
        else:
            significant = res < alpha
            model_comp_descr = (model_comp_descr +
                                'p < {:3.3f}'.format(alpha) +
                                ', uncorrected (' + str(n_tests) +
                                ' model-pair comparisons)')
        if result.cv_method == 'bootstrap_rdm':
            model_comp_descr = model_comp_descr + \
            '\nInference by bootstrap resampling of subjects.'
        elif result.cv_method == 'bootstrap_pattern':
            model_comp_descr = model_comp_descr + \
            '\nInference by bootstrap resampling of experimental conditions.'
        elif result.cv_method == 'bootstrap' or result.cv_method == \
        'bootstrap_crossval':
            model_comp_descr = model_comp_descr + \
            '\nInference by bootstrap resampling of subjects and experimental conditions.'
        model_comp_descr = model_comp_descr + '\nError bars indicate the'
        if error_bars == 'CI':
            model_comp_descr = (model_comp_descr +
                                ' {:3.0f}'.format(round(1-eb_alpha)*100) +
                                '% confidence interval.')
        elif error_bars == 'SEM':
            model_comp_descr = (model_comp_descr +
                                ' standard error of the mean.')
        axbar.set_title(model_comp_descr, fontsize=fs2)
        axbar.set_xlim(ax.get_xlim())        
        if 'nili' in plot_pair_tests.lower(): 
            plot_nili_bars(axbar, significant)
        elif 'golan' in plot_pair_tests.lower():
            plot_golan_wings(axbar, significant, perf, sort, colors) 
        elif 'arrows' in plot_pair_tests.lower():
            plot_arrows(axbar, significant)


def plot_nili_bars(axbar, significant):        
    """ plots the results of the pairwise inferential model comparisons in the
    form of a set of black horizontal bars connecting significantly different
    models as in the 2014 RSA Toolbox (Nili et al. 2014).
    Args:
        axbar: Matplotlib axes handle to plot in
        significant: Boolean matrix of model comparisons
    Returns:
        ---
    """        
    k = 1
    for i in range(significant.shape[0]):
        k += 1
        for j in range(i + 1, significant.shape[0]):
            if significant[i, j]:
                axbar.plot((i, j), (k, k), 'k-', linewidth=2)
                k += 1
    axbar.set_axis_off()
    axbar.set_ylim((0, k))


def plot_golan_wings(axbar, significant, perf, sort, colors='none', 
                     always_black=False, version=3):        
    """ Plots the results of the pairwise inferential model comparisons in the
    form of black horizontal bars with a tick mark at the reference model and
    a circular bulge at each significantly different model similar to the 
    visualization in Golan, Raju, Kriegeskorte (2020).
    
    Args:
        axbar: Matplotlib axes handle to plot in
        significant: Boolean matrix of model comparisons
        version: 0 (single wing: solid circle anchor and open circles),
                 1 (single wing: tick anchor and circles), 
                 2 (single wing: circle anchor and up and down feathers)
                 3 (double wings: circle anchor,
                    downward dominance-indicating feathers,
                    from bottom to top in model order)
                 4 (double wings: circle anchor,
                    downward dominance-indicating feathers,
                    from bottom to top in performance order)
    
    Returns:
        ---
    
    """   
    # Define wing order
    n_models = significant.shape[0]
    wing_order = np.array(range(n_models)) # to the right by default 
    if 'ascend' in sort:
        wing_order = np.flip(wing_order) # to the left if bars are ascending
    if version == 4:
        wing_order = np.argsort(-perf)
    # Define vertical spacing
    bbox = axbar.get_window_extent().transformed(
                                     plt.gcf().dpi_scale_trans.inverted())
    h_inch = bbox.height     
    h = 1
    for wo_i in range(len(wing_order)):
        i = wing_order[wo_i]
        if version in [3,4]:
            js = np.concatenate((wing_order[0:wo_i], wing_order[wo_i+1:])).astype('int')
            js = js[np.logical_and(significant[i,js], perf[i]>perf[js])]
        else:
            js = wing_order[wo_i+1:][significant[i, wing_order[wo_i+1:]]]
        js = js[significant[i,js]]
        if len(js) > 0:
            h += 1
    axbar.set_axis_off()
    axbar.set_ylim((0, h))    
    # Draw the wings
    if always_black or colors is 'none' or colors is 'k' or colors.shape[0]==1:
        colors = np.tile([0,0,0,1],(n_models,1))
    tick_length_inch = 0.08
    k = 1
    for wo_i in range(len(wing_order)):
        i = wing_order[wo_i]
        if version in [3,4]:
            js = np.concatenate((wing_order[0:wo_i], wing_order[wo_i+1:])).astype('int')
            js = js[np.logical_and(significant[i,js], perf[i]>perf[js])]
        else:
            js = wing_order[wo_i+1:][significant[i, wing_order[wo_i+1:]]]
        js = js[significant[i,js]]
        if len(js) > 0:
            if version != 1:
                # circle anchor
                axbar.plot(i, k, markersize=8, marker='o',
                                 markeredgecolor=colors[i,:], 
                                 markerfacecolor=colors[i,:])            
            elif version == 1:
                # tick anchor
                axbar.plot((i, i), (k - tick_length_inch/h_inch*h, k), 'k-', 
                           linewidth=2) # tick
            for j in js:
                if version == 0:
                    axbar.plot(j, k, markersize=8, marker='o',
                                     markeredgecolor=colors[i,:], 
                                     markerfacecolor='w')                        
                elif version == 1:
                    axbar.plot(j, k, markersize=8, marker='o',
                                     markeredgecolor=colors[i,:], 
                                     markerfacecolor=colors[i,:])
                elif version in [2,3,4]:
                    if perf[i] > perf[j]:
                        tick_ver_end = k - tick_length_inch/h_inch*h
                    elif perf[i] < perf[j]:
                        tick_ver_end = k + tick_length_inch/h_inch*h
                    axbar.plot((j, j),(k, tick_ver_end),'-', linewidth=2,
                                                           color=colors[i,:])
            # Plot wing line
            axbar.plot((min(i,js.min()),max(i,js.max())), (k, k), 'k-', 
                                           linewidth=2, color=colors[i,:])
            k += 1


def plot_arrows(axbar, significant):
    """ Summarizes the significances with arrows. The argument significant is 
    a binary matrix of pairwise model comparisons. A nonzero value (or True) 
    indicates that the model specified by the row index beats the model 
    specified by the column index. Only the lower triangular part of compMat is used, so the upper
    triangular part need not be filled in symmetrically. The summary will be
    most concise if models are ordered from worst to best (top to bottom and 
    left to right).
    """
    # preparations
    [n,n] = significant.shape
    remaining = significant.copy()

    # capture as many comparisons as possible with double arrows
    double_arrows = list()
    for ambiguity_span in range(0, n-1): 
    # consider short double arrows first (these cover many comparisons)
        for i in range(n-1, ambiguity_span, -1):
            if significant[i:n, 0:i-ambiguity_span].all() and \
            remaining[i:n,0:i-ambiguity_span].any():
                # add double arrow
                double_arrows.append((i-ambiguity_span-1, i))
                remaining[i:n, 0:i-ambiguity_span] = 0
      
    # capture as many of the remaining comparisons as possible with arrows
    arrows = list()
    for dist2diag in range(1, n):
        for i in range(n-1,dist2diag-1,-1):
            if significant[i, 0:i-dist2diag+1].all() and \
            remaining[i, 0:i-dist2diag+1].any():
                arrows.append((i, i-dist2diag)) # add left arrow
                remaining[i, 0:i-dist2diag+1] = 0               
            if significant[i:n, i-dist2diag].all() and \
            remaining[i:n, i-dist2diag].any():
                arrows.append((i-dist2diag, i)) # add right arrow
                remaining[i:n, i-dist2diag] = 0
   
    # capture the remaining comparisons with lines
    lines = list()
    for i in range(1, n):
        for j in range(0, i-1):
            if remaining[i, j]:
                lines.append((i, j)) # add line
    
    # plot
    expected_n_lines = 6
    axbar.set_ylim((0, expected_n_lines))
    bbox = axbar.get_window_extent().transformed(
                                     plt.gcf().dpi_scale_trans.inverted())
    h_inch, w_inch = bbox.height, bbox.width
    dx = abs(np.diff(axbar.get_xlim()))
    dy = abs(np.diff(axbar.get_ylim()))
    ar = (dy/h_inch) / (dx/w_inch)
    occupied = np.zeros((len(double_arrows)+len(arrows)+len(lines), 3*n))  
    for m in range(0, int(np.ceil(n/2))):
        double_arrows_left = [(i,j) for (i,j) in double_arrows if i==m]
        if len(double_arrows_left):
            i, j = double_arrows_left[0]
            double_arrows.remove((i,j))
            if j < i: i, j = j, i
            k = 1
            while occupied[k-1, i*3+2:j*3].any(): k +=1
            if i==0:
                draw_hor_arrow(axbar, i, j, k, '->', ar)
            elif j==n-1:
                draw_hor_arrow(axbar, i, j, k, '<-', ar)
            else:
                draw_hor_arrow(axbar, i, j, k, '<->', ar)
            occupied[k-1, i*3+2:j*3+1] = 1

        double_arrows_right = [(i,j) for (i,j) in double_arrows if j==n-1-m]
        if len(double_arrows_right):
            i, j = double_arrows_right[0]
            double_arrows.remove((i,j))
            k = 1
            while occupied[k-1, i*3+2:j*3].any(): k +=1
            if i==0:
                draw_hor_arrow(axbar, i, j, k, '->', ar)
            elif j==n-1:
                draw_hor_arrow(axbar, i, j, k, '<-', ar)
            else:
                draw_hor_arrow(axbar, i, j, k, '<->', ar)
            occupied[k-1, i*3+2:j*3+1] = 1

    for m in range(0,int(np.ceil(n/2))):
        arrows_left = [(i,j) for (i,j) in arrows if (i<j and i==m) or 
                       (j<i and j==m)]
        if len(arrows_left):
            i, j = arrows_left[0]
            arrows.remove((i,j))
            k = 1
            while occupied[k-1, i*3+2:j*3].any(): k +=1
            draw_hor_arrow(axbar, i, j, k, '->', ar)
            occupied[k-1, i*3+2:j*3+1] = 1

        arrows_right = [(i,j) for (i,j) in arrows if (i<j and j==n-1-m) or 
                        (j<i and i==n-1-m)]
        if len(arrows_right):
            i, j = arrows_right[0]
            arrows.remove((i,j))
            k = 1
            while occupied[k-1, i*3+2:j*3].any(): k +=1
            draw_hor_arrow(axbar, i, j, k, '->', ar)
            occupied[k-1, i*3+2:j*3+1] = 1
            
    for m in range(0,int(np.ceil(n/2))):
        lines_left = [(i,j) for (i,j) in lines if i==m]
        while len(lines_left):
            i, j = lines_left.pop()
            lines.remove((i,j))
            k = 1
            while occupied[k-1, i*3+2:j*3].any(): k +=1
            axbar.plot((i, j), (k, k), 'k-', linewidth=2)
            occupied[k-1, i*3+2:j*3] = 1

        lines_right = [(i,j) for (i,j) in lines if j==n-1-m]
        while len(lines_right):
            i, j = lines_right.pop()
            lines.remove((i,j))
            k = 1
            while occupied[k-1, i*3+2:j*3].any(): k +=1
            axbar.plot((i, j), (k, k), 'k-', linewidth=2)
            occupied[k-1, i*3+2:j*3] = 1   
    if occupied.shape[0] > 0:
        h = occupied.sum(axis=1).nonzero()[0].max()+1
    else:
        h = 0
    axbar.set_ylim((0, max(expected_n_lines, h)))
    axbar.set_axis_off()
    

def draw_hor_arrow(ax, x1, x2, y, style, ar):
    hw, hl = 0.15*ar, 0.15
    s = 0.25 # shortening
    d = x2-x1
    if style == '->':
        ax.arrow(x1, y, np.sign(d)*(abs(d)-s), 0, head_width=hw, 
                 head_length=hl, length_includes_head = True, fc='k', ec='k')
        ax.plot(x1, y, 'k', markersize=8, marker='o')   
    elif style == '<-':
        ax.arrow(x2, y, np.sign(-d)*(abs(d)-s), 0, head_width=hw, 
                 head_length=hl, length_includes_head = True, fc='k', ec='k')
        ax.plot(x2, y, 'k', markersize=8, marker='o')   
    elif style == '<->':
        c = (x1+x2)/2
        l = abs(x2-x1)
        ax.arrow(c, y, +(l/2-s), 0, head_width=hw, head_length=hl, 
                 length_includes_head = True, fc='k', ec='k')
        ax.arrow(c, y, -(l/2-s), 0, head_width=hw, head_length=hl, 
                 length_includes_head = True, fc='k', ec='k')
        


    