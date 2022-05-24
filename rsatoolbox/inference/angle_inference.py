#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for doing inference on angles in representational geometry.
"""

import itertools as it
import numpy as np
from typing import List, Tuple, Union

import rsatoolbox
import rsatoolbox.data as rsd  # abbreviation to deal with dataset


# TODO: maybe refactor; have a function for computing the full XCM, and then compute the parallelism score using it,
# instead of computing one at a time? Could save compute time.


def compute_xcrossnobis(data_array: np.ndarray,
                        convert_to_cosine: bool = True,
                        neg_distance_behavior: str = 'nan') -> float:
    """Compute the x-crossnobis estimator from input data. Computed over the average of the two
    possible train/test splits.

    Args:
        data_array: Input array with shape (conditions x folds x nchannels); there must be 4 conditions and 2 folds.
            The difference vectors are: c1 --> c2, c3 --> c4.
        convert_to_cosine: Whether to normalize the x-crossnobis estimator to a cosine.
        neg_distance_behavior: How to behave in case a cross-validated distance
            is negative. Put 'nan' to ignore such observations, put 'zero'
            to set these cosines to zero (no correlations), put 'fail' to make function
            return an error, put 'keep' to keep as-is.
            The 'keep' option will just keep the values if not doing cosine normalization;
            if converting to cosines, it'll use the length of the actual difference vectors
            instead of the cross-validated distances (i.e., the XCM diagonals).

    Returns: X-crossnobis estimator.
    #TODO: account for noise covariance. Probably best to fold this into existing toolbox code for this.
    """

    if neg_distance_behavior not in ['nan', 'zero', 'keep', 'fail']:
        raise ValueError("Only supported options for handling negative distances are 'nan' and 'zero'.")

    if data_array.shape[0:2] != (4, 2):
        raise ValueError("Please specify input array with shape (conditions x folds x nchannels);"
                         "there must be 4 conditions and 2 folds.")

    diff_vec1_fold1 = data_array[1, 0, :] - data_array[0, 0, :]
    diff_vec1_fold2 = data_array[1, 1, :] - data_array[0, 1, :]
    diff_vec2_fold1 = data_array[3, 0, :] - data_array[2, 0, :]
    diff_vec2_fold2 = data_array[3, 1, :] - data_array[2, 1, :]

    # Compute for both directions.

    xcrossnobis_dir1 = np.dot(diff_vec1_fold1, diff_vec2_fold2)
    xcrossnobis_dir2 = np.dot(diff_vec1_fold2, diff_vec2_fold1)

    # Get cross-validated vector lengths.

    diff_vec1_norm = np.dot(diff_vec1_fold1, diff_vec1_fold2)
    diff_vec2_norm = np.dot(diff_vec2_fold1, diff_vec2_fold2)

    if (diff_vec1_norm <= 0) or (diff_vec2_norm <= 0):
        if neg_distance_behavior == 'fail':
            raise ValueError("One of the cross-validated distances is negative and"
                             "neg_distance_behavior was set to 'fail'")
        elif neg_distance_behavior == 'nan':
            return np.nan
        elif neg_distance_behavior == 'zero':
            return 0

    if not convert_to_cosine:  # Return as-is if no cosine normalization.
        return float(np.mean([xcrossnobis_dir1, xcrossnobis_dir2]))

    if neg_distance_behavior != 'keep':
        norm_factor = (diff_vec1_norm ** .5) * (diff_vec2_norm ** .5)
        xcrossnobis_dir1_cos = xcrossnobis_dir1 / norm_factor
        xcrossnobis_dir2_cos = xcrossnobis_dir2 / norm_factor
    elif neg_distance_behavior == 'keep':
        norm_factor1 = np.linalg.norm(diff_vec1_fold1) * np.linalg.norm(diff_vec2_fold2)
        norm_factor2 = np.linalg.norm(diff_vec1_fold2) * np.linalg.norm(diff_vec2_fold1)
        xcrossnobis_dir1_cos = xcrossnobis_dir1 / norm_factor1
        xcrossnobis_dir2_cos = xcrossnobis_dir1 / norm_factor2

    return float(np.mean([xcrossnobis_dir1_cos, xcrossnobis_dir2_cos]))


def get_dichotomy_parallelism_score(dataset: rsd.Dataset,
                                    cond_descriptor: str,
                                    split1: List[str],
                                    split2: List[str],
                                    cv_descriptor: str,
                                    fold1: Union[str, int],
                                    fold2: Union[str, int],
                                    neg_distance_behavior: str = 'nan') -> Tuple[float, List[Tuple[str, str]]]:
    """This computes a parallelism score for a given dichotomy (i.e.,
    balanced split of conditions), using a mixture of the method
    described in Bernardi et al. (2020) and the X-Crossnobis method
    that I (JohnMark) and Niko have been exploring.

    Approach:
    1) Go through each way of pairing up the stimuli in each dichotomy one-to-one.
    2) For each such pairing assignment, compute the mean (cross-validated)
        cosine angle between all the resulting difference vectors.
    3) Return the maximum such cosine, and the pairings that yielded this.

    Args:
        dataset: RSA toolbox dataset object.
        cond_descriptor: Name of descriptor to use for dividing the stimuli into two splits.
        split1: List of observation/condition descriptors in split 1.
        split2: List of observation/condition descriptors in split 2.
        cv_descriptor: Name of descriptor to use to split the dataset into two folds.
        fold1: Value of the cv_descriptor for the first CV fold.
        fold2: Value of the cv_descriptor for the first CV fold.
        neg_distance_behavior: How to behave in case a cross-validated distance
            is negative. Put 'nan' to ignore such observations, put 'zero'
            to set these cosines to zero (no correlations).

    Returns:
        Maximum parallelism score across all possible pairings, and the pairings that yielded this score.

    #TODO fix variable names, "split" and "fold" are confusing, "pairing" is confusing (1:1 pairing of stimuli,
        or pairs of difference vectors)
    #TODO: Allow multiple crossval folds, not just two.
    #TODO: Vectorize code somehow (operate over RDM instead of dataset?)
    """

    if neg_distance_behavior not in ['nan', 'zero']:
        raise ValueError("Only supported options for handling negative distances are 'nan' and 'zero'.")

    if len(split1) != len(split2):
        raise ValueError("Only balanced dichotomies are supported; make sure the two splits have same length.")
    num_conds = len(split1)

    # Now generate all possible 1:1 pairings of stimuli; keep split1 fixed, just shuffle all values
    # of split2 and get the pairings.

    pairings_list = [zip(split1, split2_shuffled) for p, split2_shuffled in
                     enumerate(it.permutations(split2))]
    pairing_parallelism_scores = []

    # Loop through pairings, for each one compute the parallelism score.

    for pairing in pairings_list:
        # For each pairing, compute the average of pairwise cosines between difference vectors.

        pairing_cosines = []
        diff_vec_pairs = it.combinations(range(num_conds), 2)
        for diff_vec1_ind, diff_vec2_ind in diff_vec_pairs:
            diff_vec1_cond1 = pairing[diff_vec1_ind][0]
            diff_vec1_cond2 = pairing[diff_vec1_ind][1]
            diff_vec2_cond1 = pairing[diff_vec2_ind][0]
            diff_vec2_cond2 = pairing[diff_vec2_ind][1]

            #  Pull out the data associated with each condition and fold (8 vectors) and compute
            #  cosine-normalized X-Crossnobis.

            fold_data = []
            for fold in [fold1, fold2]:  # TODO: how efficient is the subsetting here?
                fold_data.append(np.vstack([(dataset.subset_obs(by=cond_descriptor, value=diff_vec1_cond1)
                                             .subset_obs(by=cv_descriptor, value=fold)),
                                            (dataset.subset_obs(by=cond_descriptor, value=diff_vec1_cond2)
                                             .subset_obs(by=cv_descriptor, value=fold)),
                                            (dataset.subset_obs(by=cond_descriptor, value=diff_vec2_cond1)
                                             .subset_obs(by=cv_descriptor, value=fold)),
                                            (dataset.subset_obs(by=cond_descriptor, value=diff_vec2_cond2)
                                             .subset_obs(by=cv_descriptor, value=fold))]))
            fold_data = np.dstack(fold_data)  # 4 conditions * nchannels * 2 folds; rearrange.
            fold_data = np.swapaxes(fold_data, 1, 2)  # 4 conditions * 2 folds * nchannels
            cosine_val = compute_xcrossnobis(fold_data)
            pairing_cosines.append(cosine_val)

        pair_mean_cosine = np.nanmean(pairing_cosines)
        pairing_parallelism_scores.append(pair_mean_cosine)

    max_parallelism_score = np.max(pairing_parallelism_scores)
    best_pairing_index = pairing_parallelism_scores.index(max_parallelism_score)
    best_pairing = pairings_list[best_pairing_index]
    return max_parallelism_score, best_pairing
