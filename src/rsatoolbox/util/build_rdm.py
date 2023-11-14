#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""helper methods to create RDMs at the end of calculations"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from copy import deepcopy
import numpy as np
from rsatoolbox.rdm.rdms import RDMs
from rsatoolbox.data import average_dataset_by

if TYPE_CHECKING:
    from rsatoolbox.data.base import DatasetBase
    from numpy.typing import NDArray


def _build_rdms(
            utv: NDArray,
            ds: DatasetBase,
            method: str,
            obs_desc_name: str | None,
            obs_desc_vals: Optional[NDArray] = None,
            cv: Optional[NDArray] = None,
            noise: Optional[NDArray] = None
        ) -> RDMs:
    rdms = RDMs(
        dissimilarities=np.array([utv]),
        dissimilarity_measure=method,
        rdm_descriptors=deepcopy(ds.descriptors)
    )
    if (obs_desc_vals is None) and (obs_desc_name is not None):
        # obtain the unique values in the target obs descriptor
        _, obs_desc_vals, _ = average_dataset_by(ds, obs_desc_name)

    if _averaging_occurred(ds, obs_desc_name, obs_desc_vals):
        orig_obs_desc_vals = np.asarray(ds.obs_descriptors[obs_desc_name])
        for dname, dvals in ds.obs_descriptors.items():
            dvals = np.asarray(dvals)
            avg_dvals = np.full_like(obs_desc_vals, np.nan, dtype=dvals.dtype)
            for i, v in enumerate(obs_desc_vals):
                subset = dvals[orig_obs_desc_vals == v]
                if len(set(subset)) > 1:
                    break
                avg_dvals[i] = subset[0]
            else:
                rdms.pattern_descriptors[dname] = avg_dvals
    else:
        rdms.pattern_descriptors = deepcopy(ds.obs_descriptors)
    # Additional rdm_descriptors
    if noise is not None:
        rdms.descriptors['noise'] = noise
    if cv is not None:
        rdms.descriptors['cv_descriptor'] = cv
    return rdms


def _averaging_occurred(
            ds: DatasetBase,
            obs_desc_name: str | None,
            obs_desc_vals: NDArray | None
        ) -> bool:
    if obs_desc_name is None:
        return False
    orig_obs_desc_vals = ds.obs_descriptors[obs_desc_name]
    return len(obs_desc_vals) != len(orig_obs_desc_vals)
