"""Conversions from rsatoolbox classes to pandas table objects
"""
from __future__ import annotations
from typing import TYPE_CHECKING
from pandas import DataFrame
import numpy
if TYPE_CHECKING:
    from rsatoolbox.rdm.rdms import RDMs


def rdms_to_df(rdms: RDMs) -> DataFrame:
    """Create DataFrame representation of the RDMs object

    A column for:
    - dissimilarity
    - each pattern descriptor
    - each rdm descriptor

    Multiple RDMs are stacked row-wise.
    See also the `RDMs.to_df()` method which calls this function

    Args:
        rdms (RDMs): the object to convert

    Returns:
        DataFrame: long-form pandas DataFrame with
            dissimilarities and descriptors.
    """
    n_rdms, n_conds = rdms.dissimilarities.shape
    cols = dict(dissimilarity=rdms.dissimilarities.ravel())
    for dname, dvals in rdms.rdm_descriptors.items():
        cols[dname] = numpy.repeat(dvals, n_conds)
    for dname, dvals in rdms.pattern_descriptors.items():
        cols[dname] = numpy.tile(dvals, n_rdms)
    return DataFrame(cols)
