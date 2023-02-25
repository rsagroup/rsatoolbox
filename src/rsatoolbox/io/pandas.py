"""Conversions from rsatoolbox classes to pandas table objects
"""
from __future__ import annotations
from typing import TYPE_CHECKING
from pandas import DataFrame
import numpy
from numpy import asarray
if TYPE_CHECKING:
    from rsatoolbox.rdm.rdms import RDMs


def rdms_to_df(rdms: RDMs) -> DataFrame:
    """Create DataFrame representation of the RDMs object

    A column for:
    - dissimilarity
    - each rdm descriptor
    - two for each pattern descriptor, suffixed by _1 and _2 respectively

    Multiple RDMs are stacked row-wise.
    See also the `RDMs.to_df()` method which calls this function

    Args:
        rdms (RDMs): the object to convert

    Returns:
        DataFrame: long-form pandas DataFrame with
            dissimilarities and descriptors.
    """
    n_rdms, n_pairs = rdms.dissimilarities.shape
    cols = dict(dissimilarity=rdms.dissimilarities.ravel())
    for dname, dvals in rdms.rdm_descriptors.items():
        # rename the default index desc as that has special meaning in df
        cname = 'rdm_index' if dname == 'index' else dname
        cols[cname] = numpy.repeat(dvals, n_pairs)
    for dname, dvals in rdms.pattern_descriptors.items():
        ix = numpy.triu_indices(len(dvals), 1)
        # rename the default index desc as that has special meaning in df
        cname = 'pattern_index' if dname == 'index' else dname
        for p in (0, 1):
            cols[f'{cname}_{p+1}'] = numpy.tile(asarray(dvals)[ix[p]], n_rdms)
    return DataFrame(cols)
