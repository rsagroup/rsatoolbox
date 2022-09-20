"""Functions to select pairs
"""
# pylint: disable=redefined-builtin
from __future__ import annotations
from typing import TYPE_CHECKING
from pandas import DataFrame
import numpy
from scipy.stats import rankdata
if TYPE_CHECKING:
    from rsatoolbox.rdm.rdms import RDMs


def pairs_by_percentile(rdms: RDMs, min: float = 0, max: float = 100,
        **kwargs) -> DataFrame:
    """Select pairs within a percentile range.

    Filter pairs first by providing the `with_pattern` argument.

    Args:
        rdms (RDMs): RDMs object
        min (float, optional): Lower percentile bound. Defaults to 0.
        max (float, optional): Upper percentile bound. Defaults to 100.
        kwargs: Pattern Descriptor value to match.

    Returns:
        DataFrame: Wide form DataFrame where each row represents a pair.
    """
    (desc, val) = list(kwargs.items())[0]
    row_mask = rdms.pattern_descriptors[desc] == val
    mats = rdms.get_matrices()
    row = mats[0, row_mask, :].squeeze()
    pair_dissims = row[~row_mask]
    percs = rankdata(pair_dissims, 'average') / pair_dissims.size * 100
    matches = numpy.logical_and(percs >= min, percs <= max)
    matches_mask = numpy.full_like(row, False, dtype=bool)
    matches_mask[~row_mask] = matches
    columns = dict()
    columns[desc] = rdms.pattern_descriptors[desc][matches_mask]
    columns['dissim'] = row[matches_mask]
    return DataFrame(columns)
