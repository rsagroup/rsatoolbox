"""Functions to select pairs
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Optional, Any
from pandas import DataFrame
if TYPE_CHECKING:
    from rsatoolbox.rdm.rdms import RDMs


def pairs_by_percentile(rdms: RDMs, min: float=0, max: float=100, 
    with_pattern: Optional[Dict[str, Any]]=None) -> DataFrame:
    """Select pairs within a percentile range.

    Filter pairs first by providing the `with_pattern` argument.

    Args:
        rdms (RDMs): RDMs object
        min (float, optional): Lower percentile bound. Defaults to 0.
        max (float, optional): Upper percentile bound. Defaults to 100.
        with_pattern (Optional[Dict], optional): Pattern Descriptor value to
            match. Defaults to None.

    Returns:
        DataFrame: Wide form DataFrame where each row represents a pair.
    """
    return DataFrame()
