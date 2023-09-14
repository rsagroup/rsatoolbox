"""Tools to navigate output of fmriprep, the fmri preprocessing pipeline
"""
from __future__ import annotations
from typing import TYPE_CHECKING, List, Dict
if TYPE_CHECKING:
    from numpy.typing import NDArray


def find_fmriprep_runs(bids_root_path: str) -> List[FmriprepRun]:
    """
    find all sub/ses/task/run entries for which we have a bold entry
    then find related: (by space [T1w])

    """
    ## call io.bids.xyz here
    return []


class FmriprepRun:
    """Represents a single fmriprep BOLD run and metadata

    - events
    - meta
    - brain mask (other desc)
    - aparc+seg (other desc) 
    """

    def get_data(self) -> NDArray:
        pass
    
    def to_descriptors(self) -> Dict:
        return dict()
