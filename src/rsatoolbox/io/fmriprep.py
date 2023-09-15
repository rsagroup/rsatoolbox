"""Tools to navigate output of fmriprep, the fmri preprocessing pipeline
"""
from __future__ import annotations
from typing import TYPE_CHECKING, List, Dict, Optional
from rsatoolbox.io.bids import BidsLayout
if TYPE_CHECKING:
    from numpy.typing import NDArray
    from rsatoolbox.io.bids import BidsFile
    from numpy.typing import NDArray


def find_fmriprep_runs(
        bids_root_path: str,
        tasks: Optional[List[str]]=None,
        space='MNI152NLin2009cAsym') -> List[FmriprepRun]:
    """
    find all sub/ses/task/run entries for which we have a bold entry
    then find related: (by space [T1w])

    """
    bids = BidsLayout(bids_root_path)
    files = bids.find_derivative_files(
        derivative='fmriprep',
        tasks=tasks,
        space=space,
        desc='preproc_bold'
    )
    return [FmriprepRun(f) for f in files]


class FmriprepRun:
    """Represents a single fmriprep BOLD run and metadata

    - events
    - meta
    - brain mask (other desc)
    - aparc+seg (other desc) 
    """
    def __init__(self, bids: BidsFile) -> None:
        pass

    def get_data(self) -> NDArray:
        pass
    
    def to_descriptors(self) -> Dict:
        """_summary_

        self.descriptors = parse_input_descriptor(descriptors)
        self.obs_descriptors = parse_input_descriptor(obs_descriptors)
        self.channel_descriptors = parse_input_descriptor(channel_descriptors)

        Returns:
            Dict: _description_
        """
        return dict()
