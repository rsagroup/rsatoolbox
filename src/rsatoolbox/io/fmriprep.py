"""Tools to navigate output of fmriprep, the fmri preprocessing pipeline
"""
from __future__ import annotations
from typing import TYPE_CHECKING, List, Dict, Optional
from rsatoolbox.io.bids import BidsLayout
if TYPE_CHECKING:
    from rsatoolbox.io.bids import BidsMriFile


def find_fmriprep_runs(
        bids_root_path: str,
        tasks: Optional[List[str]]=None) -> List[FmriprepRun]:
    """Find all sub/ses/task/run entries for which there is a preproc_bold file
    """
    bids = BidsLayout(bids_root_path)
    files = bids.find_mri_derivative_files(
        derivative='fmriprep',
        tasks=tasks,
        desc='preproc_bold'
    )
    return [FmriprepRun(f) for f in files]


class FmriprepRun:
    """Represents a single fmriprep BOLD run and metadata
    """

    boldFile: BidsMriFile

    def __init__(self, boldFile: BidsMriFile) -> None:
        self.boldFile = boldFile

    def get_data(self):
        return self.boldFile.get_data()

    def get_events(self):
        return self.boldFile.get_events()
    
    def get_meta(self):
        return self.boldFile.get_meta()
    
    def get_brain_mask(self):
        return self.boldFile.get_mri_sibling(desc='brain_mask').get_data()
    
    def get_parcellation(self):
        return self.boldFile.get_mri_sibling(desc='aparcaseg').get_data()
    
    # def get_confounds(self):
    #     return self.boldFile.get_tsv_sibling(desc='confounds').get_data() ## tsv
    
    def to_descriptors(self) -> Dict:
        """Get dictionary of dataset, observation and channel- level descriptors

        self.descriptors = parse_input_descriptor(descriptors)
        -> shape, meta


        self.obs_descriptors = parse_input_descriptor(obs_descriptors)
        -> events

        self.channel_descriptors = parse_input_descriptor(channel_descriptors)
        -> vox index

        Returns:
            Dict: kwargs for DatasetBase
        """
        return dict()
    
    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}>'
