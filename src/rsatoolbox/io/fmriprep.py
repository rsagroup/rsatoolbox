"""Tools to navigate output of fmriprep, the fmri preprocessing pipeline
"""
from __future__ import annotations
from os.path import relpath, join
from typing import TYPE_CHECKING, List, Dict, Optional
import numpy
from scipy.interpolate import pchip
from rsatoolbox.io.bids import BidsLayout
from rsatoolbox.io.hrf import HRF
if TYPE_CHECKING:
    from rsatoolbox.io.bids import BidsMriFile
    from pandas import DataFrame
    from numpy.typing import NDArray


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
    
    def to_descriptors(self, collapse_by_trial_type: bool=False) -> Dict:
        """Get dictionary of dataset, observation and channel- level descriptors

        Returns:
            Dict: kwargs for DatasetBase with keys:
                descriptors: sub, ses, run and task BIDS entities
                obs_descriptors: trial_type from BIDS events
                channel_descriptors: empty
        """
        ds_descs = dict()
        ds_descs['sub'] = self.boldFile.sub
        if self.boldFile.ses:
            ds_descs['ses'] = self.boldFile.ses
        if self.boldFile.run:
            ds_descs['run'] = self.boldFile.run
        if self.boldFile.run:
            ds_descs['task'] = self.boldFile.task
        obs_descs = dict()
        if collapse_by_trial_type:
            obs_descs['trial_type'] = self.boldFile.get_events()['trial_type'].unique()
        else:
            obs_descs['trial_type'] = self.boldFile.get_events()['trial_type'].values
        return dict(
            descriptors=ds_descs,
            obs_descriptors=obs_descs,
            channel_descriptors=dict()
        )
    
    def __repr__(self) -> str:
        fmriprep_prefix = join('derivatives', 'fmriprep')
        fp_path = relpath(self.boldFile.relpath, fmriprep_prefix)
        return f'<{self.__class__.__name__} [{fp_path}]>'


def make_design_matrix(events: DataFrame, tr: float, n_vols: int) -> NDArray:
    """Create a matrix of HRF-convolved predictors from BIDS events

    Args:
        events (DataFrame): BIDS-style table of events
        tr (float): Time to repeat scan in seconds

    Returns:
        NDArray: volumes * conditions
    """
    block_dur = numpy.median(events.duration)

    ## convolve a standard HRF to the block shape in the design
    STANDARD_TR = 0.1
    hrf = numpy.convolve(HRF, numpy.ones(int(block_dur/STANDARD_TR)))

    ## timepoints in block (32x)
    timepts_block = numpy.arange(0, int((hrf.size-1)*STANDARD_TR), tr)

    # resample to desired TR
    hrf = pchip(numpy.arange(hrf.size)*STANDARD_TR, hrf)(timepts_block)
    hrf = hrf / hrf.max()

    ## make design matrix
    conditions = events.trial_type.unique()
    dm = numpy.zeros((n_vols, conditions.size))
    all_times = numpy.linspace(0, tr*(n_vols-1), n_vols)
    hrf_times = numpy.linspace(0, tr*(len(hrf)-1), len(hrf))
    for c, condition in enumerate(conditions):
        onsets = events[events.trial_type == condition].onset.values
        yvals = numpy.zeros((n_vols))
        # loop over blocks
        for o in onsets:
            # interpolate to find values at the data sampling time points
            f = pchip(o + hrf_times, hrf, extrapolate=False)(all_times)
            yvals = yvals + numpy.nan_to_num(f)
        dm[:, c] = yvals
    return dm