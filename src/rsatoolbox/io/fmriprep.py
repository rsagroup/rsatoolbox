"""Tools to navigate output of fmriprep, the fmri preprocessing pipeline
"""
from __future__ import annotations
from os.path import relpath, join
from typing import TYPE_CHECKING, List, Dict, Optional, Tuple
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
        tasks: Optional[List[str]] = None) -> List[FmriprepRun]:
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

    @property
    def run(self):
        return self.boldFile.run

    @property
    def sub(self):
        return self.boldFile.sub

    @property
    def ses(self):
        return self.boldFile.ses

    def __init__(self, boldFile: BidsMriFile) -> None:
        self.boldFile = boldFile

    def get_data(self, masked: bool = False):
        data = self.boldFile.get_data()
        if masked:
            return data[self.get_mask(), :]
        else:
            return data.reshape([-1, data.shape[-1]])

    def get_events(self):
        return self.boldFile.get_events()

    def get_meta(self):
        return self.boldFile.get_meta()

    def get_mask(self) -> NDArray:
        mask_file = self.boldFile.get_mri_sibling(desc='brain', suffix='mask')
        return mask_file.get_data().astype(bool)

    def get_confounds(self, cf_names: Optional[List[str]] = None) -> DataFrame:
        """_summary_

        Returns:
            DataFrame: _description_
        """
        cf_names = cf_names or ['global_signal', 'csf', 'white_matter',
                                'trans_x', 'trans_y', 'trans_z', 'rot_x',
                                'rot_y', 'rot_z']
        confounds_file = self.boldFile.get_table_sibling(
            desc='confounds', suffix='timeseries')
        df = confounds_file.get_frame()
        return df[cf_names]

    def get_parcellation(self):
        parc_file = self.boldFile.get_mri_sibling(
            desc='aparcaseg', suffix='dseg')
        return parc_file.get_data().astype(int)

    def get_parcellation_labels(self):
        parc_file = self.boldFile.get_mri_sibling(
            desc='aparcaseg', suffix='dseg')
        return parc_file.get_key().get_frame().set_index('index')

    def to_descriptors(
        self,
        collapse_by_trial_type: bool = False,
        masked: bool = False
    ) -> Dict:
        """Get dictionary of dataset, observation and
        channel- level descriptors

        Returns:
            Dict: kwargs for DatasetBase with keys:
                descriptors: sub, ses, run and task BIDS entities
                obs_descriptors: trial_type from BIDS events
                channel_descriptors: empty
        """
        return dict(
            descriptors=self.get_dataset_descriptors(),
            obs_descriptors=self.get_obs_descriptors(collapse_by_trial_type),
            channel_descriptors=self.get_channel_descriptors(masked)
        )

    def get_dataset_descriptors(self) -> Dict:
        ds_descs = dict()
        ds_descs['sub'] = self.boldFile.sub
        if self.boldFile.ses:
            ds_descs['ses'] = self.boldFile.ses
        if self.boldFile.run:
            ds_descs['run'] = self.boldFile.run
        if self.boldFile.run:
            ds_descs['task'] = self.boldFile.task
        return ds_descs

    def get_obs_descriptors(
            self, collapse_by_trial_type: bool = False
    ) -> Dict:
        obs_descs = dict()
        trial_type = self.boldFile.get_events()['trial_type']
        if collapse_by_trial_type:
            obs_descs['trial_type'] = trial_type.unique()
        else:
            obs_descs['trial_type'] = trial_type.values
        return obs_descs

    def get_channel_descriptors(self, masked: bool = False) -> Dict:
        parc_ix_3d = self.get_parcellation()
        if masked:
            parc_ix = parc_ix_3d[self.get_mask()]
        else:
            parc_ix = parc_ix_3d.ravel()
        labels_df = self.get_parcellation_labels()
        labels = labels_df.loc[parc_ix]['name'].values
        return dict(aparcaseg=labels)

    def __repr__(self) -> str:
        fmriprep_prefix = join('derivatives', 'fmriprep')
        fp_path = relpath(self.boldFile.relpath, fmriprep_prefix)
        return f'<{self.__class__.__name__} [{fp_path}]>'


def make_design_matrix(
        events: DataFrame,
        tr: float,
        n_vols: int,
        confounds: Optional[DataFrame]
) -> Tuple[NDArray, NDArray, int]:
    """Create a matrix of HRF-convolved predictors from BIDS events

    Args:
        events (DataFrame): BIDS-style table of events
        tr (float): Time to repeat scan in seconds
        n_vols (int): duration of the matrix (max extend beyond design)
        confounds (DataFrame): A table of BOLD confounds

    Returns:
        Tuple of:
            NDArray: volumes * conditions
            NDArray: boolean mask to signifiy predictors vs confounds
            int: degrees of freedom
    """
    block_dur = numpy.median(events.duration)

    # convolve a standard HRF to the block shape in the design
    STANDARD_TR = 0.1
    hrf = numpy.convolve(HRF, numpy.ones(int(block_dur/STANDARD_TR)))

    # timepoints in block (32x)
    timepts_block = numpy.arange(0, int((hrf.size-1)*STANDARD_TR), tr)

    # resample to desired TR
    hrf = pchip(numpy.arange(hrf.size)*STANDARD_TR, hrf)(timepts_block)
    hrf = hrf / hrf.max()

    # make design matrix
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
    pred_mask = numpy.ones(dm.shape[1])

    if confounds is not None:
        assert confounds.shape[0] == n_vols
        # derivatives have n/a values for first vol
        cf = confounds.dropna(axis=1).values
        # cf = (cf - cf.mean(axis=0)) / (cf.max(axis=0) - cf.min(axis=0))
        dm = numpy.hstack([dm, cf])
        pred_mask = numpy.hstack([pred_mask, numpy.zeros(cf.shape[1])])
    dm = (dm - dm.mean(axis=0)) / (dm.max(axis=0) - dm.min(axis=0))
    dof = n_vols - dm.shape[1]

    return dm, pred_mask.astype(bool), dof
