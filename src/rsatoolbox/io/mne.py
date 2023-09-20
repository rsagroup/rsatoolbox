from __future__ import annotations
from typing import Optional, Dict, TYPE_CHECKING
from os.path import basename
from rsatoolbox.data.dataset import TemporalDataset
if TYPE_CHECKING:
    from mne.epochs import EpochsFIF


def read_epochs(fpath: str) -> TemporalDataset:
    """Create TemporalDataset from epochs in mne FIF file

    Args:
        fpath (str): Full path to epochs file

    Returns:
        TemporalDataset: dataset with epochs
    """
    # pylint: disable-next=import-outside-toplevel
    from mne import read_epochs as mne_read_epochs
    epo = mne_read_epochs(fpath, preload=True, verbose='error')
    fname = basename(fpath)
    descs = dict(filename=fname, **descriptors_from_bids_filename(fname))
    return dataset_from_epochs(epo, descs)


def dataset_from_epochs(
            epochs: EpochsFIF,
            descriptors: Optional[Dict] = None
        ) -> TemporalDataset:
    """Create TemporalDataset from MNE epochs object

    Args:
        fpath (str): Full path to epochs file

    Returns:
        TemporalDataset: dataset with epochs
    """
    descriptors = descriptors or dict()
    return TemporalDataset(
        measurements=epochs.get_data(),
        descriptors=descriptors,
        obs_descriptors=dict(event=epochs.events[:, 2]),
        channel_descriptors=dict(name=epochs.ch_names),
        time_descriptors=dict(time=epochs.times)
    )


def descriptors_from_bids_filename(fname: str) -> Dict[str, str]:
    """parse a filename for BIDS-style entities

    Args:
        fname (str): filename

    Returns:
        Dict[str, str]: sub, run or task descriptors
    """
    descs = dict()
    for dname in ['sub', 'run', 'task']:
        for segment in fname.split('_'):
            if segment.startswith(dname + '-'):
                descs[dname] = segment[len(dname)+1:]
    return descs
