from __future__ import annotations
from typing import Dict
from os.path import basename
from rsatoolbox.data.dataset import TemporalDataset
try:
    from mne import read_epochs
except Exception:
    pass


def load_epochs(fpath: str) -> TemporalDataset:
    epo = read_epochs(fpath, preload=True, verbose='error')
    fname = basename(fpath)
    descs = dict(filename=fname)
    fname_descs = descriptors_from_bids_filename(fname)
    return TemporalDataset(
        measurements=epo.get_data(),
        descriptors={**descs, **fname_descs},
        obs_descriptors=dict(triggers=epo.events[:, 2]),
        channel_descriptors=dict(name=epo.ch_names),
        time_descriptors=dict(time=epo.times)
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
