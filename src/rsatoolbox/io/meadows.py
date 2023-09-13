"""Covers import of data downloaded from the
`Meadows online behavior platform <https://meadows-research.com/>`_.


For information on available file types see the meadows
`documentation on downloads <https://meadows-research.com/documentation\
/researcher/downloads/>`_.
"""
from __future__ import annotations
from os.path import basename
from typing import TYPE_CHECKING, Dict, Union, Tuple, List
import json
import warnings
from rsatoolbox.io.petnames import PETNAMES
import numpy
from scipy.io import loadmat
from rsatoolbox.rdm.rdms import RDMs
if TYPE_CHECKING:
    from numpy.typing import NDArray
    InfoDict = Dict[str, Union[str, int]]
    RdmsComps = Tuple[NDArray, List[str], List[str], List[str], List[int]]


def load_rdms(fpath: str, sort: bool=True) -> RDMs:
    """Read a Meadows results file and return any RDMs as an rsatoolbox object

    Args:
        fpath (str): path to .mat Meadows results file
        sort (bool): whether to sort the RDM based on the stimulus names

    Raises:
        ValueError: Will raise an error if the file is missing an expected
            variable. This can happen if the file does not contain MA task
            data.

    Returns:
        RDMs: All rdms found in the data file as an RDMs object
    """
    info = extract_filename_segments(fpath)
    if info['filetype'] == 'mat':
        utvs, stimuli, pnames, tnames, tidx = load_rdms_comps_mat(fpath, info)
    elif info['filetype'] == 'json':
        utvs, stimuli, pnames, tnames, tidx = load_rdms_comps_json(fpath, info)
    else:
        raise ValueError('Unsupported file type')

    conds = [f.split('.')[0] for f in stimuli]

    rdm_descriptors = {}
    rdm_descriptors['participant'] = pnames
    if tnames is not None:
        rdm_descriptors['task'] = tnames
    if tidx is not None:
        rdm_descriptors['task_index'] = tidx

    rdms = RDMs(
        utvs,
        dissimilarity_measure='euclidean',
        descriptors=dict(experiment_name=info['experiment_name']),
        rdm_descriptors=rdm_descriptors,
        pattern_descriptors=dict(conds=conds),
    )
    if sort:
        rdms.sort_by(conds='alpha')
    return rdms


def load_rdms_comps_mat(fpath: str, info: InfoDict) -> RdmsComps:
    """Load rdms components from a Meadows mat file

    Args:
        fpath (str): full file path
        info (InfoDict): dictionary describing the file name

    Raises:
        ValueError: File missing variable: X

    Returns:
        Tuple[NDArray, List[str], List[str], List[str], List[int]]: tuple of rdms components
    """
    data = loadmat(fpath)
    tidx = None
    tnames = None
    if info['participant_scope'] == 'single':
        for var in ('stimuli', 'rdmutv'):
            if var not in data:
                raise ValueError(f'File missing variable: {var}')
        utvs = data['rdmutv']
        stimuli = data['stimuli']
        pnames = [info['participant']]
        tidx = [int(info['task_index'])]
    else:
        stim_vars = [v for v in data.keys() if v[:7] == 'stimuli']
        stimuli = data[stim_vars[0]]
        pnames = ['-'.join(v.split('_')[1:]) for v in stim_vars]
        utv_vars = ['rdmutv_' + p.replace('-', '_') for p in pnames]
        utvs = numpy.squeeze(numpy.stack([data[v] for v in utv_vars]))
        tnames = [info['task_name']] * len(pnames)
    return utvs, stimuli, pnames, tnames, tidx


def load_rdms_comps_json(fpath: str, info: InfoDict) -> RdmsComps:
    """Load rdms components from a Meadows json file

    Args:
        fpath (str): full file path
        info (InfoDict): dictionary describing the file name

    Raises:
        ValueError: Multi-participant json files not supported yet
        ValueError: Single-task json files not supported yet
        ValueError: Unexpected structure in json file

    Returns:
        Tuple[NDArray, List[str], List[str], List[str], List[int]]: tuple of rdms components
    """
    STIM_MISMATCH = 'Varying stimuli among ma tasks, only selecting matching'

    if info['participant_scope'] == 'multiple':
        raise ValueError('Multi-participant json files not supported yet')
    if info['task_scope'] == 'single':
        raise ValueError('Single-task json files not supported yet')

    with open(fpath, encoding='utf-8') as fhandle:
        data = json.load(fhandle)

    if not isinstance(data.get('tasks'), list):
        raise ValueError('Unexpected structure in json file')

    utvs = []
    stimuli = []
    tnames = []
    tidx = []
    for t, task in enumerate(data['tasks']):
        task_meta = task.get('task', {})
        if task_meta.get('task_type') != 'multiarrange':
            continue

        task_stimuli = [s['name'] for s in task['stimuli']]
        if len(utvs) == 0:
            stimuli = task_stimuli
        else:
            if stimuli != task_stimuli:
                warnings.warn(STIM_MISMATCH)
                continue

        utvs.append(task['rdm'])
        tnames.append(task_meta['name'])
        tidx.append(t)

    utvs = numpy.asarray(utvs)
    pnames = [info['participant']] * len(tnames)
    return utvs, stimuli, pnames, tnames, tidx


def extract_filename_segments(fpath: str) -> InfoDict:
    """Get information from the name of a downloaded results file

    Will determine:
        * participant_scope: 'single' or 'multiple', how many participant
            sessions this file covers.
        * task_scope: 'single' or 'multiple', how many experiment tasks this
            file covers.
        * participant: the Meadows nickname of the participant, if this is a
            single participation file.
        * task_index: the 1-based index of the task in the experiment, if
            this is a single participant file.
        * task_name: the name of the task in the experiment, if
            this is not a single participant file.
        * version: the experiment version as a string.
        * experiment_name: name of the experiment on Meadows.
        * structure: the structure of the data contained, one of 'tree',
            'events', '1D', '2D', etc.
        * filetype: the file extension and file format used to serialize the
            data.

    Args:
        fpath (str): File system path to downloaded file

    Returns:
        dict: Dictionary with the fields described above.
    """
    fname, ext = basename(fpath).split('.')
    segments = fname.split('_')
    info = dict(
        version=segments[3].replace('v', ''),
        experiment_name=segments[1],
        structure=segments[-1],
        filetype=ext
    )
    if segments[-2].isdigit():
        info['task_scope'] = 'single'
        info['participant_scope'] = 'single'
        info['participant'] = segments[-3]
        info['task_index'] = int(segments[-2])
    elif is_petname(segments[-2]):
        info['task_scope'] = 'multiple'
        info['participant_scope'] = 'single'
        info['participant'] = segments[-2]
    else:
        info['task_scope'] = 'single'
        info['participant_scope'] = 'multiple'
        info['task_name'] = segments[-2]
    return info


def is_petname(segment: str) -> bool:
    """Check whether the given string matches a name as generated by Petname

    Args:
        segment (str): potential name

    Returns:
        bool: True if the string is a petname
    """
    if '-' in segment:
        parts = segment.split('-')
        if len(parts) == 2:
            if parts[1] in PETNAMES:
                return True
    return False
