"""Covers import of data downloaded from the
`Meadows online behavior platform <https://meadows-research.com/>`_.


For information on available file types see the meadows
`documentation on downloads <https://meadows-research.com/documentation\
/researcher/downloads/>`_.
"""
from os.path import basename
import numpy
from scipy.io import loadmat
from rsatoolbox.rdm.rdms import RDMs


def load_rdms(fpath, sort=True):
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
    data = loadmat(fpath)
    if info['participant_scope'] == 'single':
        for var in ('stimuli', 'rdmutv'):
            if var not in data:
                raise ValueError(f'File missing variable: {var}')
        utvs = data['rdmutv']
        stimuli_fnames = data['stimuli']
        pnames = [info['participant']]
    else:
        stim_vars = [v for v in data.keys() if v[:7] == 'stimuli']
        stimuli_fnames = data[stim_vars[0]]
        pnames = ['-'.join(v.split('_')[1:]) for v in stim_vars]
        utv_vars = ['rdmutv_' + p.replace('-', '_') for p in pnames]
        utvs = numpy.squeeze(numpy.stack([data[v] for v in utv_vars]))

    desc_info_keys = (
        'participant',
        'task_index',
        'task_name',
        'experiment_name'
    )
    conds = [f.split('.')[0] for f in stimuli_fnames]
    rdms = RDMs(
        utvs,
        dissimilarity_measure='euclidean',
        descriptors={k: info[k] for k in desc_info_keys if k in info},
        rdm_descriptors=dict(participants=pnames),
        pattern_descriptors=dict(conds=conds),
    )
    if sort:
        rdms.sort_by(conds='alpha')
    return rdms


def extract_filename_segments(fpath):
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
        task_scope='single',
        version=segments[3].replace('v', ''),
        experiment_name=segments[1],
        structure=segments[-1],
        filetype=ext
    )
    if segments[-2].isdigit():
        info['participant_scope'] = 'single'
        info['participant'] = segments[-3]
        info['task_index'] = int(segments[-2])
    else:
        info['participant_scope'] = 'multiple'
        info['task_name'] = segments[-2]
    return info
