from os.path import basename
from scipy.io import loadmat
from pyrsa.rdm.rdms import RDMs


def load_rdms(fpath):
    """Read a Meadows results file and return any RDMs as a pyrsa object

    Args:
        fpath (str): path to .mat Meadows results file

    Raises:
        ValueError: Will raise an error if the file is missing an expected
            variable. This can happen if the file does not contain MA task 
            data.

    Returns:
        RDMs: All rdms found in the data file as an RDMs object
    """
    data = loadmat(fpath)
    for var in ('stimuli', 'rdmutv'):
        if var not in data:
            raise ValueError(f'File missing variable: {var}')
    
    return RDMs(
        data['rdmutv'],
        dissimilarity_measure='euclidean',
        descriptors=dict(),
        pattern_descriptors=dict(),
    )


def extract_filename_segments(fpath):
    """Get information from the name of a donwloaded results file

    Will determine:
    - participant_scope: 'single' or 'multiple', how many participant sessions
        this file covers
    - task_scope: 'single' or 'multiple', how many experiment tasks this file
        covers
    - participant: the Meadows nickname of the participant, if this is a 
        single participation file.
    - task_index: the 1-based index of the task in the experiment, if 
        this is a single task file.
    - version: the experiment version as a string.
    - experiment_name: name of the experiment on Meadows.
    - structure: the structure of the data contained, one of 'tree', ,
        'events', '1D', '2D', etc.
    - filetype: the file extension and file format used to serialize the data.

    Args:
        fpath (str): File system path to downloaded file

    Returns:
        dict: Dictionary with the fields described above.
    """
    fname, ext = basename(fpath).split('.')
    segments = fname.split('_')
    return dict(
        participant_scope='single',
        task_scope='single',
        participant=segments[-3],
        task_index=int(segments[-2]),
        version=segments[3].replace('v', ''),
        experiment_name=segments[1],
        structure=segments[-1],
        filetype=ext
    )
