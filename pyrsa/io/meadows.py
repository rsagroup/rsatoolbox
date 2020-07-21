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
