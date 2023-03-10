"""
saving to and reading from pickle files
"""
import pickle


def write_dict_pkl(file, dictionary):
    """ writes a nested dictionary containing strings & arrays as data into
    a pickle file

    Args:
        file: a filename or opened writable file
        dictionary(dict): the dict to be saved

    """
    if isinstance(file, str):
        file = open(file, 'wb')
    dictionary['rsatoolbox_version'] = '0.0.1'
    pickle.dump(dictionary, file, protocol=-1)


def read_dict_pkl(file):
    """ writes a nested dictionary containing strings & arrays as data into
    a pickle file

    Args:
        file: a filename or opened readable file

    Returns:
        dictionary(dict): the loaded dict


    """
    if isinstance(file, str):
        file = open(file, 'rb')
    data = pickle.load(file)
    return data
