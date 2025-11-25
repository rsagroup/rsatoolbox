"""Base class for Dataset
"""
from __future__ import annotations
from rsatoolbox.util.descriptor_utils import check_descriptor_length_error
from rsatoolbox.util.descriptor_utils import format_descriptor
from rsatoolbox.util.descriptor_utils import parse_input_descriptor
from rsatoolbox.io.hdf5 import write_dict_hdf5
from rsatoolbox.io.pkl import write_dict_pkl
from rsatoolbox.util.file_io import remove_file


class DatasetBase:
    """
    Abstract dataset class.
    Defines members that every class needs to have, but does not
    implement any interesting behavior. Inherit from this class
    to define specific dataset types

    Args:
        measurements (numpy.ndarray): n_obs x n_channel 2d-array,
        descriptors (dict):           descriptors (metadata)
        obs_descriptors (dict):       observation descriptors (all
            are array-like with shape = (n_obs,...))
        channel_descriptors (dict):   channel descriptors (all are
            array-like with shape = (n_channel,...))

    Returns:
        dataset object
    """

    def __init__(self, measurements, descriptors=None,
                 obs_descriptors=None, channel_descriptors=None,
                 check_dims=True):
        if measurements.ndim != 2:
            raise AttributeError(
                "measurements must be in dimension n_obs x n_channel")
        self.measurements = measurements
        self.n_obs, self.n_channel = self.measurements.shape
        if check_dims:
            check_descriptor_length_error(obs_descriptors,
                                          "obs_descriptors",
                                          self.n_obs
                                          )
            check_descriptor_length_error(channel_descriptors,
                                          "channel_descriptors",
                                          self.n_channel
                                          )
        self.descriptors = parse_input_descriptor(descriptors)
        self.obs_descriptors = parse_input_descriptor(obs_descriptors)
        self.channel_descriptors = parse_input_descriptor(channel_descriptors)

    def __repr__(self):
        """
        defines string which is printed for the object
        """
        return (f'rsatoolbox.data.{self.__class__.__name__}(\n'
                f'measurements = \n{self.measurements}\n'
                f'descriptors = \n{self.descriptors}\n'
                f'obs_descriptors = \n{self.obs_descriptors}\n'
                f'channel_descriptors = \n{self.channel_descriptors}\n'
                )

    def __str__(self):
        """
        defines the output of print
        """
        string_desc = format_descriptor(self.descriptors)
        string_obs_desc = format_descriptor(self.obs_descriptors)
        string_channel_desc = format_descriptor(self.channel_descriptors)
        if self.measurements.shape[0] > 5:
            measurements = self.measurements[:5, :]
        else:
            measurements = self.measurements
        return (f'rsatoolbox.data.{self.__class__.__name__}\n'
                f'measurements = \n{measurements}\n...\n\n'
                f'descriptors: \n{string_desc}\n\n'
                f'obs_descriptors: \n{string_obs_desc}\n\n'
                f'channel_descriptors: \n{string_channel_desc}\n'
                )

    def __eq__(self, other: object) -> bool:
        """Equality check, to be implemented in the specific
        Dataset class

        Args:
            other (DatasetBase): The object to compare to.

        Raises:
            NotImplementedError: This is not valid if not implemented
                by the specific Dataset class

        Returns:
            bool: Never returns
        """
        if isinstance(other, DatasetBase):
            raise NotImplementedError()
        else:
            return False

    def copy(self) -> DatasetBase:
        """Copy Dataset
        To be implemented in child class

        Raises:
            NotImplementedError: raised if not implemented

        Returns:
            DatasetBase: Never returns
        """
        raise NotImplementedError

    def split_obs(self, by):
        """ Returns a list Datasets split by obs

        Args:
            by(String): the descriptor by which the splitting is made

        Returns:
            list of Datasets, splitted by the selected obs_descriptor

        """
        raise NotImplementedError(
            "split_obs function not implemented in used Dataset class!")

    def split_channel(self, by):
        """ Returns a list Datasets split by channels

        Args:
            by(String): the descriptor by which the splitting is made

        Returns:
            list of Datasets,  splitted by the selected channel_descriptor

        """
        raise NotImplementedError(
            "split_channel function not implemented in used Dataset class!")

    def subset_obs(self, by, value):
        """ Returns a subsetted Dataset defined by certain obs value

        Args:
            by(String): the descriptor by which the subset selection is made
                from obs dimension
            value:      the value by which the subset selection is made
                from obs dimension

        Returns:
            Dataset, with subset defined by the selected obs_descriptor

        """
        raise NotImplementedError(
            "subset_obs function not implemented in used Dataset class!")

    def subset_channel(self, by, value):
        """ Returns a subsetted Dataset defined by certain channel value

        Args:
            by(String): the descriptor by which the subset selection is made
                from channel dimension
            value:      the value by which the subset selection is made
                from channel dimension

        Returns:
            Dataset, with subset defined by the selected channel_descriptor

        """
        raise NotImplementedError(
            "subset_channel function not implemented in used Dataset class!")

    def save(self, filename, file_type='hdf5', overwrite=False):
        """ Saves the dataset object to a file

        Args:
            filename(String): path to the file
                [or opened file]
            file_type(String): Type of file to create:
                hdf5: hdf5 file
                pkl: pickle file
            overwrite(Boolean): overwrites file if it already exists

        """
        data_dict = self.to_dict()
        if overwrite:
            remove_file(filename)
        if file_type == 'hdf5':
            write_dict_hdf5(filename, data_dict)
        elif file_type == 'pkl':
            write_dict_pkl(filename, data_dict)

    def to_dict(self):
        """ Generates a dictionary which contains the information to
        recreate the dataset object. Used for saving to disc

        Returns:
            data_dict(dict): dictionary with dataset information

        """
        data_dict = {}
        data_dict['measurements'] = self.measurements
        data_dict['descriptors'] = self.descriptors
        data_dict['obs_descriptors'] = self.obs_descriptors
        data_dict['channel_descriptors'] = self.channel_descriptors
        data_dict['type'] = type(self).__name__
        return data_dict
