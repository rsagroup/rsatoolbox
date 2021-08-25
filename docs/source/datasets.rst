.. _datasets:

Defining the data set
=====================
The first step in an RSA is to bring the data into the correct format. The RSA toolbox uses an own Class, ``rsatoolbox.data.Dataset``.
The main content of such a dataset object is a measurement by channel matrix of measured data. Additionally it allows for descriptor variables
for the measurements, channels and the whole data object, which are added as python dictionaries.

The simplest method for generating a dataset object is based on a numpy array of data in the right format. Then you can simply call the
`Dataset` constructor to generate the object. For example, the following code creates a dataset with 10 random observations of 6 channels:

.. code-block:: python

    import numpy, rsatoolbox
    data = rsatoolbox.data.Dataset(numpy.random.rand(10, 6))

To add descriptors to the dataset, we need to define a dictionary of them with lists with one entry for each measurement of channel.
As an example, the following variation of the code above adds a descriptor which says that the 10 measurements were taken from 5 stimuli
and which ones correspond to which stimulus and adds a label 'l' vs. 'r' for left and right measurement channels:

.. code-block:: python

    import numpy, rsatoolbox
    side = ['l', 'l', 'l', 'r', 'r', 'r']
    stimulus = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    data = rsatoolbox.data.Dataset(
        numpy.random.rand(10, 6),
        channel_descriptors={'side': side},
        obs_descriptors={'stimulus': stimulus})

These descriptors are used by donwnstream processing of the data to define how the measurements are combined into RDMs and can be used for
manipulating the data before RDM creation as well. It is thus convenient to add all meta-information you might need to the dataset object.

To manipulate the datasets, have a look at the functions of the dataset object
``sort_by``, ``split_channel``, ``split_obs``, ``subset_channel``, ``subset_obs``.

The dataset objects can also be saved to hdf5 files using their method ``save`` as in and loaded with the ``rsatoolbox.data.load_dataset`` function:

.. code-block:: python

    data.save('test.hdf5')
    data_loaded = rsatoolbox.data.load_dataset('test.hdf5')



TODO: TIPS TO IMPORT FMRI / EEG ETC DATA
