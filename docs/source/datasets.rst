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

Datasets can also be created (and converted to) DataFrame objects from the pandas library:

.. code-block:: python

    df = data_in.to_DataFrame()
    data_out = Dataset.from_DataFrame(df)

The dataset objects can also be saved to hdf5 files using their method ``save`` as in and loaded with the ``rsatoolbox.data.load_dataset`` function:

.. code-block:: python

    data.save('test.hdf5')
    data_loaded = rsatoolbox.data.load_dataset('test.hdf5')


.. _TemporalDatasets:

Temporal data sets
--------------------

Datasets with a temporal dimension are represented by the class ``rsatoolbox.data.TemporalDataset``. This class is a subclass of the
``rsatoolbox.data.Dataset`` class. The main difference is that the TemporalDataset expects ``measurements`` of shape 
``(n_observations, n_channels, n_timepoints)`` and has descriptors for the temporal dimension (``time_descriptor``).

As an example, we assume to have measured data from 10 trials, each with six EEG channels and a timecourse of 2s 
(from -.5 to 1.5 seconds, stimulus onset at 0 seconds).


.. code-block:: python

    import numpy, rsatoolbox

    channel_names = ['Oz', 'O1', 'O2', 'PO3', 'PO4', 'POz']  # channel names
    stimulus = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4] # stimulus idx, each stimulus was presented twice

    sampling_rate = 30 # in Hz
    t = numpy.arange(-.5, 1.5, 1/sampling_rate) # time vector

    n_observations = len(stimulus)
    n_channels = len(channel_names)
    n_timepoints = len(t)

    measurements = numpy.random.randn(n_observations, n_channels, n_timepoints)  # random data

    data = rsatoolbox.data.TemporalDataset(
        measurements,
        channel_descriptors={'names': channel_names},
        obs_descriptors={'stimulus': stimulus},
        time_descriptors={'time': t}
        )

Beyond the functions to manipulate the data provided by ``rsatoolbox.data.Dataset``, the ``rsatoolbox.data.TemporalDataset`` class provides the following functions:
``split_time``, ``subset_time``, ``bin_time``, ``convert_to_dataset``.


.. _Spiking_Data

Spiking Data
------------

Import of spiking data into rsatoolbox tends to be relatively easy, because the data often comes in the form of numpy arrays already.
You can find an example in :ref:`demo_spikes<demo_spikes.nblink>` .

