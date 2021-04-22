.. _getting_started:

Getting started
===============

The easiest way to install rsatoolbox is with pip:

.. code-block:: sh

    pip install rsatoolbox

To use rsatoolbox:

.. code-block:: python

    import numpy, rsatoolbox
    data = rsatoolbox.data.Dataset(numpy.random.rand(10, 5))
    rdms = rsatoolbox.rdm.calc_rdm(data)
    rsatoolbox.vis.show_rdm(rdms)

Also make sure your setup meets the requirements to run the toolbox with the relevant toolboxes installed (see requirements.txt). 

As in introduction, we recommend having a look at the Jupyter notebooks in ``demos``.

