.. _operations:

Operations on RDMs
==================

The library provides various ways to manipulate RDMs object. 


Matrices and Vectors
--------------------


Concatenate and split
---------------------


Sort and order
--------------


Missing data
------------

If you have several RDMs, but they don't all cover all conditions,
you may want to expand them into larger RDMs with missing values,
so that you can compare them or perform other operations on them
that require them to have the same dimensions. This can be achieved
with the :meth:`~pyrsa.rdm.rdms.RDMs.expand` method:

.. code-block:: python

    from numpy import array
    from pyrsa.rdm.rdms import RDMs
    rdms1 = RDMs(
        array(1, 2, 3),
        pattern_descriptors={'conds': ['a', 'b', 'c']}
    )
    rdms2 = RDMs(
        array(6, 7, 8),
        pattern_descriptors={'conds': ['b', 'c', 'd']}
    )
    partial_rdms = RDMs.expand()
    partial_rdms.n_conds  ## this is now 4
