.. _operations:

Operations on RDMs
==================

The rsatoolbox provides various ways to manipulate RDMs objects.


Matrices and Vectors
--------------------
To access the underlying RDMs in vectorized or matrix form you can use
the methods ``get_vectors`` and ``get_matrices``. The RDMs are returned
as 2D or 3D numpy arrays respectively.

Subset and Split
----------------
There are 2x2 methods for extracting parts of the RDMs object:
``subset``, ``subsample``, ``subset_pattern``, and ``subsample_pattern``.
All four methods take the name of a descriptor and the values to be selected as input.

``subset`` and ``subsample`` select some of the RDMs, ``subset_pattern`` and ``subsample_pattern``
select some of the patterns/conditions. The ``subset`` variant ignores repetitions in the allowed set,
the subsample function uses as many repetitions as provided in the set.

For example ``rdms.subset('index', [0,2,2])`` selects only the rdms number 0 and 2
and will thus contain 2 RDMs, while ``rdms.subsample('index', [0,2,2])`` repeats rdm number 2 and thus contains 3 RDMs.

When patterns are repeated in ``subsample_pattern`` the resulting RDM will contain entries
that correspond to the similarity of a condition to itself. These are set to ``NaN``.

Concatenate
-----------
For concatenating RDMs there is a function called ``rsatoolbox.rdm.concat``,
which takes a list of RDMs as input and returns a RDMs object containing all RDMs.

Also RDMs objects have a method ``append``, which allows appending a single other RDMs object.

Only RDMs objects with an equal number of conditions can be concatenated.


Missing data
------------

If you have several RDMs, but they don't all cover all conditions,
you may want to expand them into larger RDMs with missing values,
so that you can compare them or perform other operations on them
that require them to have the same dimensions. This can be achieved
with the :meth:`~rsatoolbox.rdm.combine.from_partials` function:

.. code-block:: python

    from numpy import array
    from rsatoolbox.rdm.rdms import RDMs
    from rsatoolbox.rdm.combine import from_partials
    rdms1 = RDMs(
        array(1, 2, 3),
        pattern_descriptors={'conds': ['a', 'b', 'c']}
    )
    rdms2 = RDMs(
        array(6, 7, 8),
        pattern_descriptors={'conds': ['b', 'c', 'd']}
    )
    partial_rdms = from_partials([rdms1, rdms2])
    partial_rdms.n_conds  ## this is now 4

Sort and reorder
----------------
To change the order of conditions/patterns in the RDMs object there are two functions
``reorder`` and ``sort_by``.

``reorder`` expects a new order of conditions/patterns, e.g. ``rdms.reorder([1,2,0])``
will change the order of conditions, moving the first condition to the end.

``sort_by`` sorts conditions according to a descriptor, e.g. ``rdms.sort_by(condition='alpha')``
sorts the conditions according to the 'condition' descriptor alphanumerically.

**Caution:** Both ``reorder`` and ``sort_by`` operate in place, i.e. the RDMs object rdms is changed by calling them!


Transformations
---------------
To transform all RDM entries by a function ``rsatoolbox`` offers specific functions
for the most common transformations and a general ``transform`` function, which takes the
function to be applied as input. These functions are available in ``rsatoolbox.rdm``

The specific transformations are: ``rank_transform``, ``positive_transform``, and ``sqrt_transform``.
They take only a RDMs object as input and compute a rank transform, set all negative values to 0, or compute a square root of each value respectively.

For example:

.. code-block:: python

    rdms_rank = rsatoolbox.rdm.rank_transform(rdms)

will produce a rank transformed version of the data in ``rdms``

The general ``rsatoolbox.rdm.transform`` function takes a function to be applied as an input and can thus
implement any transform on the RDM.

To compute the square of each RDM entry you could use the following code for example:

.. code-block:: python

    def square(x):
        return x ** 2
    rdms_square = rsatoolbox.rdm.transform(rdms, square)

The function you pass must take a 2D numpy array of vectorized RDMs as input and return an array of equal shape.
