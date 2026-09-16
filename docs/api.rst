API reference
=============

The NumPy entry points work with the base installation. Tensor, history, and
data helpers need the relevant optional dependencies.

Clustering
----------

.. autofunction:: flashjet.cluster

.. autoclass:: flashjet.ClusterOutput
   :members:

NumPy reference
---------------

.. autofunction:: flashjet.cluster_event

.. autoclass:: flashjet.ClusterSequenceRef
   :members:

History helpers
---------------

.. automodule:: flashjet.history
   :members: splitting_scales_from_history, exclusive_jets_from_history, lund_coordinates_from_history, groom_from_history

Data helpers
------------

.. automodule:: flashjet.data
   :members: collate, to_gpu_batches, gpu_batch_ready
