# py_proto.core

## Graph construction

```{eval-rst}
.. autoclass:: py_proto.core.EdgeStore
   :members:

.. autoclass:: py_proto.core.CsrGraph
   :members:
```

## HNSW approximate nearest neighbours

```{eval-rst}
.. autoclass:: py_proto.core.HNSWConfig
   :members:

.. autoclass:: py_proto.core.HNSWState
   :members:

.. autoclass:: py_proto.core.HNSWIndex
   :members:
```

## Community detection

```{eval-rst}
.. autoclass:: py_proto.core.LeidenObjective
   :members:

.. autofunction:: py_proto.core.find_communities
```

## Graph algorithms

```{eval-rst}
.. autofunction:: py_proto.core.connected_components

.. autofunction:: py_proto.core.largest_component

.. autofunction:: py_proto.core.partition
```

## Exact search

```{eval-rst}
.. autofunction:: py_proto.core.exact_edges

.. autofunction:: py_proto.core.exact_nearest_neighbors
```
