# refnd.core

## Graph construction

```{eval-rst}
.. autoclass:: refnd.core.EdgeStore
   :members:

.. autoclass:: refnd.core.CsrGraph
   :members:
```

## HNSW approximate nearest neighbours

```{eval-rst}
.. autoclass:: refnd.core.HNSWConfig
   :members:

.. autoclass:: refnd.core.HNSWState
   :members:

.. autoclass:: refnd.core.HNSWIndex
   :members:
```

## Graph algorithms

```{eval-rst}
.. autoclass:: refnd.core.LeidenObjective
   :members:

.. autofunction:: refnd.core.find_communities

.. autofunction:: refnd.core.connected_components

.. autofunction:: refnd.core.partition
```

## Exact search

```{eval-rst}
.. autofunction:: refnd.core.exact_edges

.. autofunction:: refnd.core.exact_nearest_neighbors
```
