"""SatQuery AI backend package.

Web layer only: HTTP surface, schemas, storage, geospatial validation, input
interpretation, constrained task routing, asynchronous job control, artifact
serving and report building.

Model inference is deliberately **not** implemented here. The three specialists and
their preprocessing adapters sit behind the contracts in :mod:`app.models.base` and
:mod:`app.preprocessing.base`, and the worker reaches them through the single seam in
:mod:`app.workers.pipeline`. See ``docs/backend/OWNERSHIP.md``.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
