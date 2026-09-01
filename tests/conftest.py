"""Make bare imports (``graph``, ``ontology``, ``retrieval``, ``llm``, ``api``, ...)
resolve against ``backend/`` during test collection, matching how the app is
actually served in production (``uvicorn ... --app-dir backend``).

Without this, running pytest from the repo root puts the repo root itself on
sys.path, so a bare ``import graph...`` (as used internally by e.g.
``backend/retrieval/src/retrieval/triplet_retriever.py``) could silently
resolve against an unrelated top-level package instead of ``backend/graph``.
Inserting ``backend/`` at the front of sys.path removes that ambiguity.
"""

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
