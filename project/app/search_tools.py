"""
Search tool for the agent (minsearch :class:`~minsearch.VectorSearch`).
"""
from __future__ import annotations

import math
import numbers
from typing import Any, List

from sentence_transformers import SentenceTransformer


def _json_safe(obj: Any) -> Any:
    """Tool results must be JSON-serializable for the LLM API."""
    if obj is None or isinstance(obj, bool):
        return obj
    if isinstance(obj, str):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, int) and not isinstance(obj, bool):
        return obj
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return _json_safe(obj.item())
        except Exception:
            return str(obj)
    if isinstance(obj, numbers.Integral):
        return int(obj)
    if isinstance(obj, numbers.Real):
        return _json_safe(float(obj))
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    return str(obj)


class SearchTool:
    def __init__(self, vindex, embedding_model: SentenceTransformer):
        self.vindex = vindex
        self.embedding_model = embedding_model

    def search(self, query: str) -> List[Any]:
        """
        Embed the query and run similarity search over the vector index.

        Args:
            query: Natural-language search query.

        Returns:
            Up to 5 chunk dicts, JSON-safe for the API.
        """
        q = self.embedding_model.encode(query)
        rows = self.vindex.search(q, num_results=5)
        return _json_safe(rows)
