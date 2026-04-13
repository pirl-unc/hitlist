# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gene symbol / Ensembl ID resolution with HGNC synonym support.

Translates user input (a current symbol, an old/alias symbol, or an
Ensembl gene ID) into the set of identifiers that appear in the
observations table's ``gene_name`` and ``gene_id`` columns.

Usage::

    from hitlist.genes import resolve_gene_query

    spec = resolve_gene_query("MART-1")   # → {"names": {"MLANA"}, "ids": set()}
    spec = resolve_gene_query("ENSG00000120337")
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

_HGNC_SEARCH_URL = "https://rest.genenames.org/search"
_HGNC_FETCH_URL = "https://rest.genenames.org/fetch"


def _cache_path() -> Path:
    from .downloads import data_dir

    d = data_dir() / "gene_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d / "hgnc_lookups.json"


def _load_cache() -> dict:
    p = _cache_path()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    p = _cache_path()
    p.write_text(json.dumps(cache, indent=2, default=str) + "\n")


def _fetch_hgnc(query: str, timeout: int = 10) -> list[dict]:
    """Search HGNC for a symbol / alias / previous symbol match."""
    q = urllib.parse.quote(query)
    url = f"{_HGNC_SEARCH_URL}/symbol:{q}+OR+alias_symbol:{q}+OR+prev_symbol:{q}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
    except Exception:
        return []
    return d.get("response", {}).get("docs", []) or []


@lru_cache(maxsize=1024)
def resolve_hgnc_symbol(query: str) -> tuple[str, ...]:
    """Return the canonical HGNC symbols matching a query (via name/alias/prev).

    Results are cached on disk under ``~/.hitlist/gene_cache/`` to avoid
    repeated REST calls.
    """
    if not query:
        return ()
    cache = _load_cache()
    cached = cache.get(query)
    if cached is not None:
        return tuple(cached.get("symbols", []))

    docs = _fetch_hgnc(query)
    symbols = tuple(d["symbol"] for d in docs if d.get("symbol"))
    cache[query] = {
        "symbols": list(symbols),
        "resolved_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_cache(cache)
    return symbols


def _is_ensembl_gene_id(s: str) -> bool:
    return s.startswith("ENSG") and s[4:].replace(".", "").isdigit()


def resolve_gene_query(
    query: str,
    use_hgnc: bool = True,
) -> dict[str, set[str]]:
    """Turn a user-supplied gene string into a set of gene_name + gene_id matchers.

    Handles:
    - Ensembl gene IDs (``"ENSG00000120337"``) → added to ``ids``
    - Current HGNC symbols (``"PRAME"``) → added to ``names``
    - Previous / alias symbols (``"MART-1"`` → ``"MLANA"``) via HGNC REST

    Returns
    -------
    dict
        ``{"names": {...}, "ids": {...}}`` — rows match if their
        ``gene_name`` is in ``names`` OR ``gene_id`` is in ``ids``.
    """
    names: set[str] = set()
    ids: set[str] = set()

    if not query:
        return {"names": names, "ids": ids}

    stripped = query.strip()
    if not stripped:
        return {"names": names, "ids": ids}

    # Multiple values — comma-split
    if "," in stripped:
        for piece in stripped.split(","):
            piece = piece.strip()
            if not piece:
                continue
            sub = resolve_gene_query(piece, use_hgnc=use_hgnc)
            names |= sub["names"]
            ids |= sub["ids"]
        return {"names": names, "ids": ids}

    if _is_ensembl_gene_id(stripped):
        ids.add(stripped)
        return {"names": names, "ids": ids}

    # Always treat as a potential symbol (exact match first)
    names.add(stripped)
    names.add(stripped.upper())  # Ensembl symbols are usually uppercase

    if use_hgnc:
        synonyms = resolve_hgnc_symbol(stripped)
        for sym in synonyms:
            names.add(sym)

    return {"names": names, "ids": ids}
