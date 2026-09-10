"""Load hand-maintained curation YAML with duplicate keys rejected (#454).

PyYAML resolves a duplicate mapping key by keeping the last value, silently.
On a hand-maintained file that is a data-loss mode with no symptom: a second
``mhc:`` on one ``ms_samples`` record discards the first genotype, a second
``synonyms:`` on a cell line drops the first list, and every downstream check
still passes because the record is well-formed. The duplicate-*PMID* guard in
:func:`hitlist.curation.load_pmid_overrides` exists for the same failure one
level up (#438); this closes it at the key level.

Every packaged YAML loads through :func:`load_curation_yaml`, and
``tests/test_curation_yaml.py`` asserts nothing in the package bypasses it,
so the next registry added gets the guard by default rather than by
remembering.

Deliberately a leaf module — it imports only PyYAML — so the lightweight
registry and download modules can use it without pulling in the curation
stack, and :mod:`hitlist.conditions` no longer needs a lazy import to dodge
its cycle with :mod:`hitlist.curation`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from importlib.abc import Traversable


class UniqueKeyLoader(yaml.SafeLoader):
    """``SafeLoader`` that rejects a mapping key declared twice."""

    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}; PyYAML would keep only the last "
                    f"value and discard the first with no error",
                    key_node.start_mark,
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def load_curation_yaml(source: str | os.PathLike[str] | Traversable) -> Any:
    """Parse the YAML document at ``source``, rejecting duplicate mapping keys.

    ``source`` is a filesystem path or an ``importlib.resources`` traversable
    (anything with ``read_text``). Returns whatever the document holds —
    ``None`` for an empty file, exactly as ``yaml.safe_load`` would — so
    callers keep their ``or []`` / ``or {}`` defaults.
    """
    text = source.read_text() if hasattr(source, "read_text") else Path(source).read_text()
    return yaml.load(text, Loader=UniqueKeyLoader)
