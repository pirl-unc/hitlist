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

"""hitlist: curated and harmonized MHC ligand mass spectrometry data for pMHC target selection and model training.

Side effects at import time
---------------------------
Importing ``hitlist`` (or any submodule) does one thing:

- Sets ``pandas.options.future.infer_string = True``.  This is the pandas 3.0
   default, available as a future flag in 2.1+.  Cuts string-column memory
   ~5x at every layer of the build / load pipeline by switching pandas from
   ``object`` dtype (Python ``str`` references, ~50-100 bytes/cell) to
   pyarrow-backed ``StringDtype`` (~10 bytes/cell, identical layout to parquet).

   Downstream consumers that import hitlist inherit the behavior — this is
   forward-compatible (pandas 3.x will do this by default).  Code that depends
   on the legacy ``object`` representation can opt back out::

       import hitlist
       import pandas as pd
       pd.options.future.infer_string = False

   The flag is wrapped in :func:`contextlib.suppress(AttributeError)` so a
   future pandas release that removes the option (after promoting it to the
   permanent default) won't break import.

It never touches the filesystem: no data directory is created or cleaned.
Releases before the #579 fix ``rmtree``-d ``<data_dir>/index`` (a cache
retired in 1.30.41) on every import, following ``HITLIST_DATA_DIR``.
A leftover legacy ``index/`` directory is inert and safe to delete by hand.
"""

import contextlib as _contextlib

import pandas as _pd

with _contextlib.suppress(AttributeError):
    _pd.options.future.infer_string = True

from .version import __version__  # after the pandas flag, like every submodule

# ── Curated public API ───────────────────────────────────────────────────────
#
# Resolved lazily (PEP 562) so ``import hitlist`` stays fast and free of import
# cycles — the heavy submodules (builder/export/proteome pull in pyensembl etc.)
# are only imported on first attribute access. Listed in ``__all__`` so the API
# is discoverable via ``dir(hitlist)`` / autocomplete and stable across refactors.
_PUBLIC_API: dict[str, str] = {
    "write_cta_evidence_bundle": ".evidence_bundle",
    "write_tissue_blacklist_bundle": ".evidence_bundle",
    "verify_evidence_bundle": ".evidence_bundle",
    "build_tissue_blacklist": ".tissue_blacklist",
    "load_contributors": ".provenance",
    "load_lineage": ".lineage",
    "write_training_bundle": ".training_bundle",
    "verify_training_bundle": ".training_bundle",
    "audit_training_bundles": ".training_bundle",
    "audit_splits": ".split_audit",
    # Public enumerations shared by scanners, APIs, and CLIs.
    "MHC_ALLELE_PROVENANCE_VALUES": ".curation",
    "normalize_serotype_query": ".curation",
    # Build the cached indexes, or ask whether they are current without
    # building, printing, or downloading anything (#448).
    "build_observations": ".builder",
    "observations_cache_is_current": ".observations",
    "mappings_cache_is_current": ".mappings",
    # Load the built indexes (filters documented on each function).
    "load_ms_observations": ".observations",
    "load_observations": ".observations",
    "load_binding": ".observations",
    "load_other_assays": ".observations",
    "assay_annotations": ".assays",
    "load_all_evidence": ".observations",
    # Curated MHC typing. `sample_alleles_for_pmid` is the sample-level
    # answer, available for every study; the peptide-level functions below
    # need a `peptide_attributions` CSV and so are empty for most studies.
    # The `*_for_pmid` group returns whole read-only maps; the `attribute_*`
    # pair answers for one peptide without materializing them.
    "sample_alleles_for_pmid": ".curation",
    "peptide_alleles_for_pmid": ".curation",
    "peptide_typings_for_pmid": ".curation",
    "attribute_peptide_to_sample_alleles": ".curation",
    "attribute_peptide_to_per_sample_typings": ".curation",
    # Generate derived tables.
    "generate_ms_observations_table": ".export",
    "generate_training_table": ".export",
    # Proteome enumeration + gene sets.
    "ProteomeIndex": ".proteome",
    "DetectabilitySearchSpace": ".detectability",
    "build_detectability_training_set": ".detectability",
    "iter_detectability_training_set": ".detectability",
    "export_detectability_training_set": ".detectability",
    "load_gene_set": ".genes",
    "list_gene_sets": ".genes",
    # Dataset registry (download / register / resolve).
    "fetch": ".downloads",
    "refresh": ".downloads",
    "register": ".downloads",
    "remove": ".downloads",
    "get_path": ".downloads",
    "info": ".downloads",
    "list_datasets": ".downloads",
    "available_datasets": ".downloads",
    "download_to_file": ".downloads",
    "VersionedDatasetRegistry": ".downloads",
}

__all__ = ["__version__", *sorted(_PUBLIC_API)]


def __getattr__(name: str):
    module = _PUBLIC_API.get(name)
    if module is None:
        raise AttributeError(f"module 'hitlist' has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module, __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)
