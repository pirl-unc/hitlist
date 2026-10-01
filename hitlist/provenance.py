"""Lossless source contributors without adding observation weight (#622).

IDs are readable row locators scoped to the source snapshots in build metadata.
They are not biological identities. The temporary graph lives on disk so capture
does not require another corpus-sized Python collection.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from urllib.parse import quote

import pandas as pd

SCHEMA_VERSION = 1
RELATIONS = {
    "assay_copy": 1,
    "database_copy": 2,
    "within_file_overlap": 4,
    "supplementary_overlap": 8,
    "reference_overlap": 16,
}
CONTRIBUTOR_COLUMNS = [
    "provenance_id",
    "source_record_id",
    "source_dataset",
    "source_row",
    "original_pmid",
    "original_fields",
    "source_row_values",
    "attributed_sample_label",
    "attribution_evidence",
    "relationships",
    "relationship_status",
]


def file_digest(path: Path) -> dict:
    """Content fingerprint, streaming even for multi-gigabyte source CSVs."""
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "size_bytes": Path(path).stat().st_size}


def contributors_path() -> Path:
    from .downloads import data_dir

    return data_dir() / "observation_contributors.parquet"


class ContributorCollector:
    """Build-local directed retention graph; all writes use one transaction."""

    def __enter__(self):
        self._temporary = tempfile.TemporaryDirectory(prefix="hitlist-contributors-")
        self.db = sqlite3.connect(str(Path(self._temporary.name) / "records.sqlite"))
        self.db.executescript("""
            PRAGMA temp_store=FILE;
            CREATE TABLE records (id TEXT PRIMARY KEY, dataset TEXT, row_number INTEGER,
                                  fields TEXT, row_values TEXT, pmid TEXT);
            CREATE TABLE edges (source TEXT, target TEXT, relation INTEGER, label TEXT);
            CREATE INDEX edge_target ON edges(target);
            CREATE TABLE retained (id TEXT PRIMARY KEY);
            CREATE TABLE assays (scan_id INTEGER, iri TEXT, record_id TEXT,
                                 PRIMARY KEY (scan_id, iri));
        """)
        self.sources = {}
        self._scan_count = 0
        return self

    def __exit__(self, *exc):
        self.db.close()
        self._temporary.cleanup()

    def register_source(self, dataset: str, path: Path, *, description=""):
        snapshot = {
            "path": str(path),
            **file_digest(path),
            "upstream_description": description,
            "locator_basis": "ingested_csv_logical_data_row_1_based",
            "upstream_record_locator_status": (
                "assay_iri_when_reported" if dataset in {"iedb", "cedar"} else "not_available"
            ),
        }
        if dataset in self.sources and self.sources[dataset] != snapshot:
            raise ValueError(f"Conflicting source snapshots for {dataset}")
        self.sources[dataset] = snapshot

    def record(self, dataset, row_number, fields, row_values) -> str:
        record_id = f"record:{quote(dataset, safe='')}:row:{row_number}"
        self.db.execute(
            "INSERT INTO records VALUES (?, ?, ?, ?, ?, ?)",
            (
                record_id,
                dataset,
                int(row_number),
                json.dumps(fields, sort_keys=True),
                json.dumps(row_values),
                str(fields.get("pmid", fields.get("manifest", {}).get("pmid", ""))),
            ),
        )
        return record_id

    def observe(self, record_id: str, sample_label="") -> str:
        node = f"observation:{record_id}"
        if sample_label:
            node += f"|sample:{quote(sample_label, safe='')}"
        self.db.execute("INSERT INTO edges VALUES (?, ?, 0, ?)", (record_id, node, sample_label))
        return node

    def begin_scan(self):
        self._scan_count += 1
        return self._scan_count

    def duplicate_assay(self, scan_id, iri, record_id):
        inserted = self.db.execute(
            "INSERT OR IGNORE INTO assays VALUES (?, ?, ?)", (scan_id, iri, record_id)
        ).rowcount
        if inserted:
            return False
        first = self.db.execute(
            "SELECT record_id FROM assays WHERE scan_id=? AND iri=?", (scan_id, iri)
        ).fetchone()[0]
        self.redirect(record_id, first, "assay_copy")
        return True

    def redirect(self, source: str, target: str, relationship: str):
        if source == target:
            raise ValueError("A contributor cannot redirect to itself")
        self.db.execute(
            "INSERT INTO edges VALUES (?, ?, ?, '')", (source, target, RELATIONS[relationship])
        )

    def retain_duplicates(self, frame, keys, relationship, *, uncertain_keys=()):
        """Record redirects before a keep-first deduplication of ``frame``."""
        if "provenance_id" not in frame:
            return
        work = frame[["provenance_id"]].copy()
        work["key"] = keys.to_numpy()
        winners = work.drop_duplicates("key").set_index("key")["provenance_id"]
        for row in work[work["key"].duplicated()].itertuples(index=False):
            self.redirect(
                row.provenance_id,
                winners.at[row.key],
                "reference_overlap" if row.key in uncertain_keys else relationship,
            )

    def write(self, frames, path: Path) -> dict:
        """Flatten only contributors of retained observations, in bounded batches."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        for frame in frames:
            if "provenance_id" in frame:
                self.db.executemany(
                    "INSERT INTO retained VALUES (?)",
                    ((value,) for value in frame["provenance_id"]),
                )
        self.db.commit()
        cursor = self.db.execute("""
            WITH RECURSIVE ancestry(id, node, relations, label) AS (
                SELECT id, id, 0, '' FROM retained
                UNION
                SELECT a.id, e.source, a.relations | e.relation,
                       CASE WHEN e.label != '' THEN e.label ELSE a.label END
                FROM ancestry a JOIN edges e ON e.target = a.node
            )
            SELECT DISTINCT a.id, r.id, r.dataset, r.row_number, r.pmid, r.fields,
                            r.row_values, a.label, a.relations
            FROM ancestry a JOIN records r ON r.id = a.node
            ORDER BY a.id, r.id, a.label, a.relations
        """)
        schema = pa.schema(
            [(c, pa.int64() if c == "source_row" else pa.string()) for c in CONTRIBUTOR_COLUMNS]
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".parquet.partial")
        n_links = 0
        try:
            with pq.ParquetWriter(temporary, schema) as writer:
                while batch := cursor.fetchmany(10000):
                    records = []
                    for node, source, dataset, row, pmid, fields, values, label, flags in batch:
                        records.append(
                            dict(
                                zip(
                                    CONTRIBUTOR_COLUMNS,
                                    (
                                        node,
                                        source,
                                        dataset,
                                        row,
                                        pmid,
                                        fields,
                                        values,
                                        label,
                                        "curated_peptide_map" if label else "not_attributed",
                                        json.dumps(
                                            [k for k, v in RELATIONS.items() if flags & v]
                                            or ["retained"]
                                        ),
                                        "overlap_unresolved" if flags & 28 else "source_record",
                                    ),
                                )
                            )
                        )
                    writer.write_table(pa.Table.from_pylist(records, schema=schema))
                    n_links += len(records)
            # Catch source replacement during a long build instead of binding
            # rows to the digest of bytes that were never actually scanned.
            for source in self.sources.values():
                if file_digest(Path(source["path"])) != {
                    k: source[k] for k in ("sha256", "size_bytes")
                }:
                    raise ValueError("Source changed during build; rebuild observations")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return {
            "schema_version": SCHEMA_VERSION,
            "sources": self.sources,
            "contributors": {"file": path.name, **file_digest(path)},
            "n_contributor_links": n_links,
        }


def _contributor_contract(*, verify_hashes=False):
    """Check captured artifacts using stats, or hashes for contributor reads."""
    from .builder import _binding_path, _cache_meta, _observations_path

    metadata = _cache_meta()
    contract = metadata.get("provenance")
    if not contract:
        return None
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported contributor schema; rebuild observations")
    for name, path, expected in [
        ("contributors", contributors_path(), contract["contributors"]),
        ("observations", _observations_path(), contract["indexes"]["observations"]),
        ("binding", _binding_path(), contract["indexes"]["binding"]),
    ]:
        matches = path.exists() and path.stat().st_size == expected["size_bytes"]
        if matches:
            stamp = metadata.get("parquets", {}).get(name, {})
            # Copied/downloaded artifacts can have new mtimes without changed
            # bytes. Verify that case instead of requiring an unnecessary rebuild.
            if verify_hashes or stamp.get("mtime") != path.stat().st_mtime:
                matches = file_digest(path)["sha256"] == expected["sha256"]
        if not matches:
            raise ValueError(f"Provenance artifact mismatch: {path.name}; rebuild observations")
    return contract


def load_contributors(provenance_ids=None) -> pd.DataFrame:
    """Load source contributors after verifying sidecar and index content hashes.

    Legacy indexes return an empty table; a claimed capture must be intact.
    """
    contract = _contributor_contract(verify_hashes=True)
    if not contract:
        if provenance_ids is not None and any(provenance_ids):
            raise ValueError("Provenance metadata missing; rebuild observations")
        return pd.DataFrame(columns=CONTRIBUTOR_COLUMNS)
    filters = None
    if provenance_ids is not None:
        ids = sorted(set(provenance_ids) - {""})
        if not ids:
            return pd.DataFrame(columns=CONTRIBUTOR_COLUMNS)
        filters = [("provenance_id", "in", ids)]
    return pd.read_parquet(contributors_path(), filters=filters)
