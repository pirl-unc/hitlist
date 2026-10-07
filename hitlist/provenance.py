"""Lossless source contributors without adding observation weight (#622).

IDs are readable row locators scoped to the source snapshots in build metadata.
They are not biological identities. The temporary graph lives on disk so capture
does not require another corpus-sized Python collection.
"""

from __future__ import annotations

import errno
import json
import math
import os
import shutil
import sqlite3
import tempfile
import time
import zlib
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


class ProvenanceStorageError(RuntimeError):
    """The disposable contributor graph or output filesystem ran out of capacity."""


def _env_bytes(name, default):
    value = float(os.environ.get(name, default))
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite, nonnegative number of GiB")
    return int(value * 1024**3)


class ContributorCollector:
    """Lossless, build-local graph with a capped disposable scratch database.

    Scratch is not a restart checkpoint: a failed build discards it. All graph
    work sets live in the capped database, never a corpus-wide SQLite sort.
    """

    _ROOT_BATCH_SIZE = 256
    _OUTPUT_BATCH_BYTES = 4 * 1024**2
    _OUTPUT_BATCH_ROWS = 10000

    def __init__(self, *, scratch_dir=None, max_scratch_bytes=None, min_free_bytes=None):
        self.scratch_dir = Path(
            scratch_dir or os.environ.get("HITLIST_PROVENANCE_SCRATCH_DIR") or tempfile.gettempdir()
        )
        self.max_scratch_bytes = (
            _env_bytes("HITLIST_PROVENANCE_MAX_GB", "16")
            if max_scratch_bytes is None
            else max_scratch_bytes
        )
        self.min_free_bytes = (
            _env_bytes("HITLIST_PROVENANCE_MIN_FREE_GB", "1")
            if min_free_bytes is None
            else min_free_bytes
        )
        if self.max_scratch_bytes < 4096 or self.min_free_bytes < 0:
            raise ValueError(
                "Scratch budget must be at least 4096 bytes; free-space reserve must be nonnegative"
            )
        self._next_space_check = 0
        self._capacity_error = None
        self._published = False
        self._output_directory = None
        self._temporary = None
        self.db = None
        self.sources = {}
        self._scan_count = 0

    def _storage_error(self, reason):
        return ProvenanceStorageError(
            f"Contributor build stopped: {reason}. Scratch directory: {self.scratch_dir}; "
            f"database limit: {self.max_scratch_bytes / 1024**3:.3f} GiB; "
            f"free-space reserve: {self.min_free_bytes / 1024**3:.3f} GiB. "
            "Free space or choose HITLIST_PROVENANCE_SCRATCH_DIR (default: TMPDIR); "
            "increase HITLIST_PROVENANCE_MAX_GB only on a filesystem with capacity. "
            "The required space depends on the source rows and contributor links. "
            "Existing published contributor data is preserved; owned scratch and partial "
            "output are removed. Retry the build after correcting capacity."
        )

    def _check_space(self, *, force=False):
        now = time.monotonic()
        if not force and now < self._next_space_check:
            return
        self._next_space_check = now + 1
        for directory in {self.scratch_dir, self._output_directory} - {None}:
            free = shutil.disk_usage(directory).free
            if free <= self.min_free_bytes:
                raise self._storage_error(
                    f"{directory} has only {free / 1024**3:.3f} GiB available"
                )

    def _progress(self):
        # Long SQL statements must also notice other processes filling the disk.
        try:
            self._check_space()
        except ProvenanceStorageError as error:
            self._capacity_error = error
            return 1
        return 0

    def _execute(self, sql, parameters=()):
        self._check_space()
        try:
            return self.db.execute(sql, parameters)
        except sqlite3.Error as error:
            translated = self._translate_error(error)
            if translated is not None:
                raise translated from error
            raise

    def _close(self):
        if self.db is not None:
            self.db.close()
        if self._temporary is not None:
            self._temporary.cleanup()

    def _translate_error(self, error):
        # The enclosing builder also writes other artifacts after write().
        # Their failures must not claim the old contributor file is intact.
        if self._published:
            return None
        if self._capacity_error is not None:
            return self._capacity_error
        # Python 3.9/3.10 do not expose sqlite_errorcode on exceptions.
        if isinstance(error, sqlite3.Error) and (
            getattr(error, "sqlite_errorcode", None) == 13
            or str(error) == "database or disk is full"
        ):
            return self._storage_error("SQLite reached the scratch limit or the filesystem is full")
        if isinstance(error, OSError) and error.errno in {errno.ENOSPC, errno.EDQUOT}:
            return self._storage_error(
                f"filesystem capacity exhausted at {error.filename or self._output_directory or self.scratch_dir}"
            )
        return None

    def __enter__(self):
        try:
            self.scratch_dir.mkdir(parents=True, exist_ok=True)
            self._check_space(force=True)
            available = shutil.disk_usage(self.scratch_dir).free - self.min_free_bytes
            self.max_scratch_bytes = min(self.max_scratch_bytes, available) // 4096 * 4096
            if self.max_scratch_bytes < 4096:
                raise self._storage_error("insufficient space for the scratch database")
            self._temporary = tempfile.TemporaryDirectory(
                prefix="hitlist-contributors-", dir=self.scratch_dir
            )
            self.db = sqlite3.connect(str(Path(self._temporary.name) / "records.sqlite"))
            # Only disposable scratch uses OFF: no rollback/WAL files can grow
            # outside max_page_count. Never use these settings on published data.
            self.db.executescript(f"""
                PRAGMA page_size=4096;
                PRAGMA journal_mode=OFF;
                PRAGMA synchronous=OFF;
                PRAGMA mmap_size=0;
                PRAGMA cache_size=-8192;
                PRAGMA automatic_index=OFF;
                PRAGMA max_page_count={self.max_scratch_bytes // 4096};
                CREATE TABLE records (id TEXT PRIMARY KEY, dataset TEXT, row_number INTEGER,
                                      payload BLOB, pmid TEXT);
                CREATE TABLE edges (source TEXT, target TEXT, relation INTEGER, label TEXT);
                CREATE INDEX edge_target ON edges(target);
                CREATE TABLE retained (id TEXT PRIMARY KEY) WITHOUT ROWID;
                CREATE TABLE assays (scan_id INTEGER, iri TEXT, record_id TEXT,
                                     PRIMARY KEY (scan_id, iri)) WITHOUT ROWID;
            """)
            for table in ("ancestry", "frontier", "next_frontier"):
                self.db.execute(f"""
                    CREATE TABLE {table} (root TEXT, node TEXT, label TEXT, relations INTEGER,
                                         PRIMARY KEY (root, node, label, relations)) WITHOUT ROWID
                """)
            self.db.set_progress_handler(self._progress, 100000)
            return self
        except BaseException as error:
            self._close()
            translated = self._translate_error(error)
            if translated is not None:
                raise translated from error
            raise

    def __exit__(self, exc_type, error, traceback):
        self._close()

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
        # Compress the original JSON strings together: output stays byte-for-byte
        # compatible, including field order, Unicode escaping and unmapped cells.
        payload = zlib.compress(
            (json.dumps(fields, sort_keys=True) + "\n" + json.dumps(row_values)).encode(), level=1
        )
        self._execute(
            "INSERT INTO records VALUES (?, ?, ?, ?, ?)",
            (
                record_id,
                dataset,
                int(row_number),
                payload,
                str(fields.get("pmid", fields.get("manifest", {}).get("pmid", ""))),
            ),
        )
        return record_id

    def observe(self, record_id: str, sample_label="") -> str:
        node = f"observation:{record_id}"
        if sample_label:
            node += f"|sample:{quote(sample_label, safe='')}"
        self._execute("INSERT INTO edges VALUES (?, ?, 0, ?)", (record_id, node, sample_label))
        return node

    def begin_scan(self):
        self._scan_count += 1
        return self._scan_count

    def duplicate_assay(self, scan_id, iri, record_id):
        inserted = self._execute(
            "INSERT OR IGNORE INTO assays VALUES (?, ?, ?)", (scan_id, iri, record_id)
        ).rowcount
        if inserted:
            return False
        first = self._execute(
            "SELECT record_id FROM assays WHERE scan_id=? AND iri=?", (scan_id, iri)
        ).fetchone()[0]
        self.redirect(record_id, first, "assay_copy")
        return True

    def redirect(self, source: str, target: str, relationship: str):
        if source == target:
            raise ValueError("A contributor cannot redirect to itself")
        self._execute(
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

    def _iter_links(self):
        """Traverse narrow identities in capped tables before reading payloads.

        UNION semantics are enforced by work-table primary keys, including the
        label and accumulated flags. This preserves multiple paths and cycles.
        Separate frontier tables avoid a self-insert's implicit materialization.
        """
        last_root = ""
        while roots := self._execute(
            "SELECT id FROM retained WHERE id > ? ORDER BY id LIMIT ?",
            (last_root, self._ROOT_BATCH_SIZE),
        ).fetchall():
            last_root = roots[-1][0]
            self.db.executemany(
                "INSERT INTO frontier VALUES (?, ?, '', 0)", ((root, root) for (root,) in roots)
            )
            frontier, next_frontier = "frontier", "next_frontier"
            while True:
                self._execute(f"INSERT INTO ancestry SELECT * FROM {frontier}")
                added = self._execute(f"""
                    INSERT OR IGNORE INTO {next_frontier}
                    SELECT f.root, e.source,
                           CASE WHEN e.label != '' THEN e.label ELSE f.label END,
                           f.relations | e.relation
                    FROM {frontier} f CROSS JOIN edges e ON e.target = f.node
                    WHERE NOT EXISTS (
                        SELECT 1 FROM ancestry a
                        WHERE a.root = f.root AND a.node = e.source
                          AND a.label = CASE WHEN e.label != '' THEN e.label ELSE f.label END
                          AND a.relations = (f.relations | e.relation)
                    )
                """).rowcount
                self._execute(f"DELETE FROM {frontier}")
                if not added:
                    break
                frontier, next_frontier = next_frontier, frontier
            # CROSS JOIN keeps the ordered narrow ancestry index outermost.
            # Ordering by a.node (equal to r.id) avoids sorting full payloads.
            yield from self._execute("""
                SELECT a.root, r.id, r.dataset, r.row_number, r.pmid, r.payload,
                       a.label, a.relations
                FROM ancestry a CROSS JOIN records r ON r.id = a.node
                ORDER BY a.root, a.node, a.label, a.relations
            """)
            self._execute("DELETE FROM ancestry")

    def write(self, frames, path: Path) -> dict:
        """Flatten retained contributors with capped graph work and byte-sized output batches."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        schema = pa.schema(
            [(c, pa.int64() if c == "source_row" else pa.string()) for c in CONTRIBUTOR_COLUMNS]
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        self._output_directory = path.parent
        self._check_space(force=True)
        temporary = path.with_suffix(".parquet.partial")
        n_links = 0
        try:
            for frame in frames:
                if "provenance_id" in frame:
                    self.db.executemany(
                        "INSERT INTO retained VALUES (?)",
                        ((value,) for value in frame["provenance_id"]),
                    )
            self.db.commit()
            with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
                records = []
                batch_bytes = 0
                for node, source, dataset, row, pmid, payload, label, flags in self._iter_links():
                    decoded = zlib.decompress(payload)
                    fields, values = decoded.decode().split("\n", 1)
                    batch_bytes += len(decoded) + len(node) + len(source) + len(label)
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
                    if (
                        batch_bytes >= self._OUTPUT_BATCH_BYTES
                        or len(records) >= self._OUTPUT_BATCH_ROWS
                    ):
                        self._check_space(force=True)
                        writer.write_table(pa.Table.from_pylist(records, schema=schema))
                        n_links += len(records)
                        records.clear()
                        batch_bytes = 0
                if records:
                    self._check_space(force=True)
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
            self._published = True
        except (sqlite3.Error, OSError) as error:
            translated = self._translate_error(error)
            if translated is not None:
                raise translated from error
            raise
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
    from .builder import _binding_path, _cache_meta, _observations_path, _other_assays_path

    metadata = _cache_meta()
    contract = metadata.get("provenance")
    if not contract:
        return None
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported contributor schema; rebuild observations")
    if metadata.get("artifact_version", 0) >= 9 and "other_assays" not in contract["indexes"]:
        raise ValueError("Provenance contract missing other assays; rebuild observations")
    artifacts = [
        ("contributors", contributors_path(), contract["contributors"]),
        ("observations", _observations_path(), contract["indexes"]["observations"]),
        ("binding", _binding_path(), contract["indexes"]["binding"]),
    ]
    if "other_assays" in contract["indexes"]:
        artifacts.append(
            ("other_assays", _other_assays_path(), contract["indexes"]["other_assays"])
        )
    for name, path, expected in artifacts:
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
