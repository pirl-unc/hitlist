"""Checked canine bulk proteomics with original tables and bounded offline queries.

Bulk detection, protein inference and DLA presentation are separate evidence.
No input to this module supplies independent presentation-positive dog counts.
"""

from __future__ import annotations

import copy
import csv
import io
import json
import math
import re
import shutil
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path
from zipfile import ZipFile

import datacache
import pandas as pd
from filelock import FileLock

from .curation_yaml import load_curation_yaml
from .managed_assets import cache_bytes, check_cache_path
from .provenance import file_digest

_CATALOG = Path(__file__).parent / "data" / "canine_bulk.yaml"
_LIMITS = {
    "max_input_bytes": 256 * 2**20,
    "max_expanded_bytes": 512 * 2**20,
    "max_records": 5_000_000,
    "max_output_bytes": 2 * 2**30,
    "max_protein_residues": 1_000_000,
}
_TABLE_KEYS = {
    "sample": "sample_id",
    "atlas_search_batch": "atlas_search_batch_id",
    "peptide": "peptide_id",
    "peptide_instance": "peptide_instance_id",
    "peptide_instance_search_batch": "peptide_instance_search_batch_id",
    "biosequence": "biosequence_id",
    "peptide_mapping": "peptide_mapping_id",
    "protein_identification": "protein_identification_id",
    "modified_peptide_instance": "modified_peptide_instance_id",
    "modified_peptide_instance_search_batch": "modified_peptide_instance_search_batch_id",
}
_METADATA_TABLES = (
    "atlas_build",
    "atlas_build_sample",
    "biosequence_set",
    "instrument",
    "proteases",
)
_LINKS = (
    ("atlas_search_batch", "sample_id", "sample"),
    ("peptide_instance", "peptide_id", "peptide"),
    ("peptide_instance_search_batch", "peptide_instance_id", "peptide_instance"),
    ("peptide_instance_search_batch", "atlas_search_batch_id", "atlas_search_batch"),
    ("peptide_mapping", "peptide_instance_id", "peptide_instance"),
    ("peptide_mapping", "matched_biosequence_id", "biosequence"),
    ("protein_identification", "biosequence_id", "biosequence"),
    ("modified_peptide_instance", "peptide_instance_id", "peptide_instance"),
    (
        "modified_peptide_instance_search_batch",
        "modified_peptide_instance_id",
        "modified_peptide_instance",
    ),
    ("modified_peptide_instance_search_batch", "atlas_search_batch_id", "atlas_search_batch"),
)
_SAMPLE_COLUMNS = (
    "sample_id",
    "source_sample_id",
    "source_tissue",
    "tissue_group",
    "donor_id",
    "donor_status",
    "developmental_stage",
    "health_context",
    "tissue_status",
    "is_cell_line",
)
_COMMON = """c.source_id, c.source_taxon, 'bulk_proteomics' AS source,
    'bulk' AS evidence_kind, 0 AS is_ms_observation,
    'bulk_proteomics' AS assay_modality, 'LC-MS/MS (bulk proteomics)' AS assay_method,
    '' AS mhc_restriction, c.digestion_enzyme, c.acquisition_mode, c.instrument"""
_SAMPLE_SELECT = ", ".join("s." + name for name in _SAMPLE_COLUMNS)


def _positive(name, value):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def canine_bulk_catalog():
    """Return versioned source scope and byte-pinned assets without acquisition."""
    result = load_curation_yaml(_CATALOG)
    if result.get("schema_version") != 1:
        raise ValueError("Unsupported canine bulk catalog")
    for source_id, source in result["sources"].items():
        if any(
            not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", v)
            for v in (source_id, source["version"])
        ):
            raise ValueError("Invalid source/version path component")
        for asset in source["assets"].values():
            if (
                Path(asset["filename"]).name != asset["filename"]
                or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", asset["filename"])
                or not re.fullmatch(r"[0-9a-f]{64}", asset["sha256"])
                or not asset["url"].startswith("https://")
            ):
                raise ValueError("Invalid pinned canine bulk asset")
            _positive("size_bytes", asset["size_bytes"])
    return result


def _source(source_id):
    sources = canine_bulk_catalog()["sources"]
    if source_id not in sources:
        raise ValueError(f"Unknown canine bulk source {source_id!r}; available: {sorted(sources)}")
    return {"source_id": source_id, **copy.deepcopy(sources[source_id])}


def fetch_canine_bulk_asset(
    source_id, asset, *, max_cache_bytes=2**30, max_asset_bytes=256 * 2**20
):
    """Explicit bounded acquisition; verified cache hits, no implicit eviction."""
    from .downloads import data_dir

    _positive("max_cache_bytes", max_cache_bytes)
    _positive("max_asset_bytes", max_asset_bytes)
    source = _source(source_id)
    spec = source["assets"][asset]
    if spec["size_bytes"] > max_asset_bytes:
        raise ValueError("Asset exceeds max_asset_bytes")
    root = data_dir() / "canine_bulk"
    path = root / source_id / source["version"] / spec["filename"]
    check_cache_path(root, path)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".assets.lock"
    check_cache_path(root, lock)
    with FileLock(str(lock)):
        check_cache_path(root, path)
        if path.exists():
            _check_asset(path, spec)
            return path
        if cache_bytes(root) + spec["size_bytes"] + 2 * 2**20 > max_cache_bytes:
            raise ValueError("Managed canine assets exceed max_cache_bytes; no files were evicted")

        def progress(done, total):
            if done > spec["size_bytes"]:
                raise ValueError("Transfer exceeds pinned size")

        datacache.fetch_file(
            spec["url"],
            destination=path,
            expected_sha256=spec["sha256"],
            expected_size=spec["size_bytes"],
            raw=True,
            resume=True,
            record_provenance=True,
            progress_callback=progress,
            chunk_size=2**20,
            timeout=60,
        )
    return path


def _check_asset(path, spec):
    expected = {name: spec[name] for name in ("sha256", "size_bytes")}
    if not Path(path).is_file() or file_digest(Path(path)) != expected:
        raise ValueError(f"Source asset checksum/size mismatch: {path}")


def _quote(name):
    return '"' + name.replace('"', '""') + '"'


def _import_table(db, stream, name, limits, key=None):
    reader = csv.DictReader(stream, delimiter="\t")
    columns = reader.fieldnames
    if not columns or len(set(columns)) != len(columns) or "_source_row" in columns:
        raise ValueError(f"Invalid columns for {name}")
    if key is not None and key not in columns:
        raise ValueError(f"{name} lacks {key}")
    db.execute(
        f"CREATE TABLE {_quote(name)} (_source_row INTEGER PRIMARY KEY, "
        + ", ".join(f"{_quote(c)} TEXT" for c in columns)
        + ")"
    )
    if key:
        db.execute(f"CREATE UNIQUE INDEX {_quote(name + '_key')} ON {_quote(name)} ({_quote(key)})")
    insert = f"INSERT INTO {_quote(name)} VALUES ({','.join('?' for _ in range(len(columns) + 1))})"
    batch = []
    for source_row, row in enumerate(reader, 1):
        if source_row > limits["max_records"]:
            raise ValueError(f"{name} exceeds max_records")
        if None in row or any(v is None for v in row.values()) or (key and row[key] in ("", "\\N")):
            raise ValueError(f"Malformed {name} source row {source_row}")
        batch.append((source_row, *(None if row[c] == "\\N" else row[c] for c in columns)))
        if len(batch) == 2000:
            db.executemany(insert, batch)
            batch.clear()
    db.executemany(insert, batch)


def _validate_links(db):
    for child, column, parent in _LINKS:
        pk = _TABLE_KEYS[parent]
        query = (
            f'SELECT 1 FROM "raw_{child}" c LEFT JOIN "raw_{parent}" p '
            f'ON c."{column}"=p."{pk}" WHERE p."{pk}" IS NULL LIMIT 1'
        )
        if db.execute(query).fetchone():
            raise ValueError(f"Unresolved source relationship: {child}.{column} -> {parent}")
    for table in ("peptide_instance_search_batch", "modified_peptide_instance_search_batch"):
        for (value,) in db.execute(f'SELECT DISTINCT n_observations FROM "raw_{table}"'):
            if value not in (None, "") and not value.isdigit():
                raise ValueError("Reported observation counts must be nonnegative integers")


def _samples(db, source):
    db.execute(
        "CREATE TABLE samples ("
        + ", ".join(_quote(n) + " TEXT" for n in _SAMPLE_COLUMNS)
        + ", PRIMARY KEY (sample_id))"
    )
    atlas = source["adapter"] == "peptideatlas"
    rows = db.execute(
        "SELECT sample_id, sample_title FROM raw_sample"
        if atlas
        else "SELECT DISTINCT BioReplicate, Condition FROM raw_proteins"
    ).fetchall()
    for identity, tissue in rows:
        if not identity or not tissue:
            raise ValueError("Source samples need explicit sample and tissue identities")
        context_name = source.get("sample_context_by_id", {}).get(
            identity, source["default_sample_context"]
        )
        context = source["sample_contexts"][context_name]
        values = (
            source["source_id"] + ":" + identity,
            identity,
            tissue,
            source["tissue_groups"].get(tissue.lower(), ""),
            *(context[name] for name in _SAMPLE_COLUMNS[4:]),
        )
        db.execute("INSERT INTO samples VALUES (" + ",".join("?" for _ in values) + ")", values)


def _atlas_views(db):
    joined = """FROM raw_peptide_instance_search_batch d
        JOIN raw_peptide_instance i USING (peptide_instance_id)
        JOIN raw_peptide p USING (peptide_id)
        JOIN raw_atlas_search_batch b USING (atlas_search_batch_id)
        JOIN samples s ON s.source_sample_id=b.sample_id
        LEFT JOIN mapping_summary g USING (peptide_instance_id) CROSS JOIN context c"""
    db.execute(f"""CREATE VIEW peptides AS SELECT {_COMMON}, {_SAMPLE_SELECT},
        c.source_id || ':peptide_instance_search_batch:' || d.peptide_instance_search_batch_id AS source_record_id,
        p.peptide_sequence AS peptide, i.peptide_instance_id,
        d.atlas_search_batch_id AS acquisition_group_id, CAST(b.n_runs AS INTEGER) AS n_runs,
        CAST(NULLIF(d.n_observations,'') AS INTEGER) AS n_reported_observations,
        g.n_target_mappings, g.n_decoy_mappings, g.n_contaminant_mappings,
        CAST(d.n_observations AS INTEGER)>0 AND g.n_target_mappings>0 AS is_bulk_detection,
        i.best_probability AS peptide_best_probability,
        'tryptic peptide detection' AS response_measured,
        CASE WHEN CAST(d.n_observations AS INTEGER)>0 AND g.n_target_mappings>0 THEN 'Positive' ELSE '' END AS qualitative_measurement,
        CASE WHEN CAST(d.n_observations AS INTEGER)>0 AND g.n_target_mappings>0 THEN 'detected'
            WHEN g.n_decoy_mappings>0 THEN 'decoy_only'
            WHEN g.n_contaminant_mappings>0 THEN 'contaminant_only'
            ELSE 'unestablished' END AS detection_status
        {joined}""")
    db.execute(f"""CREATE VIEW mappings AS SELECT {_COMMON}, m.peptide_mapping_id,
        m.peptide_instance_id, p.peptide_sequence AS peptide, b.biosequence_name AS protein_id,
        m.matched_biosequence_id, m.start_in_biosequence, m.end_in_biosequence,
        CASE WHEN b.biosequence_name GLOB 'DECOY_*' THEN 'decoy'
            WHEN b.biosequence_name GLOB 'CONTAM_*' THEN 'contaminant'
            ELSE 'source_target' END AS protein_origin,
        CASE WHEN substr(b.biosequence_seq, CAST(m.start_in_biosequence AS INTEGER),
            CAST(m.end_in_biosequence AS INTEGER)-CAST(m.start_in_biosequence AS INTEGER)+1)=p.peptide_sequence
            THEN 'exact' ELSE 'reported_nonexact' END AS match_kind
        FROM raw_peptide_mapping m JOIN raw_biosequence b ON b.biosequence_id=m.matched_biosequence_id
        JOIN raw_peptide_instance i USING (peptide_instance_id) JOIN raw_peptide p USING (peptide_id)
        CROSS JOIN context c""")
    db.execute("""CREATE VIEW mapping_summary AS SELECT peptide_instance_id,
        SUM(protein_origin='source_target') AS n_target_mappings,
        SUM(protein_origin='decoy') AS n_decoy_mappings,
        SUM(protein_origin='contaminant') AS n_contaminant_mappings
        FROM mappings GROUP BY peptide_instance_id""")
    db.execute(f"""CREATE VIEW modifications AS SELECT {_COMMON}, {_SAMPLE_SELECT},
        c.source_id || ':modified_peptide_instance_search_batch:' || d.modified_peptide_instance_search_batch_id AS source_record_id,
        m.peptide_instance_id, p.peptide_sequence AS peptide,
        m.modified_peptide_sequence AS modified_peptide, m.peptide_charge,
        m.best_probability AS modified_peptide_best_probability,
        d.atlas_search_batch_id AS acquisition_group_id,
        CAST(NULLIF(d.n_observations,'') AS INTEGER) AS n_reported_observations
        FROM raw_modified_peptide_instance_search_batch d
        JOIN raw_modified_peptide_instance m USING (modified_peptide_instance_id)
        JOIN raw_peptide_instance i USING (peptide_instance_id) JOIN raw_peptide p USING (peptide_id)
        JOIN raw_atlas_search_batch b USING (atlas_search_batch_id)
        JOIN samples s ON s.source_sample_id=b.sample_id CROSS JOIN context c""")
    db.execute(f"""CREATE VIEW proteins AS SELECT {_COMMON},
        c.source_id || ':protein_identification:' || p.protein_identification_id AS source_record_id,
        b.biosequence_name AS protein_id, p.protein_group_number AS protein_group_id,
        CASE WHEN b.biosequence_name GLOB 'DECOY_*' THEN 'decoy'
            WHEN b.biosequence_name GLOB 'CONTAM_*' THEN 'contaminant'
            ELSE 'source_target' END AS protein_origin,
        'protein_inference' AS detection_status, '' AS sample_id, '' AS donor_id,
        'aggregate_unresolved' AS donor_status, p.probability AS protein_probability,
        p.n_observations AS n_reported_observations,
        'protein inference' AS response_measured, '' AS qualitative_measurement
        FROM raw_protein_identification p JOIN raw_biosequence b USING (biosequence_id)
        CROSS JOIN context c""")


_NONFINITE = ("nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity")


def _validate_abundances(db):
    for column in ("Abundance", "IBAQ", "rIBAQ", "sumIBAQ"):
        for (value,) in db.execute(f'SELECT DISTINCT "{column}" FROM raw_proteins'):
            token = value.strip(" \t\r\n\v\f").lower() if value is not None else ""
            if token not in ("", "na", *_NONFINITE):
                if not re.fullmatch(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:e[+-]?[0-9]+)?", token):
                    raise ValueError(f"Malformed numeric value in {column}: {value!r}")
                if not math.isfinite(float(token)):
                    raise ValueError(f"Expected a finite measurement in {column}: {value!r}")


def _pfizer_views(db):
    nonfinite = ",".join(repr(s) for s in _NONFINITE)

    def normalized(column):
        return f"lower(trim(p.{column}, ' ' || char(9) || char(10) || char(11) || char(12) || char(13)))"

    def number(column):
        return (
            f"CASE WHEN {normalized(column)} IN ('na','',{nonfinite}) "
            f"THEN NULL ELSE CAST(p.{column} AS REAL) END"
        )

    def status(column):
        return (
            f"CASE WHEN p.{column} IS NULL OR {normalized(column)} IN ('na','') THEN 'missing' "
            f"WHEN {normalized(column)} IN ({nonfinite}) "
            "THEN 'reported_nonfinite' ELSE 'reported_finite' END"
        )

    db.execute(f"""CREATE VIEW proteins AS SELECT {_COMMON}, {_SAMPLE_SELECT},
        c.source_id || ':protein_table:' || p._source_row AS source_record_id,
        p.Protein AS protein_id, p.ProteinGroup AS protein_group_id,
        'reported_representative_only' AS protein_group_resolution,
        p.Run AS acquisition_group_id, p.Mixture AS mixture, p.Channel AS channel,
        p.TechRepMixture AS technical_replicate,
        {number("Abundance")} AS reported_abundance,
        {status("Abundance")} AS reported_abundance_status,
        'source Abundance; normalization unresolved' AS reported_abundance_unit,
        {number("IBAQ")} AS ibaq, {status("IBAQ")} AS ibaq_status,
        {number("rIBAQ")} AS relative_ibaq, {status("rIBAQ")} AS relative_ibaq_status,
        {number("sumIBAQ")} AS summed_ibaq, {status("sumIBAQ")} AS summed_ibaq_status,
        'protein_abundance' AS detection_status, 'protein abundance' AS response_measured,
        '' AS qualitative_measurement
        FROM raw_proteins p JOIN samples s ON s.source_sample_id=p.BioReplicate CROSS JOIN context c""")
    db.execute(f"""CREATE VIEW peptides AS SELECT {_COMMON}, {_SAMPLE_SELECT},
        '' AS peptide, '' AS source_record_id, 'tryptic peptide detection' AS response_measured,
        '' AS qualitative_measurement FROM samples s CROSS JOIN context c WHERE 0""")


def _reference(db, path, limits):
    db.execute("CREATE TABLE reference_sequences (protein_id TEXT PRIMARY KEY, sequence TEXT)")
    name, sequence = None, []

    def flush():
        if name is not None:
            protein = "".join(sequence)
            if not protein:
                raise ValueError("Empty reference protein")
            db.execute("INSERT INTO reference_sequences VALUES (?,?)", (name, protein))

    n_records, n_residues = 0, 0
    with path.open() as stream:
        for line in stream:
            if line.startswith(">"):
                flush()
                token = line[1:].split()[0]
                name = token.split("|")[1] if token.startswith(("sp|", "tr|")) else token
                sequence, n_residues = [], 0
                n_records += 1
                if n_records > limits["max_records"]:
                    raise ValueError("Reference exceeds max_records")
            elif line.strip():
                if name is None:
                    raise ValueError("Reference sequence precedes header")
                part = line.strip()
                n_residues += len(part)
                if n_residues > limits["max_protein_residues"]:
                    raise ValueError("Reference exceeds max_protein_residues")
                sequence.append(part)
        flush()
    if not n_records:
        raise ValueError("Reference contains no proteins")


def curate_canine_bulk(
    source_id, output_dir, *, local_assets=None, allow_download=False, limits=None
):
    """Import a reviewed source without treating bulk-MS rows as DLA ligands."""
    from .version import __version__

    bounds = {**_LIMITS, **(limits or {})}
    if set(bounds) != set(_LIMITS):
        raise ValueError("Unknown bulk input/output limit")
    for name, value in bounds.items():
        _positive(name, value)
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(target)
    source = _source(source_id)
    paths = {}
    if set(local_assets or {}) - set(source["assets"]):
        raise ValueError("Unknown local source asset")
    for name, spec in source["assets"].items():
        if spec["size_bytes"] > bounds["max_input_bytes"]:
            raise ValueError("Source exceeds max_input_bytes")
        if name in (local_assets or {}):
            path = Path(local_assets[name])
        elif allow_download:
            path = fetch_canine_bulk_asset(
                source_id, name, max_asset_bytes=bounds["max_input_bytes"]
            )
        else:
            raise ValueError(f"Supply checked local_assets[{name!r}] or explicitly allow_download")
        _check_asset(path, spec)
        paths[name] = path
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    previous_field_limit = csv.field_size_limit()
    try:
        available = (
            bounds["max_output_bytes"] - source["assets"]["reference"]["size_bytes"] - 2 * 2**20
        )
        if available < 4096:
            raise ValueError("Bulk evidence exceeds max_output_bytes")
        shutil.copyfile(paths["reference"], staging / "reference.fasta")
        with closing(sqlite3.connect(staging / "bulk.sqlite")) as db, db:
            db.execute("PRAGMA page_size=4096")
            db.execute(f"PRAGMA max_page_count={available // 4096}")
            db.execute("PRAGMA cache_size=-8192")
            db.execute("PRAGMA temp_store=FILE")
            db.execute(
                "CREATE TABLE context (source_id TEXT, source_taxon INTEGER, digestion_enzyme TEXT, acquisition_mode TEXT, instrument TEXT)"
            )
            db.execute(
                "INSERT INTO context VALUES (?,?,?,?,?)",
                tuple(
                    source[k]
                    for k in (
                        "source_id",
                        "source_taxon",
                        "digestion_enzyme",
                        "acquisition_mode",
                        "instrument",
                    )
                ),
            )
            csv.field_size_limit(2**20)
            if source["adapter"] == "peptideatlas":
                with ZipFile(paths["table"]) as archive:
                    names = archive.namelist()
                    if len(set(names)) != len(names) or any(Path(n).name != n for n in names):
                        raise ValueError("Unsafe or duplicate archive member")
                    if sum(i.file_size for i in archive.infolist()) > bounds["max_expanded_bytes"]:
                        raise ValueError("Archive exceeds max_expanded_bytes")
                    for table, key in _TABLE_KEYS.items():
                        with archive.open(table + ".tsv") as raw:
                            _import_table(db, io.TextIOWrapper(raw), "raw_" + table, bounds, key)
                    for table in _METADATA_TABLES:
                        if table + ".tsv" in names:
                            with archive.open(table + ".tsv") as raw:
                                _import_table(db, io.TextIOWrapper(raw), "raw_" + table, bounds)
                _validate_links(db)
                _samples(db, source)
                _atlas_views(db)
            else:
                with paths["table"].open() as stream:
                    _import_table(db, stream, "raw_proteins", bounds)
                _samples(db, source)
                _validate_abundances(db)
                _pfizer_views(db)
            _reference(db, staging / "reference.fasta", bounds)
            if (
                source["adapter"] == "peptideatlas"
                and db.execute(
                    "SELECT 1 FROM raw_biosequence b LEFT JOIN reference_sequences r "
                    "ON b.biosequence_name=r.protein_id "
                    "WHERE r.protein_id IS NULL OR b.biosequence_seq!=r.sequence LIMIT 1"
                ).fetchone()
            ):
                raise ValueError(
                    "Reported protein identity/sequence disagrees with reference FASTA"
                )
            tables = [
                r[0]
                for r in db.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table','view') ORDER BY name"
                )
            ]
            counts = {
                name: db.execute(f"SELECT count(*) FROM {_quote(name)}").fetchone()[0]
                for name in tables
            }
            db.commit()
        for name, path in paths.items():
            _check_asset(path, source["assets"][name])
        manifest = {
            "schema_version": 1,
            "kind": "canine_bulk",
            "hitlist_version": __version__,
            "source": source,
            "limits": bounds,
            "presentation_coverage": "missing",
            "n_records_by_table": counts,
            "artifacts": {n: file_digest(staging / n) for n in ("bulk.sqlite", "reference.fasta")},
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        if cache_bytes(staging) > bounds["max_output_bytes"]:
            raise ValueError("Bulk evidence exceeds max_output_bytes")
        verify_bulk_evidence(staging)
        target.mkdir()
        try:
            for name in ("bulk.sqlite", "reference.fasta", "manifest.json"):
                (staging / name).replace(target / name)
        except BaseException:
            shutil.rmtree(target)
            raise
    except sqlite3.Error as error:
        raise ValueError(f"Bulk source relationship/storage failure: {error}") from error
    finally:
        csv.field_size_limit(previous_field_limit)
        shutil.rmtree(staging)
    return target / "manifest.json"


def _read_db(directory):
    db = sqlite3.connect(
        (Path(directory) / "bulk.sqlite").resolve().as_uri() + "?mode=ro", uri=True
    )
    db.execute("PRAGMA query_only=ON")
    db.execute("PRAGMA trusted_schema=OFF")
    return db


def verify_bulk_evidence(directory):
    """Check frozen bytes and source graph without network or live catalogs."""
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "canine_bulk":
        raise ValueError("Unsupported bulk evidence schema")
    if set(manifest["artifacts"]) != {"bulk.sqlite", "reference.fasta"}:
        raise ValueError("Incomplete bulk evidence artifacts")
    for name, digest in manifest["artifacts"].items():
        if file_digest(root / name) != digest:
            raise ValueError(f"Bulk evidence artifact mismatch: {name}")
    with closing(_read_db(root)) as db:
        adapter = manifest["source"]["adapter"]
        if (
            adapter not in ("peptideatlas", "pfizer")
            or manifest["presentation_coverage"] != "missing"
        ):
            raise ValueError("Invalid bulk evidence source/scope")
        # Schema 1 fixes the normalized evidence semantics. A rehashed database
        # must not turn bulk detections into ligand evidence by changing a view.
        with closing(sqlite3.connect(":memory:")) as expected:
            (_atlas_views if adapter == "peptideatlas" else _pfizer_views)(expected)
            query = "SELECT name, sql FROM sqlite_master WHERE type='view' ORDER BY name"
            if db.execute(query).fetchall() != expected.execute(query).fetchall():
                raise ValueError("Bulk evidence view semantics mismatch")
        names = [
            r[0]
            for r in db.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','view') ORDER BY name"
            )
        ]
        counts = {
            name: db.execute(f"SELECT count(*) FROM {_quote(name)}").fetchone()[0] for name in names
        }
        if counts != manifest["n_records_by_table"]:
            raise ValueError("Bulk evidence table counts mismatch")
        if adapter == "peptideatlas":
            _validate_links(db)
        else:
            _validate_abundances(db)
    return manifest


def load_bulk_evidence(directory, *, table="peptides", tissue=None, peptide=None, max_rows=250000):
    """Read one bounded table/view; oversize selections raise, never truncate.

    Use table='mappings' for all reported peptide/protein alternatives and
    table='raw_<source table>' for unchanged cells and logical source rows.
    Samples are pools or experiment aggregates, not independent dog identities.
    """
    _positive("max_rows", max_rows)
    # Explicit verification is separate from every small query over a large
    # artifact; the writer verifies before publication. Reads remain read-only.
    manifest = json.loads((Path(directory) / "manifest.json").read_text())
    if manifest.get("kind") != "canine_bulk" or manifest.get("schema_version") != 1:
        raise ValueError("Unsupported bulk evidence schema")
    if table not in manifest["n_records_by_table"]:
        raise ValueError(f"Unknown evidence table {table!r}")
    with closing(_read_db(directory)) as db:
        columns = {r[1] for r in db.execute(f"PRAGMA table_info({_quote(table)})")}
        predicates, params = [], []
        for column, value in (("source_tissue", tissue), ("peptide", peptide)):
            if value is not None:
                if column not in columns or not isinstance(value, str) or not value:
                    raise ValueError(f"Table {table!r} does not support this {column} filter")
                predicates.append(f"lower({_quote(column)}) = lower(?)")
                params.append(value)
        query = f"SELECT * FROM {_quote(table)}" + (
            " WHERE " + " AND ".join(predicates) if predicates else ""
        )
        frame = pd.read_sql_query(query + " LIMIT ?", db, params=[*params, max_rows + 1])
    if len(frame) > max_rows:
        raise ValueError(
            "Selection exceeds max_rows; narrow tissue/peptide or explicitly raise the bound"
        )
    if "is_cell_line" in frame:
        frame["is_cell_line"] = frame.is_cell_line.eq("1")
    if "is_ms_observation" in frame:
        frame["is_ms_observation"] = frame.is_ms_observation.astype(bool)
    return frame
