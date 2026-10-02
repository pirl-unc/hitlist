"""Replay retained gene/locus statements against persisted candidate coverage.

This is an audit of candidate inference, not a rebuild of the source corpus.
Previously promoted class-only restrictions cannot be recovered from parquet.
SQLite distinct sets and bounded Arrow batches avoid a full in-memory export.
"""

import argparse
import hashlib
import importlib.metadata
import json
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from mhcgnomes import Class2Locus, Gene, MhcClass

from hitlist import curation
from hitlist.version import __version__


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(observations, output, *, batch_size=8192):
    """Write counts, per-paper/locus gains and a reproducibility manifest."""
    observations, output = Path(observations), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checkout = Path(__file__).resolve().parents[1]
    if Path(curation.__file__).resolve() != checkout / "hitlist/curation.py":
        raise RuntimeError("Audit imported a different checkout")
    parquet = pq.ParquetFile(observations)
    columns = [
        "peptide",
        "pmid",
        "mhc_restriction",
        "mhc_class",
        "host_mhc_types",
        "mhc_allele_set",
        "attributed_sample_label",
    ]
    required = set(columns) - {"attributed_sample_label"}
    if not required.issubset(parquet.schema.names):
        raise ValueError(f"Missing audit columns: {sorted(required - set(parquet.schema.names))}")
    columns = [column for column in columns if column in parquet.schema.names]
    counts = Counter()
    groups = {}
    with (
        tempfile.TemporaryDirectory(prefix="hitlist-candidate-audit-") as directory,
        sqlite3.connect(str(Path(directory) / "peptides.sqlite")) as database,
    ):
        database.execute("PRAGMA cache_size=-4096")
        database.execute("PRAGMA temp_store=FILE")
        database.execute(
            "CREATE TABLE coverage (scope TEXT, pmid TEXT, peptide TEXT, "
            "before INTEGER, after INTEGER, PRIMARY KEY(scope, pmid, peptide)) WITHOUT ROWID"
        )
        insert = (
            "INSERT INTO coverage VALUES (?, ?, ?, ?, ?) ON CONFLICT(scope, pmid, peptide) "
            "DO UPDATE SET before=MAX(before, excluded.before), after=MAX(after, excluded.after)"
        )
        for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
            entries = []
            for row in batch.to_pylist():
                counts["n_rows"] += 1
                restriction = row.get("mhc_restriction") or ""
                pmid = str(row.get("pmid") or "")
                peptide = row.get("peptide") or ""
                before = bool(row.get("mhc_allele_set"))
                parsed = curation._cached_parse(restriction)
                scope = (
                    "blank"
                    if not restriction
                    else (
                        "gene_locus"
                        if isinstance(parsed, (Gene, Class2Locus))
                        else ("class_only" if isinstance(parsed, MhcClass) else "")
                    )
                )
                after = before
                if scope:
                    context = curation.pmid_mhc_species_context(pmid)
                    label = row.get("attributed_sample_label") or ""
                    attributed = frozenset()
                    if label and curation.peptide_attribution_applies_to_row(pmid, restriction):
                        attributed = curation._pmid_sample_alleles(int(pmid)).get(
                            label, frozenset()
                        )
                    candidates, provenance, _ = curation.expand_allele_set(
                        restriction,
                        row.get("host_mhc_types") or "",
                        pmid,
                        row.get("mhc_class") or "",
                        attributed,
                        species_context=context,
                    )
                    after = bool(candidates)
                    key = (scope, pmid, restriction)
                    group = groups.setdefault(key, Counter())
                    group["n_rows"] += 1
                    group["n_rows_before"] += before
                    group["n_rows_after"] += after
                    group["n_rows_gained"] += after and not before
                    group["n_rows_lost"] += before and not after
                    group[f"n_rows_{provenance}"] += 1
                    if scope == "blank":
                        available = bool(
                            curation._parse_host_mhc_types(row.get("host_mhc_types") or "")
                        )
                        if pmid:
                            available |= bool(curation._pmid_allele_pool(int(pmid)))
                        group["n_rows_with_typing"] += available
                    if peptide:
                        entries.append((scope, pmid, peptide, before, after))
                        entries.append((scope + ":" + restriction, pmid, peptide, before, after))
                if peptide and (before or after):
                    entries.append(("corpus", pmid, peptide, before, after))
            database.executemany(insert, entries)
            database.commit()

        def peptide_counts(scope, pmid=None):
            where = "scope=?" + (" AND pmid=?" if pmid is not None else "")
            parameters = (scope, pmid) if pmid is not None else (scope,)
            # A sequence shared across papers counts once globally.
            rows = database.execute(
                "SELECT COUNT(*), COALESCE(SUM(before),0), COALESCE(SUM(after),0), "
                "COALESCE(SUM(after AND NOT before),0), COALESCE(SUM(before AND NOT after),0) "
                "FROM (SELECT peptide, MAX(before) AS before, MAX(after) AS after "
                f"FROM coverage WHERE {where} GROUP BY peptide)",
                parameters,
            ).fetchone()
            return dict(
                zip(
                    [
                        "n_peptides",
                        "n_peptides_before",
                        "n_peptides_after",
                        "n_peptides_gained",
                        "n_peptides_lost",
                    ],
                    rows,
                )
            )

        per_locus = []
        per_paper = {}
        for (scope, pmid, restriction), group in sorted(groups.items()):
            record = {"scope": scope, "pmid": pmid, "mhc_restriction": restriction, **group}
            record.update(peptide_counts(scope + ":" + restriction, pmid))
            per_locus.append(record)
            paper = per_paper.setdefault((scope, pmid), Counter())
            paper.update(group)
        papers = []
        for (scope, pmid), paper in sorted(per_paper.items()):
            record = {"scope": scope, "pmid": pmid, **paper, **peptide_counts(scope, pmid)}
            record["corpus_candidate_peptides"] = peptide_counts("corpus", pmid)
            papers.append(record)
        summary = {"n_rows": counts["n_rows"]}
        for scope in ("gene_locus", "class_only", "blank"):
            total = Counter()
            for (group_scope, _), paper in per_paper.items():
                if group_scope == scope:
                    total.update(paper)
            summary[scope] = {**total, **peptide_counts(scope)}
        summary["corpus_candidate_peptides"] = peptide_counts("corpus")
    manifest = {
        "audit_version": 1,
        "hitlist_version": __version__,
        "mhcgnomes_version": importlib.metadata.version("mhcgnomes"),
        "scope": "Retained gene/locus and class-only restrictions, plus blanks; persisted candidate baseline; other rows unchanged",
        "input": {
            "file": observations.name,
            "sha256": sha256(observations),
            "n_rows": counts["n_rows"],
        },
        "code_sha256": {
            str(path.relative_to(checkout)): sha256(path)
            for path in [
                Path(__file__).resolve(),
                checkout / "hitlist/curation.py",
            ]
        },
        "curation_sha256": sha256(checkout / "hitlist/data/pmid_overrides.yaml"),
    }
    for name, value in [
        ("manifest", manifest),
        ("summary", summary),
        ("per_paper", papers),
        ("per_locus", per_locus),
    ]:
        (output / f"{name}.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.observations, args.output), indent=2, sort_keys=True))
