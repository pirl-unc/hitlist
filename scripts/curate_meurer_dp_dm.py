"""Package verified PXD017154 native identifications for the shared MS scanner.

The input is the independently verified native extraction pinned in Presto's
2026-11-10_1454 source-recovery family, not converted mzIdentML pass flags.
No scores, spectral counts or nondetections become affinity/abundance labels.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml

SOURCE_SHA256 = "daaa04962f690a184c553ac8ec55c8545f40bdfbecac61ce89a8997d94124beb"
SOURCE_RECORD = (
    "https://github.com/pirl-unc/presto/tree/"
    "e21d6dbf135400b5d8a598fca4016a8cbc217f53/"
    "experiments/2026-11-10_1454_codex_dp-dm-source-recovery"
)
SUPPLEMENT = "https://ars.els-cdn.com/content/image/1-s2.0-S000649712100313X-mmc1.pdf"
EXPECTED_COUNTS = {
    "DP10_A": 3052,
    "DP10_B": 3552,
    "DP10_DM_A": 1687,
    "DP10_DM_B": 1728,
    "DP401_DM_A": 1602,
    "DP401_DM_B": 1597,
    "DP402_A": 2956,
    "DP402_B": 2815,
    "DP402_DM_A": 1446,
    "DP402_DM_B": 1112,
}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _write_csv(path, rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    payload = gzip.compress(buffer.getvalue().encode(), mtime=0)
    path.write_bytes(payload)
    return {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}


def package_source(source: Path, output: Path) -> dict:
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != SOURCE_SHA256:
        raise ValueError("Input is not the pinned independently verified native extraction")
    output.mkdir(parents=True, exist_ok=True)
    if list(output.glob("meurer_2021_dp_dm_*")):
        raise FileExistsError("Refusing to overwrite existing source assets")
    by_sample = defaultdict(list)
    controls = []
    seen = set()
    sources = {}
    for line in gzip.decompress(payload).splitlines():
        row = json.loads(line)
        sample = row["sample"]
        label = sample["biological_sample"]
        key = (label, row["peptide"])
        if key in seen:
            raise ValueError(f"Repeated biological-sample peptide: {key}")
        seen.add(key)
        if row["training_eligible"] or row["split"] != "unpartitioned":
            raise ValueError("Source extraction must not make consumer split decisions")
        if row["has_ptm"] or row["peptide_modifications"] or not row["supporting_psms"]:
            raise ValueError("Expected source-qualified unmodified native PSM support")
        for support in row["supporting_psms"]:
            name, digest = support["source_file"], support["source_sha256"]
            if name in sources and sources[name] != digest:
                raise ValueError(f"Conflicting original file identity: {name}")
            sources[name] = digest
        record = {
            "peptide": row["peptide"],
            "mhc_class": "II",
            "mhc_restriction": (
                f"{sample['mhc_alpha']}/{sample['mhc_beta']}"
                if sample["mhc_alpha"] and sample["mhc_beta"]
                else ""
            ),
            "has_ptm": "false",
            "peptide_modifications": "",
            "attributed_sample_label": label,
            "source_accession": "PXD017154",
            "native_confidence": "high",
            "native_target_fpr_high": "0.01",
            "capture_role": sample["capture_role"],
            "observed_in_no_dp_control": str(row["observed_in_no_dp_control"]).lower(),
            "source_qualified_candidate": str(row["source_qualified_candidate"]).lower(),
            "n_supporting_psms": len(row["supporting_psms"]),
            "n_acquisitions": len(row["runs"]),
            "source_acquisitions": _json(row["runs"]),
            "native_psm_support": _json(row["supporting_psms"]),
            "native_protein_annotations": _json(row["protein_annotations"]),
            "source_sample_metadata": _json(sample),
        }
        if row["source_qualified_candidate"]:
            if (
                sample["capture_role"] != "engineered_dp_capture"
                or row["observed_in_no_dp_control"]
            ):
                raise ValueError("A capture control cannot enter the positive ligand table")
            by_sample[label].append(record)
        else:
            controls.append(record)
    counts = {name: len(rows) for name, rows in by_sample.items()}
    if counts != EXPECTED_COUNTS or len(controls) != 87 or len(sources) != 32:
        raise ValueError(f"Pinned source census changed: {counts}, controls={len(controls)}")
    entries = []
    for label, rows in sorted(by_sample.items()):
        filename = f"meurer_2021_dp_dm_{label.lower()}.csv.gz"
        receipt = _write_csv(output / filename, rows)
        entries.append(
            {
                "pmid": 33025005,
                "file": filename,
                **receipt,
                "study_label": f"Meurer 2021 — HLA-DP/DM HeLa {label}",
                "source": f"PXD017154 original MSF native high-confidence PSMs; {SOURCE_RECORD}",
                "note": (
                    "Biological-sample peptide detections, technical acquisitions collapsed with "
                    "native PSM/file/form lineage retained. Conservative unmodified, unique visible "
                    "rank-1, protein-mapped subset; no-DP controls and control-overlap sequences "
                    "excluded. No affinity, abundance, or nondetection-negative labels."
                ),
                "defaults": {
                    "host": "Homo sapiens (human)",
                    "species": "Homo sapiens",
                    "source_organism": "Homo sapiens",
                    "culture_condition": "Cell Line / Clone",
                    "cell_name": "HeLa",
                    "source_tissue": "Cervix",
                    "disease": "cervical cancer",
                    "attributed_sample_label": label,
                },
            }
        )
    controls_name = "meurer_2021_dp_dm_controls.csv.gz"
    controls_receipt = _write_csv(output / controls_name, controls)
    source_manifest = {
        "pmid": 33025005,
        "accession": "PXD017154",
        "primary_supplement": SUPPLEMENT,
        "source_recovery": SOURCE_RECORD,
        "input_sha256": SOURCE_SHA256,
        "source_original_msf_sha256": sources,
        "n_candidate_peptide_sample_rows": sum(counts.values()),
        "n_distinct_candidate_peptides": len(
            {r["peptide"] for rows in by_sample.values() for r in rows}
        ),
        "n_biological_samples": len(counts),
        "sample_candidate_counts": counts,
        "control_companion": {"file": controls_name, **controls_receipt, "n_rows": len(controls)},
        "control_role_counts": dict(Counter(r["capture_role"] for r in controls)),
        "confidence_contract": "Native ConfidenceLevel=3; TargetFPRHigh=0.01; converted passThreshold not used",
        "input_scope": "Unpartitioned source evidence; consumer overlap/split qualification still required",
        "supplementary_entries": entries,
    }
    (output / "meurer_2021_dp_dm_source.yaml").write_text(
        yaml.safe_dump(source_manifest, sort_keys=False, allow_unicode=True, width=100)
    )
    return source_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = package_source(args.source, args.output)
    print(json.dumps({k: v for k, v in manifest.items() if k.startswith("n_")}, indent=2))


if __name__ == "__main__":
    main()
