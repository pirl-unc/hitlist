"""Source-backed biological-arm projection for the Gomez-Zepeda deposit.

The original S3/S5 rows and memberships remain immutable.  This revision adds
an explicit arm vocabulary, a membership-to-arm relation, and a derived-row
projection.  A derived peptide can therefore be represented as supporting one
or several biological arms without pretending that a merged row has a donor or
single restricting condition.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .evidence_snapshot import snapshot_file
from .gomez_source_lineage import MEMBERSHIPS, RESULTS, verify_gomez_source_package
from .provenance import file_digest

PACKAGE_SCHEMA_VERSION = 12
CATALOG_SCHEMA_VERSION = 17
REVISION = "gomez_zepeda_2024_arm_projection_v1"
PARENT_SHA256 = "ade958582dd2eb580f4e02187cd7445621e72fc07ec294a97624a09ba24a2659"
PARENT_MANIFEST = Path("manifests/gomez_source.json")
ARTICLE_XML_SHA256 = "3311abf23d895b7f37f18cd8dbf5ea3a273d785d6c34db24b2aaa0b5a1d97260"
ROOT = Path("gomez_projection")
ARMS = ROOT / "arms.parquet"
MEMBERSHIP_ARMS = ROOT / "membership_arms.parquet"
DERIVED_SUPPORT = ROOT / "derived_arm_support.parquet"
DERIVED_PROJECTION = ROOT / "derived_arm_projection.parquet"
ASSAY_CONTEXT = ROOT / "assay_context.parquet"
REVIEW = ROOT / "review.json"
ARTICLE_XML = ROOT / "article.xml"

ARM_DEFINITIONS = (
    {
        "arm_id": "gomez:jy_wild_type",
        "material_id": "JY",
        "sample_group": "JY",
        "arm_label": "JY wild-type",
        "condition_id": "jy_wild_type",
        "condition_status": "annotated",
        "condition_evidence": "primary_source",
        "condition_reference": "PMC10937930 Experiment design; MOESM7 S4; MOESM9 S6",
        "condition_control": "untreated",
        "condition_transfection": "",
        "condition_material": "cultured",
        "mhc": "HLA-A*02:01 HLA-B*07:02 HLA-C*07:02",
        "source_phase_scope": "s3_method_comparison;s5_spike_experiment",
        "source_condition_scope": "JY/JY_WT/WT;JY/JY/None",
        "mapping_basis": "cell_line plus native WT condition and S3 untransfected method-comparison rows",
    },
    {
        "arm_id": "gomez:jy_spike_s1",
        "material_id": "JY",
        "sample_group": "JY",
        "arm_label": "JY spike S1 transfectant",
        "condition_id": "jy_spike_s1_transfectant",
        "condition_status": "annotated",
        "condition_evidence": "primary_source",
        "condition_reference": "PMC10937930 Cell transfection; MOESM9 S6 JY_S1",
        "condition_control": "",
        "condition_transfection": "SARS-CoV-2 spike S1",
        "condition_material": "cultured",
        "mhc": "HLA-A*02:01 HLA-B*07:02 HLA-C*07:02",
        "source_phase_scope": "s5_spike_experiment",
        "source_condition_scope": "JY/JY_S/JY_S1/S1",
        "mapping_basis": "native S5 ConditionGenotype=JY_S1 and Genotype=S1",
    },
    {
        "arm_id": "gomez:jy_spike_s2",
        "material_id": "JY",
        "sample_group": "JY",
        "arm_label": "JY spike S2 transfectant",
        "condition_id": "jy_spike_s2_transfectant",
        "condition_status": "annotated",
        "condition_evidence": "primary_source",
        "condition_reference": "PMC10937930 Cell transfection; MOESM9 S6 JY_S2",
        "condition_control": "",
        "condition_transfection": "SARS-CoV-2 spike S2",
        "condition_material": "cultured",
        "mhc": "HLA-A*02:01 HLA-B*07:02 HLA-C*07:02",
        "source_phase_scope": "s5_spike_experiment",
        "source_condition_scope": "JY/JY_S/JY_S2/S2",
        "mapping_basis": "native S5 ConditionGenotype=JY_S2 and Genotype=S2",
    },
    {
        "arm_id": "gomez:raji_wild_type",
        "material_id": "Raji",
        "sample_group": "Raji",
        "arm_label": "Raji wild-type",
        "condition_id": "raji_wild_type",
        "condition_status": "annotated",
        "condition_evidence": "primary_source",
        "condition_reference": "PMC10937930 Experiment design; MOESM9 S6 Raji_WT",
        "condition_control": "untreated",
        "condition_transfection": "",
        "condition_material": "cultured",
        "mhc": "HLA-A*03:01 HLA-B*15:10 HLA-C*03:04 HLA-C*04:01",
        "source_phase_scope": "s5_spike_experiment",
        "source_condition_scope": "Raji/Raji_WT/WT",
        "mapping_basis": "native S5 ConditionGenotype=Raji_WT and Genotype=WT",
    },
    {
        "arm_id": "gomez:raji_spike_s1",
        "material_id": "Raji",
        "sample_group": "Raji",
        "arm_label": "Raji spike S1 transfectant",
        "condition_id": "raji_spike_s1_transfectant",
        "condition_status": "annotated",
        "condition_evidence": "primary_source",
        "condition_reference": "PMC10937930 Cell transfection; MOESM9 S6 Raji_S1",
        "condition_control": "",
        "condition_transfection": "SARS-CoV-2 spike S1",
        "condition_material": "cultured",
        "mhc": "HLA-A*03:01 HLA-B*15:10 HLA-C*03:04 HLA-C*04:01",
        "source_phase_scope": "s5_spike_experiment",
        "source_condition_scope": "Raji/Raji_S/Raji_S1/S1",
        "mapping_basis": "native S5 ConditionGenotype=Raji_S1 and Genotype=S1",
    },
    {
        "arm_id": "gomez:raji_spike_s2",
        "material_id": "Raji",
        "sample_group": "Raji",
        "arm_label": "Raji spike S2 transfectant",
        "condition_id": "raji_spike_s2_transfectant",
        "condition_status": "annotated",
        "condition_evidence": "primary_source",
        "condition_reference": "PMC10937930 Cell transfection; MOESM9 S6 Raji_S2",
        "condition_control": "",
        "condition_transfection": "SARS-CoV-2 spike S2",
        "condition_material": "cultured",
        "mhc": "HLA-A*03:01 HLA-B*15:10 HLA-C*03:04 HLA-C*04:01",
        "source_phase_scope": "s5_spike_experiment",
        "source_condition_scope": "Raji/Raji_S/Raji_S2/S2",
        "mapping_basis": "native S5 ConditionGenotype=Raji_S2 and Genotype=S2",
    },
    {
        "arm_id": "gomez:hela",
        "material_id": "HeLa",
        "sample_group": "HeLa",
        "arm_label": "HeLa cervical carcinoma",
        "condition_id": "hela_cervical_carcinoma",
        "condition_status": "annotated",
        "condition_evidence": "primary_source",
        "condition_reference": "PMC10937930 Experiment design; MOESM7 S4",
        "condition_control": "untreated",
        "condition_transfection": "",
        "condition_material": "cultured",
        "mhc": "HLA-A*68:02 HLA-B*15:03 HLA-C*12:03",
        "source_phase_scope": "s3_method_comparison",
        "source_condition_scope": "HeLa/*",
        "mapping_basis": "native S3 Cell.Line=HeLa",
    },
    {
        "arm_id": "gomez:sk_mel_37",
        "material_id": "SK-MEL-37",
        "sample_group": "SK-MEL-37",
        "arm_label": "SK-MEL-37 melanoma",
        "condition_id": "sk_mel_37_melanoma",
        "condition_status": "annotated",
        "condition_evidence": "primary_source",
        "condition_reference": "PMC10937930 Experiment design; MOESM7 S4",
        "condition_control": "untreated",
        "condition_transfection": "",
        "condition_material": "cultured",
        "mhc": "HLA-A*02:01 HLA-A*11:01 HLA-B*15:01 HLA-B*56:01 HLA-C*01:02",
        "source_phase_scope": "s3_method_comparison",
        "source_condition_scope": "SK-MEL-37/*",
        "mapping_basis": "native S3 Cell.Line=SK-MEL-37",
    },
    {
        "arm_id": "gomez:human_plasma",
        "material_id": "Plasma",
        "sample_group": "Plasma",
        "arm_label": "Human plasma commercial pool",
        "condition_id": "human_plasma_commercial_pool",
        "condition_status": "annotated",
        "condition_evidence": "primary_source",
        "condition_reference": "PMC10937930 Experiment design; MOESM7 S4",
        "condition_control": "untreated",
        "condition_transfection": "",
        "condition_material": "biofluid",
        "mhc": "HLA class I",
        "source_phase_scope": "s3_method_comparison",
        "source_condition_scope": "Plasma/*",
        "mapping_basis": "native S3 Cell.Line=Plasma",
    },
)
ARM_BY_ID = {row["arm_id"]: row for row in ARM_DEFINITIONS}


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _schema(rows, fields):
    return pa.schema(
        [(name, pa.int64() if name.startswith("n_") else pa.string()) for name in fields]
    )


def _write_rows(path, rows, fields):
    schema = _schema(rows, fields)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression="zstd")
    return schema


def _arm_for_native(cell, phase, genotype, condition):
    if phase == "s3_method_comparison":
        return {
            "JY": "gomez:jy_wild_type",
            "HeLa": "gomez:hela",
            "SK-MEL-37": "gomez:sk_mel_37",
            "Plasma": "gomez:human_plasma",
        }.get(cell)
    if phase != "s5_spike_experiment" or cell not in {"JY", "Raji"}:
        raise ValueError(f"Unknown Gomez source arm: {phase} {cell}")
    if genotype not in {"WT", "S1", "S2"}:
        raise ValueError(f"Unresolved native Gomez genotype: {cell} {genotype!r}")
    return f"gomez:{cell.lower()}_" + (
        "wild_type" if genotype == "WT" else f"spike_{genotype.lower()}"
    )


def _build_projection(parent):
    result_file = pq.ParquetFile(parent / RESULTS)
    metadata = {}
    for batch in result_file.iter_batches(
        batch_size=32768,
        columns=[
            "membership_id",
            "source_phase",
            "cell_line_name",
            "condition_text",
            "ConditionGenotype",
            "Genotype",
        ],
    ):
        for row in batch.to_pylist():
            membership_id = row["membership_id"]
            genotype = row.get("ConditionGenotype") or row.get("Genotype") or ""
            arm_id = _arm_for_native(
                row["cell_line_name"],
                row["source_phase"],
                genotype.split("_")[-1] if genotype not in {"WT", "S1", "S2"} else genotype,
                row["condition_text"],
            )
            previous = metadata.setdefault(membership_id, {"arm_id": arm_id, "genotype": genotype})
            if previous != {"arm_id": arm_id, "genotype": genotype}:
                raise ValueError(f"Native source arm changes inside membership {membership_id}")

    membership_file = pq.ParquetFile(parent / MEMBERSHIPS)
    membership_rows = []
    by_derived = defaultdict(list)
    for batch in membership_file.iter_batches(batch_size=32768):
        for row in batch.to_pylist():
            meta = metadata.get(row["membership_id"])
            if meta is None:
                raise ValueError(f"Membership missing native result rows: {row['membership_id']}")
            arm = ARM_BY_ID[meta["arm_id"]]
            out = {
                **{
                    name: row.get(name)
                    for name in (
                        "membership_id",
                        "derived_observation_id",
                        "source_phase",
                        "source_asset_id",
                        "peptide_sequence",
                        "condition_text",
                        "biological_replicate_label",
                        "sample_label_raw",
                        "acquisition_id",
                        "n_source_result_rows",
                    )
                },
                "arm_id": arm["arm_id"],
                "material_id": arm["material_id"],
                "mapping_basis": arm["mapping_basis"],
                "source_condition_genotype": meta["genotype"],
            }
            membership_rows.append(out)
            if row["derived_observation_id"]:
                by_derived[row["derived_observation_id"]].append(out)

    support_rows = []
    projection_rows = []
    for derived_id in sorted(by_derived):
        members = by_derived[derived_id]
        arm_ids = sorted({row["arm_id"] for row in members})
        phases = sorted({row["source_phase"] for row in members})
        n_source_rows = sum(row["n_source_result_rows"] for row in members)
        material_ids = sorted({row["material_id"] for row in members})
        if len(material_ids) != 1:
            raise ValueError(f"Derived observation crosses source materials: {derived_id}")
        status = "unique_source_arm" if len(arm_ids) == 1 else "multi_source_arm"
        common = {
            "derived_observation_id": derived_id,
            "material_id": material_ids[0],
            "n_supported_arms": len(arm_ids),
            "supported_arm_ids_json": _json(arm_ids),
            "source_phases": ";".join(phases),
            "n_source_memberships": len(members),
            "n_source_result_rows": n_source_rows,
        }
        support_rows.append(
            {
                **common,
                "source_membership_ids_json": _json(
                    sorted(row["membership_id"] for row in members)
                ),
                "support_status": status,
            }
        )
        projection_rows.append(
            {
                **common,
                "unique_arm_id": arm_ids[0] if len(arm_ids) == 1 else "",
                "arm_projection_status": status,
                "projection_basis": "all linked original S3/S5 memberships",
            }
        )
    return membership_rows, support_rows, projection_rows


def _project_context(parent, destination, projection_rows):
    projection = {row["derived_observation_id"]: row for row in projection_rows}
    context_schema = pq.read_schema(parent / "typed/assay_context.parquet")
    source_fields = set(context_schema.names)
    writer = None
    changed = 0
    status_counts = defaultdict(int)
    context_file = ds.dataset(parent / "typed/assay_context.parquet", format="parquet")
    context_condition_fields = [
        "condition_id",
        "condition_status",
        "condition_evidence",
        "condition_reference",
        "condition_control",
        "condition_control_for",
        "condition_combination",
        "condition_knockout_genes",
        "condition_knockdown_genes",
        "condition_overexpression_genes",
        "condition_genetic_variants",
        "condition_transfection",
        "condition_transduction",
        "condition_cytokines",
        "condition_drugs",
        "condition_infection",
        "condition_stimulation",
        "condition_antigen_exposure",
        "condition_background",
        "condition_mhc_context",
        "condition_culture",
        "condition_material",
        "condition_labeling",
    ]
    table = context_file.to_table(
        filter=ds.field("observation_id").isin(list(projection)),
    )
    for batch in table.to_batches(max_chunksize=16384):
        rows = batch.to_pylist()
        for row in rows:
            item = projection.get(row["observation_id"])
            if item is None:
                continue
            changed += 1
            status = item["arm_projection_status"]
            status_counts[status] += 1
            row["sample_group"] = item["material_id"]
            row["arm_resolution"] = (
                "resolved" if status == "unique_source_arm" else "multi_arm_evidence"
            )
            row["sample_attribution"] = (
                "source_unique_arm" if status == "unique_source_arm" else "source_multi_arm"
            )
            if status == "unique_source_arm":
                arm = ARM_BY_ID[item["unique_arm_id"]]
                row.update(
                    {
                        "arm_id": arm["arm_id"],
                        "arm_link_status": "attributed_arm",
                        "sample_label": arm["arm_label"],
                        "condition_id": arm["condition_id"],
                        "condition_status": arm["condition_status"],
                        "condition_evidence": arm["condition_evidence"],
                        "condition_reference": arm["condition_reference"],
                        "condition_control": arm["condition_control"],
                        "condition_transfection": arm["condition_transfection"],
                        "condition_material": arm["condition_material"],
                        "sample_mhc": arm["mhc"],
                    }
                )
            else:
                row.update(
                    {
                        "arm_id": None,
                        "arm_link_status": "multi_arm_evidence",
                        "sample_label": None,
                        "condition_status": "mixed",
                        "condition_evidence": "primary_source",
                        "condition_reference": "PMC10937930 Experiment design; source-backed arm projection",
                        "sample_mhc": None,
                    }
                )
                for field in context_condition_fields:
                    if field not in {
                        "condition_status",
                        "condition_evidence",
                        "condition_reference",
                        "condition_material",
                    }:
                        row[field] = None
                materials = {
                    ARM_BY_ID[arm_id]["condition_material"]
                    for arm_id in json.loads(item["supported_arm_ids_json"])
                }
                row["condition_material"] = materials.pop() if len(materials) == 1 else None
            if "sample_mhc_origin" in source_fields:
                row["sample_mhc_origin"] = "source_arm_projection"
        table = pa.Table.from_pylist(rows, schema=context_schema)
        if writer is None:
            writer = pq.ParquetWriter(destination, context_schema, compression="zstd")
        writer.write_table(table)
    if writer is None:
        raise ValueError("Assay context is empty")
    writer.close()
    if changed != len(projection):
        raise ValueError(f"Context projection joined {changed} rows, expected {len(projection)}")
    return {
        "n_changed_context_rows": changed,
        "by_projection_status": dict(sorted(status_counts.items())),
    }


def _review(article_xml, membership_rows, projection_rows, context_counts):
    return {
        "revision": REVISION,
        "article_pmid": "38480730",
        "article_doi": "10.1038/s41467-024-46380-y",
        "article_xml_sha256": file_digest(article_xml)["sha256"],
        "article_xml_expected_sha256": ARTICLE_XML_SHA256,
        "source_membership_contract": {
            "native_fields": ["ConditionGenotype", "Condition", "Genotype", "BioReplicate"],
            "source_phase_scope": ["s3_method_comparison", "s5_spike_experiment"],
            "arm_count": len(ARM_DEFINITIONS),
            "n_membership_arm_rows": len(membership_rows),
            "n_derived_support_rows": len(projection_rows),
        },
        "condition_anchors": [
            "PMC10937930 Experiment design",
            "PMC10937930 Cell transfection",
            "MOESM7 S4_pep_all_06154.csv",
            "MOESM9 S6_pep_all_06133b.csv",
        ],
        "context_projection": context_counts,
        "no_rows_dropped": True,
        "no_new_assay_labels": True,
    }


def _manifest_files(staging):
    return {
        str(path.relative_to(staging)): file_digest(path)
        for path in sorted(staging.rglob("*"))
        if path.is_file() and path != staging / "manifest.json"
    }


def verify_gomez_arm_projection_package(
    directory, *, manifest_path="manifest.json", allow_additional_files=False
):
    """Verify inherited bytes, source mappings, row counts, and projection joins."""
    directory = Path(directory)
    manifest_file = directory / manifest_path
    manifest = json.loads(manifest_file.read_text())
    if (
        manifest.get("schema_version") != PACKAGE_SCHEMA_VERSION
        or manifest.get("catalog_schema_version") != CATALOG_SCHEMA_VERSION
        or manifest.get("revision") != REVISION
        or manifest.get("parent_manifest_path") != str(PARENT_MANIFEST)
        or manifest.get("parent_manifest_sha256") != PARENT_SHA256
    ):
        raise ValueError("Unsupported Gomez-Zepeda arm-projection package")
    actual = {
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file() and path != manifest_file
    }
    if not allow_additional_files and set(manifest.get("files", {})) != actual:
        raise ValueError("Gomez-Zepeda arm-projection inventory mismatch")
    if allow_additional_files and not set(manifest.get("files", {})) <= actual:
        raise ValueError("Gomez-Zepeda arm-projection inventory mismatch")
    for name, digest in manifest["files"].items():
        if file_digest(directory / name)["sha256"] != digest["sha256"]:
            raise ValueError(f"Changed arm-projection artifact: {name}")
    parent = verify_gomez_source_package(
        directory, manifest_path=PARENT_MANIFEST, allow_additional_files=True
    )
    if file_digest(directory / PARENT_MANIFEST)["sha256"] != PARENT_SHA256:
        raise ValueError("Arm-projection predecessor changed")
    if any(manifest.get(name) != parent.get(name) for name in ("adjudication_manifest_path",)):
        raise ValueError("Arm-projection package changed inherited metadata")
    article = directory / ARTICLE_XML
    if file_digest(article)["sha256"] != ARTICLE_XML_SHA256:
        raise ValueError("Gomez article XML changed")
    membership_rows, support_rows, projection_rows = _build_projection(directory)
    if pq.ParquetFile(directory / MEMBERSHIP_ARMS).metadata.num_rows != len(membership_rows):
        raise ValueError("Membership arm row count changed")
    if pq.ParquetFile(directory / DERIVED_SUPPORT).metadata.num_rows != len(support_rows):
        raise ValueError("Derived support row count changed")
    if pq.ParquetFile(directory / DERIVED_PROJECTION).metadata.num_rows != len(projection_rows):
        raise ValueError("Derived projection row count changed")
    projection_context = _project_context(
        directory, directory / ".context.verify.parquet", projection_rows
    )
    (directory / ".context.verify.parquet").unlink()
    if manifest.get("reconciliation", {}).get("context_projection") != projection_context:
        raise ValueError("Context projection reconciliation changed")
    review = json.loads((directory / REVIEW).read_text())
    if review.get("article_xml_sha256") != ARTICLE_XML_SHA256:
        raise ValueError("Review ledger article hash changed")
    return manifest


def write_gomez_arm_projection_package(predecessor_package, article_xml, destination):
    """Append source-backed arm projection to the verified schema-11 package."""
    predecessor_package = Path(predecessor_package)
    article_xml = Path(article_xml)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    parent = verify_gomez_source_package(predecessor_package)
    if file_digest(predecessor_package / "manifest.json")["sha256"] != PARENT_SHA256:
        raise ValueError("Gomez arm projection requires pinned schema-11 predecessor")
    if file_digest(article_xml)["sha256"] != ARTICLE_XML_SHA256:
        raise ValueError("Unexpected Gomez article XML")
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        for name in parent["files"]:
            snapshot_file(predecessor_package / name, staging / name)
        snapshot_file(predecessor_package / "manifest.json", staging / PARENT_MANIFEST)
        (staging / ROOT).mkdir(parents=True, exist_ok=True)
        snapshot_file(article_xml, staging / ARTICLE_XML)
        membership_rows, support_rows, projection_rows = _build_projection(staging)
        fields = {
            "arms": tuple(ARM_DEFINITIONS[0].keys()),
            "membership_arms": (
                "membership_id",
                "derived_observation_id",
                "arm_id",
                "material_id",
                "source_phase",
                "source_asset_id",
                "peptide_sequence",
                "condition_text",
                "biological_replicate_label",
                "sample_label_raw",
                "acquisition_id",
                "mapping_basis",
                "source_condition_genotype",
                "n_source_result_rows",
            ),
            "derived_support": (
                "derived_observation_id",
                "material_id",
                "n_supported_arms",
                "supported_arm_ids_json",
                "source_phases",
                "source_membership_ids_json",
                "n_source_memberships",
                "n_source_result_rows",
                "support_status",
            ),
            "derived_projection": (
                "derived_observation_id",
                "material_id",
                "n_supported_arms",
                "supported_arm_ids_json",
                "source_phases",
                "n_source_memberships",
                "n_source_result_rows",
                "unique_arm_id",
                "arm_projection_status",
                "projection_basis",
            ),
        }
        _write_rows(staging / ARMS, list(ARM_DEFINITIONS), fields["arms"])
        _write_rows(staging / MEMBERSHIP_ARMS, membership_rows, fields["membership_arms"])
        _write_rows(staging / DERIVED_SUPPORT, support_rows, fields["derived_support"])
        _write_rows(staging / DERIVED_PROJECTION, projection_rows, fields["derived_projection"])
        context_counts = _project_context(staging, staging / ASSAY_CONTEXT, projection_rows)
        review = _review(article_xml, membership_rows, projection_rows, context_counts)
        (staging / REVIEW).write_text(json.dumps(review, indent=2, sort_keys=True) + "\n")
        files = _manifest_files(staging)
        from .version import __version__

        manifest = {
            "schema_version": PACKAGE_SCHEMA_VERSION,
            "catalog_schema_version": CATALOG_SCHEMA_VERSION,
            "hitlist_version": __version__,
            "revision": REVISION,
            "parent_kind": "gomez_source",
            "parent_manifest_path": str(PARENT_MANIFEST),
            "parent_manifest_sha256": PARENT_SHA256,
            "adjudication_manifest_path": parent["adjudication_manifest_path"],
            "source_reference": parent["reference"],
            "reconciliation": {
                "n_source_arms": len(ARM_DEFINITIONS),
                "n_membership_arm_rows": len(membership_rows),
                "n_derived_support_rows": len(support_rows),
                "n_derived_projection_rows": len(projection_rows),
                "context_projection": context_counts,
            },
            "files": files,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        verify_gomez_arm_projection_package(staging)
        staging.rename(destination)
        return destination / "manifest.json"
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
