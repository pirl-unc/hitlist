"""Offline species bundles preserve counterevidence and measurement scope."""

import hashlib
import json
import shutil

import pandas as pd
import pytest

from hitlist.provenance import file_digest


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def sequence_id(value):
    return hashlib.sha256(value.encode()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


@pytest.fixture
def inputs(tmp_path):
    root = tmp_path / "inputs"
    root.mkdir()
    # Two loci have identical protein sequences, plus an unannotated background
    # protein and an I/L alternative. The candidate list must not hide either.
    sequence = "MACDEFGHIKLMNPQRSTVWY"
    (root / "reference.fasta").write_text(
        f">testis\n{sequence}\n>heart\n{sequence}\n"
        ">background\nACDEFGHIKAAAAA\n>il\nMACDEFGHLKLMNPQRSTVWY\n"
    )
    reference = {
        "assembly_accession": "fixture-assembly",
        "annotation_release": "fixture-annotation",
        "source_version": "fixture-1",
        "taxon": 9615,
        "asset_hashes": {"protein": file_digest(root / "reference.fasta")["sha256"]},
    }
    key = digest(reference)
    occurrences = [
        {
            "occurrence_id": name,
            "protein_id": name,
            "gene_id": "gene:" + name,
            "transcript_id": "tx:" + name,
            "sequence": sequence,
            "reference_key": key,
            "source": "fixture-annotation",
            "complete": True,
        }
        for name in ("testis", "heart")
    ]
    samples = [
        {
            "sample_id": name,
            "study": "study",
            "donor": "donor:" + name,
            "specimen": "specimen:" + name,
            "library": "library:" + name,
            "source": "fixture",
            "reference_key": key,
            "assay": "RNA-seq",
            "quantification": "quantifier-1",
            "unit": "TPM",
            "tissue": name,
            "taxon": 9615,
            "health": "healthy",
        }
        for name in ("testis", "heart", "brain")
    ]
    contributions = [
        {
            "sample_id": name,
            "allocation_id": "allocation:" + name,
            "occurrence_ids": [name],
            "lower": value,
            "upper": value,
            "scope": "transcript",
            "direct": True,
            "coding_assignment": True,
            "provenance": ["record:" + name],
        }
        for name, value in (("testis", 10.0), ("heart", 3.0))
    ]
    write_json(
        root / "expression.json",
        {
            "schema_version": 1,
            "reference": reference,
            "occurrences": occurrences,
            "samples": samples,
            "contributions": contributions,
        },
    )
    write_json(
        root / "candidates.json",
        [
            {
                "sequence_id": sequence_id(sequence),
                "admission_basis": "species_expression",
                "policy_id": "fixture-policy-1",
                "support": [{"sample_id": "testis", "allocation_id": "allocation:testis"}],
            }
        ],
    )
    observations = pd.DataFrame(
        [
            {
                "peptide": pep,
                "provenance_id": "prov:" + str(i),
                "pmid": 123,
                "source": "supplement",
                "qualitative_measurement": "Positive",
                "assay_method": method,
                "response_measured": "ligand presentation",
                "species": "Homo sapiens" if i == 1 else "Canis lupus familiaris",
                "host": "Homo sapiens" if i == 1 else "Canis lupus familiaris",
                "presenting_species": "Homo sapiens" if i == 1 else "Canis lupus familiaris",
                "mhc_species": "Canis sp.",
                "source_taxon": 9606 if i == 1 else 9615,
                "presenting_taxon": 9606 if i == 1 else 9615,
                "mhc_taxon": 9615,
                "mhc_restriction": "DLA-88*003:02" if i == 1 else "",
                "condition_id": "human" if i == 1 else "dog",
                "sample_attribution": "curated_sample_label",
                "attributed_sample_label": "human" if i == 1 else "dog",
            }
            for i, (pep, method) in enumerate(
                [
                    ("ACDEFGHIK", "mass spectrometry"),
                    ("LMNPQRSTV", "mass spectrometry"),
                    ("ACDEFGHIK", "fluorescence"),
                ]
            )
        ]
    )
    observations.to_parquet(root / "observations.parquet", index=False)
    contributors = pd.DataFrame(
        [
            {
                "provenance_id": "prov:" + str(i),
                "source_record_id": "record:" + str(i),
                "source_dataset": "fixture",
                "source_row": i + 1,
                "original_pmid": "123",
                "original_fields": json.dumps(
                    {"peptide": row.peptide, "il_ambiguity": "unresolved"}
                ),
                "source_row_values": json.dumps([row.peptide]),
                "attributed_sample_label": row.attributed_sample_label,
                "attribution_evidence": "curated_peptide_map",
                "relationships": '["retained"]',
                "relationship_status": "source_record",
            }
            for i, row in observations.iterrows()
        ]
    )
    contributors.to_parquet(root / "contributors.parquet", index=False)
    write_json(root / "lineage.json", {"schema_version": 1, "entities": [], "contexts": []})
    normals = pd.DataFrame(
        [
            {
                "peptide": "ACDEFGHIK",
                "donor_id": donor,
                "donor_status": "resolved",
                "source_tissue": tissue,
                "tissue_status": status,
                "is_cell_line": False,
                "source_record_id": "normal:" + str(i),
                "source_taxon": 9615,
                "assay_method": "mass spectrometry",
                "response_measured": "ligand presentation",
                "qualitative_measurement": "Positive",
                "source": "fixture",
                "mhc_restriction": allele,
            }
            for i, (donor, tissue, status, allele) in enumerate(
                [
                    ("d1", "heart", "nonmalignant", "DLA-88*003:02"),
                    ("d1", "brain", "nonmalignant", "DLA-88*012:01"),
                    ("d2", "lung", "nonmalignant", ""),
                    ("d3", "heart", "malignant", ""),
                ]
            )
        ]
    )
    normals.to_parquet(root / "normal.parquet", index=False)
    manifest = {
        "schema_version": 1,
        "reference": reference,
        "reference_complete": True,
        "reference_scope": "Complete synthetic test reference; no biological claim",
        "files": {
            name: {"path": filename, **file_digest(root / filename)}
            for name, filename in {
                "reference": "reference.fasta",
                "expression": "expression.json",
                "candidates": "candidates.json",
                "observations": "observations.parquet",
                "contributors": "contributors.parquet",
                "lineage": "lineage.json",
                "normal": "normal.parquet",
            }.items()
        },
        "peptides": ["CDEFGH", "ACDEFGHLK"],
        "include_il_equivalent": True,
        "policy": {
            "policy_id": "fixture-policy-1",
            "version": 1,
            "allowed_tissues": ["testis"],
            "required_normal_tissues": ["heart", "brain"],
            "unit": "TPM",
            "allowed_min": 5,
            "normal_max": 0.5,
            "min_normal_donors": 1,
            "tissue_blacklist": {
                "version": 1,
                "tissue_status": "nonmalignant",
                "min_donors": 2,
                "tissue_groups": {"heart": ["heart"], "brain": ["brain"], "lung": ["lung"]},
            },
        },
        "normal_source": {
            "taxon": 9615,
            "source": "synthetic-fixture",
            "version": "1",
            "coverage": "scoped",
            "coverage_note": "Synthetic records for software tests only",
        },
    }
    write_json(root / "species-evidence.json", manifest)
    return root


def update(inputs, name, transform):
    manifest = json.loads((inputs / "species-evidence.json").read_text())
    if name == "manifest":
        transform(manifest)
    else:
        path = inputs / manifest["files"][name]["path"]
        value = pd.read_parquet(path) if path.suffix == ".parquet" else json.loads(path.read_text())
        transform(value)
        if path.suffix == ".parquet":
            value.to_parquet(path, index=False)
        else:
            write_json(path, value)
        manifest["files"][name].update(file_digest(path))
    write_json(inputs / "species-evidence.json", manifest)


def bundle(inputs, tmp_path, **kwargs):
    from hitlist import write_species_evidence_bundle

    output = tmp_path / "bundle"
    write_species_evidence_bundle(output, input_manifest=inputs / "species-evidence.json", **kwargs)
    return output


def test_complete_reference_counterevidence_and_exact_ms(inputs, tmp_path):
    out = bundle(inputs, tmp_path)
    peptides = pd.read_parquet(out / "peptides.parquet").set_index("peptide")
    assert peptides.loc["ACDEFGHIK", "n_exact_proteins"] == 3
    assert not peptides.loc["ACDEFGHIK", "candidate_specific_in_reference"]
    assert peptides.loc["ACDEFGHIK", "n_native_ms_observations"] == 1
    assert peptides.loc["ACDEFGHIK", "blacklisted"]
    assert peptides.loc["ACDEFGHIK", "normal_expression_counterevidence"]
    assert peptides.loc["LMNPQRSTV", "n_native_ms_observations"] == 0
    assert peptides.loc["LMNPQRSTV", "n_heterologous_ms_observations"] == 1
    assert peptides.loc["CDEFGH", "n_exact_ms_observations"] == 0
    assert peptides.loc["ACDEFGHLK", "n_exact_ms_observations"] == 0
    mappings = pd.read_parquet(out / "mappings.parquet")
    assert set(mappings.match_kind) == {"exact", "il_equivalent"}
    assert "background" in set(
        mappings.loc[mappings.annotation_status.eq("unresolved"), "protein_id"]
    )
    groups = json.loads((out / "group_expression.json").read_text())
    assert groups[0]["normal_counterevidence"]
    assert groups[0]["missing_normal_tissues"] == ["brain", "heart"]
    risk = pd.read_parquet(out / "tissue_risk.parquet")
    assert risk.n_donors.tolist() == [2]
    excluded = pd.read_parquet(out / "excluded_observations.parquet")
    assert len(excluded) == 1 and excluded.exclusion_reason.tolist() == ["not_positive_ms"]


def test_offline_replay_and_deterministic_output(inputs, tmp_path, monkeypatch):
    from hitlist import verify_evidence_bundle, write_species_evidence_bundle

    out = bundle(inputs, tmp_path)
    other = tmp_path / "other"
    write_species_evidence_bundle(other, input_manifest=inputs / "species-evidence.json")
    assert {p.name: p.read_bytes() for p in out.rglob("*") if p.is_file()} == {
        p.name: p.read_bytes() for p in other.rglob("*") if p.is_file()
    }
    shutil.rmtree(inputs)

    def fail(*args, **kwargs):
        raise AssertionError("Live data must not be consulted")

    monkeypatch.setattr("hitlist.lineage.load_lineage", fail)
    monkeypatch.setattr("hitlist.tissue_blacklist.load_atlas_tissue_evidence", fail)
    monkeypatch.setattr("requests.sessions.Session.request", fail)
    assert verify_evidence_bundle(out)["kind"] == "species_expression"


@pytest.mark.parametrize(
    "name,change,match",
    [
        ("expression", lambda d: d["samples"][0].update(reference_key="wrong"), "reference"),
        ("expression", lambda d: d["samples"][0].update(taxon=9606), "taxon"),
        ("manifest", lambda d: d["normal_source"].update(taxon=9606), "taxon"),
        ("candidates", lambda d: d[0].update(admission_basis="human_orthology"), "admission"),
        ("candidates", lambda d: d[0].update(sequence_id="0" * 64), "sequence"),
        (
            "expression",
            lambda d: d["contributions"].append({**d["contributions"][0], "upper": 99}),
            "allocation",
        ),
    ],
)
def test_rejects_contradictory_inputs(inputs, tmp_path, name, change, match):
    update(inputs, name, change)
    with pytest.raises(ValueError, match=match):
        bundle(inputs, tmp_path)
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize("scope", ["gene", "promoter"])
def test_gene_measurement_does_not_become_isoform_abundance(inputs, tmp_path, scope):
    update(inputs, "expression", lambda d: d["contributions"][0].update(scope=scope))
    out = bundle(inputs, tmp_path)
    audit = json.loads((out / "expression_audit.json").read_text())
    testis = next(r for r in audit if r["sample_id"] == "testis")
    assert testis["scope"] == scope and testis["lower"] == 0 and testis["ambiguous"]
    group = json.loads((out / "group_expression.json").read_text())[0]
    assert group["isoform_support_unresolved"]
    assert group["normal_counterevidence"]


def test_missing_measurements_and_units_are_not_zero_or_tpm(inputs, tmp_path):
    update(inputs, "expression", lambda d: d["samples"][0].update(unit="counts"))
    out = bundle(inputs, tmp_path)
    audit = json.loads((out / "expression_audit.json").read_text())
    assert next(r for r in audit if r["sample_id"] == "testis")["unit_matches_policy"] is False
    group = json.loads((out / "group_expression.json").read_text())[0]
    assert group["isoform_support_unresolved"]
    assert "brain" in group["missing_normal_tissues"]
    assert all(not r["restricted_in_declared_panel"] for r in group["namespaces"])


def test_ambiguous_allocations_retain_all_coding_alternatives(inputs, tmp_path):
    def change(d):
        extra = {
            **d["occurrences"][0],
            "occurrence_id": "il",
            "protein_id": "il",
            "transcript_id": "tx:il",
            "gene_id": "gene:il",
            "sequence": "MACDEFGHLKLMNPQRSTVWY",
        }
        d["occurrences"].append(extra)
        d["contributions"][0]["occurrence_ids"].append("il")

    update(inputs, "expression", change)
    out = bundle(inputs, tmp_path)
    audit = json.loads((out / "expression_audit.json").read_text())
    rows = [r for r in audit if r["sample_id"] == "testis"]
    assert len(rows) == 2
    assert all(r["lower"] == 0 and r["ambiguous"] for r in rows)
    assert all(r["occurrence_ids"] == ["il", "testis"] for r in rows)


def test_explicit_empty_canine_normal_snapshot(inputs, tmp_path):
    update(inputs, "normal", lambda d: d.drop(d.index, inplace=True))
    update(inputs, "manifest", lambda d: d["normal_source"].update(coverage="missing"))
    out = bundle(inputs, tmp_path)
    peptides = pd.read_parquet(out / "peptides.parquet")
    assert peptides.normal_ms_coverage.eq("missing").all()
    assert not peptides.blacklisted.any()
    assert (out / "forbidden_sequences.txt").read_text() == ""


def test_donors_not_alleles_or_tumors_establish_blacklist(inputs, tmp_path):
    def change(frame):
        frame.loc[frame.donor_id.eq("d2"), "donor_status"] = "unknown"

    update(inputs, "normal", change)
    out = bundle(inputs, tmp_path)
    risk = pd.read_parquet(out / "tissue_risk.parquet")
    assert risk.n_donors.tolist() == [1]
    assert not risk.blacklisted.any()
    audit = pd.read_parquet(out / "tissue_evidence.parquet")
    assert set(audit.exclusion_reason) == {"", "unresolved_donor", "not_nonmalignant_tissue"}


def test_contributor_coverage_required(inputs, tmp_path):
    update(inputs, "contributors", lambda d: d.drop(d.index[0], inplace=True))
    with pytest.raises(ValueError, match="contributor coverage"):
        bundle(inputs, tmp_path)


@pytest.mark.parametrize(
    "limit",
    [
        "max_input_bytes",
        "max_json_bytes",
        "max_table_bytes",
        "max_records",
        "max_reference_bytes",
        "max_protein_residues",
        "max_total_residues",
        "max_peptides",
        "max_mapping_rows",
        "max_output_bytes",
    ],
)
def test_resource_bounds_publish_nothing(inputs, tmp_path, limit):
    with pytest.raises(ValueError, match="exceed"):
        bundle(inputs, tmp_path, limits={limit: 1})
    assert not (tmp_path / "bundle").exists()
    assert not list(tmp_path.glob(".species-evidence-*"))


def test_replay_detects_semantic_tampering_even_with_updated_hash(inputs, tmp_path):
    from hitlist import verify_evidence_bundle

    out = bundle(inputs, tmp_path)
    path = out / "peptides.parquet"
    peptides = pd.read_parquet(path)
    peptides["n_native_ms_observations"] = 123
    peptides.to_parquet(path, index=False)
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["files"][path.name] = file_digest(path)
    write_json(out / "manifest.json", manifest)
    with pytest.raises(ValueError, match="peptides"):
        verify_evidence_bundle(out)


def test_reference_checksum_and_existing_destination(inputs, tmp_path):
    from hitlist import write_species_evidence_bundle

    out = bundle(inputs, tmp_path)
    with pytest.raises(FileExistsError):
        write_species_evidence_bundle(out, input_manifest=inputs / "species-evidence.json")
    (inputs / "reference.fasta").write_text(">changed\nACDEFGHIK\n")
    with pytest.raises(ValueError, match="checksum"):
        write_species_evidence_bundle(
            tmp_path / "changed", input_manifest=inputs / "species-evidence.json"
        )


def test_measured_testis_with_no_normal_measurements_is_unresolved(inputs, tmp_path):
    def change(data):
        data["contributions"] = data["contributions"][:1]
        data["contributions"][0]["occurrence_ids"] = ["heart", "testis"]

    update(inputs, "expression", change)
    out = bundle(inputs, tmp_path)
    peptides = pd.read_parquet(out / "peptides.parquet").set_index("peptide")
    assert peptides.loc["LMNPQRSTV", "normal_expression_unresolved"]
    group = json.loads((out / "group_expression.json").read_text())[0]
    assert group["missing_normal_tissues"] == ["brain", "heart"]


def test_resolved_normal_panel_can_pass_without_becoming_a_safety_claim(inputs, tmp_path):
    def change(data):
        for row in data["contributions"]:
            row["occurrence_ids"] = ["heart", "testis"]
        data["contributions"][1].update(lower=0.0, upper=0.0)
        data["contributions"].append(
            {**data["contributions"][1], "sample_id": "brain", "allocation_id": "allocation:brain"}
        )

    update(inputs, "expression", change)
    out = bundle(inputs, tmp_path)
    group = json.loads((out / "group_expression.json").read_text())[0]
    assert group["missing_normal_tissues"] == []
    assert not group["normal_counterevidence"]
    assert group["namespaces"][0]["restricted_in_declared_panel"]
    # The separate empirical MS exclusion remains, regardless of RNA screening.
    peptides = pd.read_parquet(out / "peptides.parquet").set_index("peptide")
    assert peptides.loc["ACDEFGHIK", "blacklisted"]


def test_lineage_is_frozen_and_unassigned_restrictions_stay_unassigned(inputs, tmp_path):
    def change(data):
        data["entities"] = [
            {
                "identity_id": "dog-donor-1",
                "kind": "donor",
                "namespace": "fixture",
                "reported_id": "dog-1",
                "evidence_reference": "fixture:paper",
                "evidence_note": "reviewed donor",
            }
        ]
        data["contexts"] = [
            {
                "pmid": 123,
                "condition_id": "dog",
                "evidence_reference": "fixture:paper",
                "donor": {"status": "resolved", "ids": ["dog-donor-1"]},
            }
        ]

    update(inputs, "lineage", change)
    out = bundle(inputs, tmp_path)
    presentation = pd.read_parquet(out / "presentation.parquet")
    dog = presentation[presentation.presentation_context.eq("native")]
    assert dog.mhc_restriction.eq("").all()
    assert dog.donor_ids.tolist() == ['["dog-donor-1"]']
    human = presentation[presentation.presentation_context.eq("heterologous_mhc")]
    assert human.donor_status.eq("unknown").all()


def test_conflicting_normal_source_identity_cannot_invent_donors(inputs, tmp_path):
    def change(frame):
        frame.loc[2, "source_record_id"] = frame.loc[0, "source_record_id"]

    update(inputs, "normal", change)
    with pytest.raises(ValueError, match="unique source rows"):
        bundle(inputs, tmp_path)


def test_occurrence_sequence_must_match_full_reference(inputs, tmp_path):
    update(
        inputs, "expression", lambda d: d["occurrences"][1].update(sequence="MACDEFGHLKLMNPQRSTVWY")
    )
    with pytest.raises(ValueError, match="disagrees with reference"):
        bundle(inputs, tmp_path)


def test_contributor_identity_cannot_be_swapped_between_peptides(inputs, tmp_path):
    def change(frame):
        frame.loc[[0, 1], "provenance_id"] = ["prov:1", "prov:0"]

    update(inputs, "contributors", change)
    with pytest.raises(ValueError, match="Contributor peptide contradicts"):
        bundle(inputs, tmp_path)
