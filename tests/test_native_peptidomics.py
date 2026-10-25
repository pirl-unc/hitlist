"""Software fixtures for native-source fidelity, never biological training data."""

import csv
import json
from datetime import datetime

import openpyxl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hitlist import native_peptidomics as source
from hitlist.evidence_package import package_family
from hitlist.evidence_store import EvidenceStore
from hitlist.material_context import links_with_minimum_precision
from hitlist.native_peptidomics_reference import candidate_spans
from hitlist.provenance import file_digest


def write_tsv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def original(tmp_path, monkeypatch):
    assets = tmp_path / "assets"
    assets.mkdir()
    contract = json.loads(source.CONTRACT.read_text())
    acquisitions = contract["acquisitions"]
    samples = [a["sample_label"] for a in acquisitions]
    for name in contract["metadata_json"]:
        (assets / name).write_text('{"role": "synthetic_software_fixture"}')
    (assets / "sahu-article.xml").write_text("<article><p>Synthetic source fixture</p></article>")
    (assets / contract["mqpar"]).write_text(
        "<MaxQuantParams><filePaths>"
        + "".join(f"<string>C:\\fixture\\{a['raw_file']}.raw</string>" for a in acquisitions)
        + "</filePaths><experiments>"
        + "".join(f"<string>{s}</string>" for s in samples)
        + "</experiments><matchBetweenRuns>true</matchBetweenRuns></MaxQuantParams>"
    )
    write_tsv(
        assets / contract["design"],
        [{"Name": a["raw_file"], "Experiment": a["sample_label"]} for a in acquisitions],
    )
    write_tsv(
        assets / "summary.txt",
        [{"Raw file": a["raw_file"], "Experiment": a["sample_label"]} for a in acquisitions],
    )
    write_tsv(
        assets / "parameters.txt",
        [
            {"Parameter": "Repeated parameter", "Value": "1"},
            {"Parameter": "Repeated parameter", "Value": "2"},
        ],
    )
    (assets / contract["reference"]).write_text(
        ">sp|R1|SOURCE1_HUMAN OX=9606\nMACDEFGHIKLMNPQRST\n"
        ">sp|R2|SOURCE2_HUMAN OX=9606\nQACDEFGHIKACDEFGHIK\n"
        ">sp|P22222|NS3CCNB1_HUMAN synthetic software fixture OX=9606\nWACDEFGHIKW\n"
        ">sp|R3|SOURCE3_HUMAN OX=9606\nQFGHLKLMNPR\n"
    )
    common = {"Reverse": "", "Potential contaminant": "", "PEP": "0.001"}

    def intensities(value):
        return {
            "Intensity": str(value),
            **{"Intensity " + s: str(value) if s == "Cont1" else "0" for s in samples},
        }

    peptides = []
    for native, sequence, parents, evidence, forms, value, gene in [
        ("0", "ACDEFGHIK", "R1;CON__C", "0;1", "0;1", 31, "SEPT8"),
        ("1", "FGHIKLMNP", "R1", "2", "2", 10, "GENE2"),
        ("2", "EEEEEEEE", "CON__C", "", "3", 0, "GENE3"),
    ]:
        peptides.append(
            {
                **common,
                "id": native,
                "Sequence": sequence,
                "Proteins": parents,
                "Evidence IDs": evidence,
                "Mod. peptide IDs": forms,
                "Gene names": gene,
                **intensities(value),
                **{"LFQ intensity " + s: str(2 * value) if s == "Cont1" else "0" for s in samples},
            }
        )
    peptides[2]["Potential contaminant"] = "+"
    peptides[2]["Intensity KD3"] = ""
    write_tsv(assets / "peptides.txt", peptides)
    forms = []
    for native, peptide_native, value, modification, evidence in [
        ("0", "0", 10, "Unmodified", "0"),
        ("1", "0", 20, "Acetyl (Protein N-term)", "1"),
        ("2", "1", 10, "Unmodified", "2"),
        ("3", "2", 0, "Unmodified", ""),
    ]:
        peptide = peptides[int(peptide_native)]
        forms.append(
            {
                **common,
                "id": native,
                "Peptide ID": peptide_native,
                "Sequence": peptide["Sequence"],
                "Proteins": peptide["Proteins"],
                "Modifications": modification,
                "Evidence IDs": evidence,
                **intensities(value),
            }
        )
    write_tsv(assets / "modificationSpecificPeptides.txt", forms)
    features = []
    for i, form in enumerate(forms[:3]):
        features.append(
            {
                **common,
                "id": str(i),
                "Peptide ID": form["Peptide ID"],
                "Mod. peptide ID": form["id"],
                "Sequence": form["Sequence"],
                "Proteins": form["Proteins"],
                "Experiment": "Cont1",
                "Raw file": acquisitions[0]["raw_file"],
                "Modifications": form["Modifications"],
                "Modified sequence": "_" + form["Sequence"] + "_",
                "Type": "MULTI-MATCH" if i == 1 else "MSMS",
                "Charge": "2",
                "Intensity": form["Intensity"],
            }
        )
    write_tsv(assets / "evidence.txt", features)
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(list(peptides[0]))
    for row in peptides:
        sheet.append(list(row.values()))
    gene_column = list(peptides[0]).index("Gene names") + 1
    sheet.cell(2, gene_column, datetime(2008, 9, 1)).number_format = "mmm-yy"
    workbook.save(assets / contract["publisher_workbook"])
    contract["source_registry"] = {"PXD018722": {"source_role": "synthetic_software_fixture"}}
    contract["assets"] = [
        {
            "filename": p.name,
            "source_url": "https://example.invalid/" + p.name,
            "role": "synthetic_software_fixture",
            "license_status": "synthetic_fixture",
            **file_digest(p),
        }
        for p in sorted(assets.iterdir())
    ]
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract))
    monkeypatch.setattr(source, "CONTRACT", contract_path)
    return assets, tmp_path / "package", contract


def build(original):
    assets, package, _ = original
    source.write_native_peptidomics_package(assets, package)
    return EvidenceStore.open(package)


def query(store, name, **kwargs):
    return store.query("native_" + name, limit=None, **kwargs).table.to_pylist()


def test_quantity_grains_raw_lfq_matching_and_missing_are_separate(original):
    store = build(original)
    assert package_family(store.manifest) == "native_peptidomics"
    quantities = query(
        store,
        "quantities",
        joins=["peptide", "acquisition", "form", "identification"],
        select=[
            "observation_kind",
            "quantity_kind",
            "quantitative_value",
            "value_status",
            "identification_type",
            "measurement_unit",
            "peptide.native_id",
            "acquisition.sample_label_raw",
            "form.native_id",
            "identification.native_id",
        ],
    )
    assert len(quantities) == 3 * 13 + 4 * 7 + 3
    first = [
        r
        for r in quantities
        if r["peptide.native_id"] == "0" and r["acquisition.sample_label_raw"] == "Cont1"
    ]
    assert sorted(r["quantitative_value"] for r in first) == [10, 10, 20, 20, 31, 62]
    raw = next(
        r
        for r in first
        if r["observation_kind"] == "peptide_sequence" and r["quantity_kind"] == "source_intensity"
    )
    assert raw["quantitative_value"] == 31 and raw["form.native_id"] is None
    matched = next(r for r in first if r["identification.native_id"] == "1")
    assert matched["identification_type"] == "MULTI-MATCH"
    assert matched["form.native_id"] == "1" and matched["quantitative_value"] == 20
    assert all(r["measurement_unit"] == "source_signal_arbitrary_units" for r in quantities)
    assert any(r["quantitative_value"] == 0 and r["value_status"] == "zero" for r in quantities)
    assert any(
        r["quantitative_value"] is None and r["value_status"] == "missing" for r in quantities
    )
    residuals = query(
        store, "reconciliations", where={"assertion_predicate": "forms_intensity_sum"}
    )
    assert any(json.loads(r["details_json"])["residual"] == "-1" for r in residuals)


def test_all_origins_and_unresolved_parents_enter_dependencies(original):
    store = build(original)
    origins = query(
        store,
        "origins",
        joins=["peptide", "reference"],
        select=[
            "peptide.native_id",
            "reference.accession_raw",
            "origin_match_type",
            "candidate_start_1based",
            "reference_role",
            "reference.organism_taxon_id",
        ],
    )
    assert len(origins) == 6
    assert len([r for r in origins if r["reference.accession_raw"] == "R2"]) == 2
    assert {r["reference.accession_raw"] for r in origins if r["peptide.native_id"] == "0"} == {
        "R1",
        "R2",
        "P22222",
    }
    reporter = next(r for r in origins if r["reference.accession_raw"] == "P22222")
    assert reporter["reference.organism_taxon_id"] == "9606"
    assert reporter["reference_role"] == "engineered_construct"
    equivalent = next(r for r in origins if r["reference.accession_raw"] == "R3")
    assert equivalent["origin_match_type"] == "isoleucine_leucine_equivalent"
    parents = query(
        store,
        "declared_parents",
        joins=["reference"],
        select=["parent_accession", "reference.accession_raw"],
    )
    assert any(
        r["parent_accession"] == "CON__C" and r["reference.accession_raw"] is None for r in parents
    )
    groups = query(store, "dependency_groups")
    # The unmapped contaminant shares a declared parent with peptide 0. It may
    # later be excluded, but must not disappear before leakage groups are built.
    assert len(groups) == 3 and len({r["dependency_group_id"] for r in groups}) == 1


def test_source_cell_gene_corruption_and_unknown_material_context(original):
    store = build(original)
    comparisons = query(
        store,
        "reconciliations",
        where={"assertion_status": "date_typed_gene_annotation_original_text_retained"},
    )
    assert len(comparisons) == 1
    details = json.loads(comparisons[0]["details_json"])
    assert details["original_text_value"] == "SEPT8"
    assert details["intended_gene_not_inferred_from_date"] is True
    assert float(details["publisher_value"]) > 39000
    peptides = query(store, "peptides", where={"native_id": "0"})
    assert peptides[0]["gene_names_raw"] == "SEPT8"
    assert any(r["cell_style_raw"] for r in query(store, "source_cells"))
    contexts = query(store, "material_contexts")
    assert len(contexts) == 9
    assert all(r["specimen_id"] is None and r["aliquot_id"] is None for r in contexts)
    assert links_with_minimum_precision(query(store, "material_links"), "exact_specimen") == ()
    acquisitions = query(store, "acquisitions")
    assert len(acquisitions) == 6
    assert all("unresolved" in r["replicate_status"] for r in acquisitions)
    records = query(
        store, "source_records", joins=["asset"], select=["source_file", "asset.source_file"]
    )
    assert all(r["source_file"] == r["asset.source_file"] for r in records)
    assert (
        store.catalog()["source_registry"]["PXD018722"]["source_role"]
        == "synthetic_software_fixture"
    )


def test_replay_rejects_typed_tampering_with_fresh_file_hashes(original):
    store = build(original)
    path = store.directory / source.ROOT / "typed/quantities.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[0]["quantitative_value"] += 1
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    manifest = json.loads((store.directory / "manifest.json").read_text())
    manifest["files"] = source._inventory(store.directory)
    (store.directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source replay differs"):
        EvidenceStore.open(store.directory)


def test_pinned_sources_no_overwrite_and_retained_failure(original):
    assets, package, _ = original
    build(original)
    with pytest.raises(FileExistsError):
        source.write_native_peptidomics_package(assets, package)
    (assets / "peptides.txt").write_text("changed source")
    with pytest.raises(ValueError, match="original source differs"):
        source.write_native_peptidomics_package(assets, package.parent / "changed")
    assert list(package.parent.glob(".changed-*/failure-context.json"))


@pytest.mark.parametrize("defect", ["duplicate_id", "unresolved_form", "wrong_acquisition"])
def test_invalid_source_identity_fails_even_with_revised_fixture_contract(original, defect):
    assets, package, contract = original
    path = assets / ("peptides.txt" if defect == "duplicate_id" else "evidence.txt")
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if defect == "duplicate_id":
        rows[1]["id"] = rows[0]["id"]
    elif defect == "unresolved_form":
        rows[0]["Mod. peptide ID"] = "absent"
    else:
        rows[0]["Raw file"] = "unrelated_run"
    write_tsv(path, rows)
    next(a for a in contract["assets"] if a["filename"] == path.name).update(file_digest(path))
    source.CONTRACT.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match=r"IDs|acquisition identity"):
        source.write_native_peptidomics_package(assets, package)


def test_prefix_enumeration_matches_independent_overlapping_brute_force():
    references = [
        {"reference_sequence": "ILILILACDIL", "id": "a"},
        {"reference_sequence": "ACDACD", "id": "b"},
    ]
    peptides = [
        {"peptide_sequence": seq, "id": str(i)}
        for i, seq in enumerate(["I", "IL", "LI", "ILIL", "ACD", "DIL", "IL", "ZZZ"])
    ]
    observed = {
        (p["id"], r["id"], start, end, kind)
        for p, r, start, end, kind in candidate_spans(references, peptides)
    }
    expected = set()
    for p in peptides:
        for r in references:
            for start in range(len(r["reference_sequence"]) - len(p["peptide_sequence"]) + 1):
                span = r["reference_sequence"][start : start + len(p["peptide_sequence"])]
                if span.replace("L", "I") == p["peptide_sequence"].replace("L", "I"):
                    kind = (
                        "literal"
                        if span == p["peptide_sequence"]
                        else "isoleucine_leucine_equivalent"
                    )
                    expected.add((p["id"], r["id"], start + 1, start + len(span), kind))
    assert observed == expected
