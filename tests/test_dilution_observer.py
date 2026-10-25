"""Small source-corruption and public-query fixtures, never biological evidence."""

import csv
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hitlist import dilution_observer as source
from hitlist.dilution_observer_compiler import comparison, number
from hitlist.evidence_package import package_family
from hitlist.evidence_store import EvidenceStore
from hitlist.material_context import links_with_minimum_precision
from hitlist.provenance import file_digest


def write_tsv(path, header, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=header, delimiter="\t")
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in header} for row in rows)


@pytest.fixture
def original(tmp_path, monkeypatch):
    assets = tmp_path / "assets"
    assets.mkdir()
    contract = json.loads(source.CONTRACT.read_text())
    form = "ISL(UniMod:695)GEHEGGGK"
    label = "ISL(Label:13C(6)15N(1))GEHEGGGK_2"
    run1 = "fixture_No1_Dilution1_02fmol-ul_DIA#1_fixture"
    run2 = "fixture_No2_Dilution1-2_01fmol-ul_DIA#2_fixture"
    project = {
        "accession": "PXD024809",
        "title": "Synthetic software fixture",
        "projectDescription": "JY HLA-I matrix (HLA-A*02:01, HLA-B*07:02, HLA-C*07:02)",
        "sampleProcessingProtocol": "Post-isolation panel IS[L(13C6; 15N)]GEHEGGGK, VYAET[L(13C6; 15N)]SGFIK. Twofold 1:2. DIA1 and DIA2 differ.",
        "dataProcessingProtocol": "Reference defaults reported; actual overrides unqualified.",
        "organisms": [{"accession": "NEWT:9606"}],
        "instruments": [{"name": "fixture instrument"}],
    }
    (assets / contract["project"]).write_text(json.dumps(project))
    (assets / contract["inventory"]).write_text(
        json.dumps([{"fileName": r.replace("#", "_") + ".raw"} for r in (run1, run2)])
    )
    (assets / contract["article"]).write_text(
        '<article><body><p>Results report 1:10.</p><table-wrap id="T1"><table><tbody>'
        "<tr><td>1</td><td>No. 1</td><td>0.2\u2009fmol/µL</td></tr>"
        "<tr><td>1:2</td><td>No. 2</td><td>0.1 fmol/µL</td></tr>"
        "</tbody></table></table-wrap></body></article>"
    )
    library = []
    for group, charge, decoy, sequence, full, group_label in [
        ("77", "2", "0", "ISLGEHEGGGK", form, label),
        ("78", "3", "0", "ISLGEHEGGGK", form, "NA"),
        ("79", "3", "0", "ISLGEHEGGGK", form, "NA"),
        ("80", "2", "1", "ISLGEHEGGGK", form, "DECOY"),
        ("81", "2", "0", "AAAAGRKTL", "AAAAGRKTL", "NA"),
    ]:
        library.append(
            {
                "TransitionGroupId": group,
                "PrecursorCharge": charge,
                "Decoy": decoy,
                "PeptideSequence": sequence,
                "ModifiedPeptideSequence": full,
                "PeptideGroupLabel": group_label,
                "TransitionId": "transition-" + group,
                "ProductMz": "123.5",
                "ProductCharge": "1",
                "LibraryIntensity": "42",
                "PrecursorMz": "234.5",
            }
        )
    write_tsv(assets / contract["library"], contract["headers"]["master_query_library"], library)

    def scored(run, feature, precursor="1", charge="2", decoy="0", full=form):
        return {
            "filename": run + ".mzML",
            "run_id": "1804006131325186151" if run == run1 else "1804006131325186152",
            "id": feature,
            "transition_group_id": precursor,
            "Charge": charge,
            "decoy": decoy,
            "Sequence": "ISLGEHEGGGK" if full == form else "AAAAGRKTL",
            "FullPeptideName": full,
            "Intensity": "12.50",
            "RT": "10",
            "leftWidth": "8.5",
            "rightWidth": "11.5",
            "peak_group_rank": "1",
            "m_score": "0.001",
            "d_score": "2",
            "aggr_Peak_Area": "1;2",
        }

    scores = [
        scored(run1, "9007199254740993"),
        scored(run1, "9007199254740992", "2", "3"),
        scored(run1, "1234567890123456789"),
        scored(run1, "1234567890123456789", "4", full="AAAAGRKTL"),
        scored(run1, "1234567890123456780", decoy="1"),
    ]
    name1, name2 = (r.replace("#", "_") + ".mzML.tsv" for r in (run1, run2))
    write_tsv(assets / name1, contract["headers"]["per_run_scored_candidates"], scores)
    write_tsv(
        assets / name2,
        contract["headers"]["per_run_scored_candidates"],
        [scored(run2, "99", "4", full="AAAAGRKTL")],
    )
    aligned = []
    for index, (run, feature, intensity, precursor, charge, full, group_label) in enumerate(
        [
            (run1, "9007199254740993", "12.5", "1", "2", form, label),
            (run1, "9007199254740992", "12.5", "2", "3", form, ""),
            (run1, "NA", "0", "3", "2", form, label),
            (run1, "NA", "NA", "5", "2", form, label),
            (run1, "777", "8", "6", "2", form, label),
            (run1, "1234567890123456789", "12.5", "7", "2", form, label),
            (run1, "1234567890123456780", "12.5", "1", "2", form, label),
            (run2, "NA", "3", "3", "2", form, label),
            (run2, "99", "12.5", "4", "2", "AAAAGRKTL", ""),
        ]
    ):
        aligned.append(
            {
                "peptide_id": str(index),
                "precursor": precursor,
                "run": run,
                "RT": "10",
                "intensity": intensity,
                "leftWidth": "8.5",
                "rightWidth": "11.5",
                "peak_group_rank": "NA" if feature == "NA" else "1",
                "m_score": "NA" if feature == "NA" else "1e-3",
                "alignment_rank": "1",
                "feature_id": feature,
                "sequence": full,
                "charge": charge,
                "group_label": group_label,
            }
        )
    write_tsv(assets / contract["aligned"], contract["headers"]["aligned_quantities"], aligned)
    roles = {a["filename"]: a["role"] for a in contract["assets"]}
    roles.update({name1: "per_run_scored_candidates", name2: "per_run_scored_candidates"})
    for name in ("nfcore-main-1.2.2.nf", "DIAlignR-1.2.2.R"):
        (assets / name).write_text("pyprophet export --max_rs_peakgroup_qvalue=${parameter}\n")
    contract.update(
        n_nominal_standards=2,
        n_nominal_levels=2,
        n_scored_assets=2,
        source_registry={"PXD024809": {"source_role": "synthetic_software_fixture"}},
    )
    contract["protocol_assertions"] = [
        {
            "predicate": "reported_DIA_method_difference",
            "value_raw": "different",
            "unit": None,
            "evidence_text": "DIA1 and DIA2 differ.",
            "subject": "project",
            "project_field": "sampleProcessingProtocol",
            "applicability": "synthetic_software_fixture_protocol",
        }
    ]
    contract["assets"] = [
        {
            "filename": p.name,
            "role": roles[p.name],
            "source_url": "https://example.invalid/" + p.name,
            "license_status": "synthetic_fixture",
            **file_digest(p),
        }
        for p in sorted(assets.iterdir())
    ]
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract))
    monkeypatch.setattr(source, "CONTRACT", contract_path)
    return assets, tmp_path / "package", contract, aligned


def build(original):
    assets, package, _, _ = original
    source.write_dilution_observer_package(assets, package)
    return EvidenceStore.open(package)


def query(store, name, **kwargs):
    return store.query("dilution_" + name, limit=None, **kwargs).table.to_pylist()


def test_literal_records_large_ids_and_public_namespace_joins(original):
    store = build(original)
    assert package_family(store.manifest) == "dilution_observer"
    rows = query(
        store,
        "aligned_quantities",
        joins=["acquisition", "library_precursor"],
        select=[
            "native_feature_id",
            "native_precursor_id",
            "quantitative_raw",
            "quantitative_value",
            "value_status",
            "acquisition.source_run_id",
            "library_precursor.native_library_group_id",
        ],
    )
    selected = next(r for r in rows if r["native_feature_id"] == "9007199254740993")
    assert selected["native_precursor_id"] == "1"
    assert selected["library_precursor.native_library_group_id"] == "77"
    assert selected["acquisition.source_run_id"] == "1804006131325186151"
    assert selected["quantitative_raw"] == "12.5" and selected["quantitative_value"] == 12.5
    assert sum(r["native_feature_id"] == "9007199254740992" for r in rows) == 1
    assert any(r["value_status"] == "zero" and r["quantitative_value"] == 0 for r in rows)
    assert any(r["value_status"] == "missing" and r["quantitative_value"] is None for r in rows)
    original_rows = query(store, "source_records", where={"source_file": original[2]["aligned"]})
    assert len(original_rows) == len(original[3])
    for saved, raw in zip(original_rows, original[3], strict=True):
        literal = json.loads(saved["source_row_json"])
        assert dict(zip(literal["headers"], literal["values"])) == raw


def test_missing_ambiguous_decoy_and_identity_disagreement_are_not_accepted(original):
    store = build(original)
    links = query(store, "feature_links", joins=["aligned", "scored"])
    states = [r["feature_link_status"] for r in links]
    assert states.count("missing_feature_id") == 3
    assert states.count("unmatched_feature_id") == 1
    assert states.count("ambiguous_feature_id") == 2
    assert states.count("id_crosscheck_disagreement") == 1
    assert any(r["target_agrees"] is False for r in links)
    aligned = query(store, "aligned_quantities")
    assert any(
        r["library_link_status"] == "ambiguous_form" and r["library_precursor_id"] is None
        for r in aligned
    )
    assert any(r["library_label_status"] == "both_missing_different_literal" for r in aligned)
    comparisons = query(store, "comparisons", where={"compared_source_field": "intensity"})
    assert len(comparisons) == 4
    assert all(r["comparison_status"] == "decimal_equal" for r in comparisons)


def test_intended_panel_forms_and_material_context_do_not_create_labels(original):
    store = build(original)
    members = query(
        store,
        "standard_memberships",
        joins=["standard", "library_precursor"],
        select=[
            "membership_status",
            "standard.peptide_sequence",
            "standard.label_position_1based",
            "library_precursor.charge_raw",
        ],
    )
    absent = [r for r in members if r["membership_status"] == "absent_from_library"]
    assert len(absent) == 1 and absent[0]["standard.peptide_sequence"] == "VYAETLSGFIK"
    assert {r["membership_status"] for r in members} == {"target", "decoy", "absent_from_library"}
    assert any(r["library_precursor.charge_raw"] == "3" for r in members)
    contexts = query(store, "material_contexts")
    assert all(r["specimen_id"] is None and r["aliquot_id"] is None for r in contexts)
    assert not links_with_minimum_precision(query(store, "material_links"), "exact_specimen")
    acquisitions = query(store, "acquisitions", joins=["nominal_level", "context"])
    assert {r["dia_index_from_name"] for r in acquisitions} == {1, 2}
    assert all("not_assumed_exchangeable" in r["replicate_status"] for r in acquisitions)
    claims = query(store, "assertions", where={"assertion_predicate": "nominal_dilution_grid"})
    assert len(claims) == 3 and all(
        r["assertion_status"] == "source_reported_conflict_retained" for r in claims
    )
    matrix = query(store, "assertions", where={"assertion_predicate": "cellular_mhc_genotype"})
    assert matrix[0]["source_scope"] == "matrix"
    assert "HLA-A*02:01" in matrix[0]["assertion_value"]
    protocol = query(
        store, "assertions", where={"assertion_predicate": "reported_DIA_method_difference"}
    )
    assert protocol[0]["assertion_value"] == "different"
    assert json.loads(protocol[0]["details_json"])["evidence_text"] == "DIA1 and DIA2 differ."


def test_public_lazy_api_exposes_the_verified_source_package():
    import hitlist

    assert hitlist.write_dilution_observer_package is source.write_dilution_observer_package
    assert hitlist.verify_dilution_observer_package is source.verify_dilution_observer_package


def test_protocol_assertions_require_literal_source_support(original):
    assets, package, contract, _ = original
    contract["protocol_assertions"][0]["evidence_text"] = "Unsupported new protocol claim"
    source.CONTRACT.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="lacks its pinned source text"):
        source.write_dilution_observer_package(assets, package)


def rehash(package, relative):
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][relative] = file_digest(package / relative)
    manifest_path.write_text(json.dumps(manifest))


def test_typed_tampering_rejected_even_after_inventory_rehash(original):
    build(original)
    package = original[1]
    relative = "dilution_observer/typed/aligned_quantities.parquet"
    table = pq.read_table(package / relative)
    rows = table.to_pylist()
    rows[0]["quantitative_value"] = 999.0
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), package / relative)
    rehash(package, relative)
    with pytest.raises(ValueError, match="original-source replay differs"):
        source.verify_dilution_observer_package(package)


def test_original_tampering_rejected_even_after_inventory_rehash(original):
    build(original)
    package = original[1]
    relative = "dilution_observer/source_assets/" + original[2]["aligned"]
    with (package / relative).open("a") as stream:
        stream.write("changed\n")
    rehash(package, relative)
    with pytest.raises(ValueError, match="Pinned dilution source differs"):
        source.verify_dilution_observer_package(package)


def test_snapshot_independence_location_overrides_and_no_overwrite(original):
    assets, package, contract, _ = original
    filename = contract["aligned"]
    source.write_dilution_observer_package(assets, package, assets={filename: assets / filename})
    snapshot = package / source.ROOT / "source_assets" / filename
    assert snapshot.stat().st_ino != (assets / filename).stat().st_ino
    (assets / filename).write_text("caller-mutated original")
    source.verify_dilution_observer_package(package)
    with pytest.raises(FileExistsError):
        source.write_dilution_observer_package(assets, package)
    with pytest.raises(ValueError, match="Unknown dilution asset"):
        source.write_dilution_observer_package(
            assets, package.parent / "other", assets={"../unknown": assets / filename}
        )


def test_source_contract_and_unknown_headers_fail_closed(original):
    assets, package, contract, _ = original
    filename = contract["aligned"]
    path = assets / filename
    path.write_text(path.read_text().replace("feature_id", "changed_feature_id", 1))
    for asset in contract["assets"]:
        if asset["filename"] == filename:
            asset.update(file_digest(path))
    source.CONTRACT.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="source headers differ"):
        source.write_dilution_observer_package(assets, package)


@pytest.mark.parametrize(
    "literal, status",
    [
        ("NA", "missing"),
        ("", "missing"),
        ("0", "zero"),
        ("-1", "negative"),
        ("nan", "missing"),
        ("Infinity", "nonfinite"),
        ("invalid", "invalid"),
        ("1e9999", "outside_float_range"),
    ],
)
def test_numeric_states_do_not_create_censor_bounds(literal, status):
    assert number(literal)[1] == status


def test_comparison_precision_and_missingness():
    contract = {"comparison_atol": "1e-12", "comparison_rtol": "1e-12"}
    assert comparison("9007199254740993", "9007199254740992", contract)[1] == "1"
    assert comparison("1.000000000001", "1", contract)[0] == "within_numeric_tolerance"
    assert comparison("2", "1", contract)[0] == "different"
    assert comparison("NA", "0", contract) == ("aligned_missing", None)
    assert comparison("", "NA", contract) == ("both_missing", None)
