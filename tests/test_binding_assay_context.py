"""Binding protocol assertions retain their source and reporting scope."""

import json

import pandas as pd
import pytest

from hitlist import (
    binding_assay_context as review,
)
from hitlist import prepare_affinity_training_data
from hitlist.binding_assay_context import (
    binding_method_review_contract,
    review_binding_assay_methods,
    summarize_binding_method_reviews,
)
from hitlist.evidence_bindings import materialized_bindings
from hitlist.evidence_package import adjudication_manifest_path, package_family
from hitlist.provenance import file_digest


def _ledger(tmp_path, monkeypatch, mutate=None):
    ledger = json.loads(review.LEDGER_SOURCE.read_text())
    if mutate:
        mutate(ledger)
    path = tmp_path / "review.json"
    path.write_text(json.dumps(ledger))
    monkeypatch.setattr(review, "LEDGER_SHA256", file_digest(path)["sha256"])
    return path


def test_reviewed_ledger_distinguishes_unknown_from_source_absence(tmp_path, monkeypatch):
    path = _ledger(tmp_path, monkeypatch)
    ledger = review._load_ledger(path)
    claims = {
        claim["predicate"]: claim
        for claim in ledger["claims"]
        if claim["predicate"] != "capture_antibody"
    }
    assert claims["assay_pH"]["status"] == "unreported"
    assert claims["assay_pH"]["value"] is None
    assert claims["incubation_temperature_alternatives"]["status"] == (
        "reported_alternatives_per_assay_unresolved"
    )
    assert claims["KD_interpretation"]["status"] == "reported_interpretation_not_measurement"
    assert claims["KD_interpretation"]["unit"] is None


@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: data["claims"][0].update(status="invented"),
        lambda data: data["claims"][3].update(value="tracer A"),
        lambda data: data["claims"][4].update(source_excerpt=None),
        lambda data: data["claims"][3].update(status="explicit_absence"),
        lambda data: data["claims"].append(data["claims"][0]),
    ],
)
def test_review_ledger_rejects_unsupported_or_unsourced_claims(tmp_path, monkeypatch, mutation):
    path = _ledger(tmp_path, monkeypatch, mutation)
    with pytest.raises(ValueError):
        review._load_ledger(path)


def test_original_source_hash_and_excerpt_are_required(tmp_path):
    ledger = json.loads(review.LEDGER_SOURCE.read_text())
    ledger["source"]["section_title"] = "Methods"
    ledger["source"]["section_id"] = "methods"
    ledger["claims"] = [ledger["claims"][0]]
    xml = tmp_path / "source.xml"
    xml.write_text(
        "<article><permissions><license>Creative Commons Attribution License</license>"
        '</permissions><sec id="methods"><title>Methods</title>'
        "<p>assays based on the inhibition of binding of a high affinity radiolabeled peptide"
        "</p></sec></article>"
    )
    ledger["source"]["source_xml_sha256"] = file_digest(xml)["sha256"]
    assert review._source_section(xml, ledger)
    ledger["claims"][0]["source_excerpt"] = "unreported incubation temperature"
    with pytest.raises(ValueError, match="excerpt missing"):
        review._source_section(xml, ledger)
    ledger["claims"][0]["source_excerpt"] = "assays based on the inhibition"
    xml.write_text(xml.read_text().replace("Methods", "Results"))
    with pytest.raises(ValueError, match="bytes changed"):
        review._source_section(xml, ledger)


def test_review_successor_adds_two_relations_without_hiding_predecessor():
    old = materialized_bindings(
        9, appended_source_ids=["pxd009092", "pxd004886"], include_adjudications=True
    )
    new = materialized_bindings(
        10, appended_source_ids=["pxd009092", "pxd004886"], include_adjudications=True
    )
    assert set(new) == set(old) | {"binding_assay_context", "binding_protocol_claim"}
    envelope = {
        "schema_version": 10,
        "parent_kind": "external",
        "review_revision": review.REVIEW_REVISION,
        "appended_sources": {"pxd009092": {}},
        "adjudication_manifest_path": "manifests/adjudication.json",
    }
    assert package_family(envelope) == "binding_review"
    assert adjudication_manifest_path(envelope) == "manifests/adjudication.json"


def test_binding_method_review_contract_and_representative_rows():
    contract = binding_method_review_contract()
    assert contract["revision"] == review.METHOD_REVIEW_REVISION
    assert "12135423" in contract["cited_protocols"]
    assert sum(r["expected_ci_corpus_v4_rows"] for r in contract["reviews"]) == 27_891

    comment_dtu = (
        "In general, this data has been generated using a quantitative biochemical "
        "peptide-HLA interaction assay (see PMID: 12135423)."
    )
    comment_loci = (
        "Peptide-HLA-I (pHLA-I) complexes were formed by refolding... using a "
        "proximity based Luminescence Oxygen Channeling Immunoassay (LOCI)."
    )
    sample = pd.DataFrame(
        [
            # Representative original #871 rows: 1715424, 1715425, 1715426 (sub 1000447)
            {
                "provenance_id": "observation:record:iedb:row:160643",
                "assay_iri": "http://www.iedb.org/assay/1715424",
                "reference_iri": "http://www.iedb.org/reference/1019354",
                "submission_id": "1000447",
                "pmid": pd.NA,
                "peptide": "SLYNTVATL",
                "mhc_restriction": "HLA-A*02:01",
                "mhc_class": "I",
                "mhc_species": "Homo sapiens",
                "assay_method": "purified MHC/competitive/fluorescence",
                "response_measured": "dissociation constant KD",
                "measurement_units": "nM",
                "measurement_inequality": "=",
                "quantitative_measurement": "64.9",
                "quantitative_value": 64.9,
                "is_binding_assay": True,
                "assay_comments": comment_dtu,
            },
            {
                "provenance_id": "observation:record:iedb:row:160644",
                "assay_iri": "http://www.iedb.org/assay/1715425",
                "reference_iri": "http://www.iedb.org/reference/1019354",
                "submission_id": "1000447",
                "pmid": pd.NA,
                "peptide": "GILGFVFTL",
                "mhc_restriction": "HLA-A*02:01",
                "mhc_class": "I",
                "mhc_species": "Homo sapiens",
                "assay_method": "purified MHC/competitive/fluorescence",
                "response_measured": "dissociation constant KD",
                "measurement_units": "nM",
                "measurement_inequality": "=",
                "quantitative_measurement": "38.1",
                "quantitative_value": 38.1,
                "is_binding_assay": True,
                "assay_comments": comment_dtu,
            },
            {
                "provenance_id": "observation:record:iedb:row:160645",
                "assay_iri": "http://www.iedb.org/assay/1715426",
                "reference_iri": "http://www.iedb.org/reference/1019354",
                "submission_id": "1000447",
                "pmid": pd.NA,
                "peptide": "RLRAEAQVK",
                "mhc_restriction": "HLA-A*31:01",
                "mhc_class": "I",
                "mhc_species": "Homo sapiens",
                "assay_method": "purified MHC/competitive/fluorescence",
                "response_measured": "dissociation constant KD",
                "measurement_units": "nM",
                "measurement_inequality": "=",
                "quantitative_measurement": "2.5",
                "quantitative_value": 2.5,
                "is_binding_assay": True,
                "assay_comments": comment_dtu,
            },
            # DTU direct fluorescence KD (~EC50) citing 12135423 (sub 1000422)
            {
                "provenance_id": "obs:dtu_direct",
                "assay_iri": "http://www.iedb.org/assay/1607390",
                "reference_iri": "http://www.iedb.org/reference/1014017",
                "submission_id": "1000422",
                "pmid": pd.NA,
                "peptide": "EVDPIGHLY",
                "mhc_restriction": "HLA-A*26:01",
                "mhc_class": "I",
                "mhc_species": "Homo sapiens",
                "assay_method": "purified MHC/direct/fluorescence",
                "response_measured": "dissociation constant KD (~EC50)",
                "measurement_units": "nM",
                "measurement_inequality": ">",
                "quantitative_measurement": "20000",
                "quantitative_value": 20000.0,
                "is_binding_assay": True,
                "assay_comments": comment_dtu,
            },
            # Literature study ref 1021191 (PMID 21169544) citing 12135423
            {
                "provenance_id": "obs:tang2011",
                "assay_iri": "http://www.iedb.org/assay/1810181",
                "reference_iri": "http://www.iedb.org/reference/1021191",
                "submission_id": "",
                "pmid": 21169544,
                "peptide": "NLVPMVATV",
                "mhc_restriction": "HLA-A*02:01",
                "mhc_class": "I",
                "mhc_species": "Homo sapiens",
                "assay_method": "purified MHC/direct/fluorescence",
                "response_measured": "dissociation constant KD (~EC50)",
                "measurement_units": "nM",
                "measurement_inequality": "=",
                "quantitative_measurement": "15.0",
                "quantitative_value": 15.0,
                "is_binding_assay": True,
                "assay_comments": "Sylvester-Hvid, et al. 2002. [PMID: 12135423].",
            },
            # Control 1: LOCI assay must NOT be converted to ELISA or flagged as conflict
            {
                "provenance_id": "obs:loci",
                "assay_iri": "http://www.iedb.org/assay/9900001",
                "reference_iri": "http://www.iedb.org/reference/1029999",
                "submission_id": "1000700",
                "pmid": pd.NA,
                "peptide": "CLGGLLTMV",
                "mhc_restriction": "HLA-A*02:01",
                "mhc_class": "I",
                "mhc_species": "Homo sapiens",
                "assay_method": "purified MHC/direct/fluorescence",
                "response_measured": "dissociation constant KD (~EC50)",
                "measurement_units": "nM",
                "measurement_inequality": "=",
                "quantitative_measurement": "40.0",
                "quantitative_value": 40.0,
                "is_binding_assay": True,
                "assay_comments": comment_loci,
            },
            # Control 2: Unreviewed competitive fluorescence KD stays unreviewed
            {
                "provenance_id": "obs:other_kd",
                "assay_iri": "http://www.iedb.org/assay/9900002",
                "reference_iri": "http://www.iedb.org/reference/1039999",
                "submission_id": "",
                "pmid": 31201628,
                "peptide": "YVLDHLIVV",
                "mhc_restriction": "HLA-A*02:01",
                "mhc_class": "I",
                "mhc_species": "Homo sapiens",
                "assay_method": "purified MHC/competitive/fluorescence",
                "response_measured": "dissociation constant KD",
                "measurement_units": "nM",
                "measurement_inequality": "=",
                "quantitative_measurement": "80.0",
                "quantitative_value": 80.0,
                "is_binding_assay": True,
                "assay_comments": "Fluorescence polarization competition assay.",
            },
            # Drift guard: upstream IEDB later updates assay_method on ref 1019354 -> flag rereview
            {
                "provenance_id": "obs:upstream_changed",
                "assay_iri": "http://www.iedb.org/assay/1715999",
                "reference_iri": "http://www.iedb.org/reference/1019354",
                "submission_id": "1000447",
                "pmid": pd.NA,
                "peptide": "GLCTLVAML",
                "mhc_restriction": "HLA-A*02:01",
                "mhc_class": "I",
                "mhc_species": "Homo sapiens",
                "assay_method": "purified MHC/direct/ELISA",
                "response_measured": "dissociation constant KD",
                "measurement_units": "nM",
                "measurement_inequality": "=",
                "quantitative_measurement": "50.0",
                "quantitative_value": 50.0,
                "is_binding_assay": True,
                "assay_comments": comment_dtu,
            },
            # Future uncataloged fluorescence reference citing 12135423 -> flag for review
            {
                "provenance_id": "obs:new_12135423",
                "assay_iri": "http://www.iedb.org/assay/9900003",
                "reference_iri": "http://www.iedb.org/reference/9999999",
                "submission_id": "",
                "pmid": 99999999,
                "peptide": "FLPSDFFPSV",
                "mhc_restriction": "HLA-A*02:01",
                "mhc_class": "I",
                "mhc_species": "Homo sapiens",
                "assay_method": "purified MHC/competitive/fluorescence",
                "response_measured": "dissociation constant KD",
                "measurement_units": "nM",
                "measurement_inequality": "=",
                "quantitative_measurement": "12.0",
                "quantitative_value": 12.0,
                "is_binding_assay": True,
                "assay_comments": "Based on PMID: 12135423.",
            },
        ]
    )

    reviewed = review_binding_assay_methods(sample)
    # Raw columns and values are never modified (no KD->IC50 or fluorescence->ELISA rewrite).
    assert list(reviewed["assay_method"]) == list(sample["assay_method"])
    assert list(reviewed["response_measured"]) == list(sample["response_measured"])
    assert list(reviewed["quantitative_measurement"]) == list(sample["quantitative_measurement"])

    by_iri = reviewed.set_index("assay_iri")
    for iri in (
        "http://www.iedb.org/assay/1715424",
        "http://www.iedb.org/assay/1715425",
        "http://www.iedb.org/assay/1715426",
    ):
        row = by_iri.loc[iri]
        assert row["method_review_status"] == "source_method_conflict_submission_scope"
        assert row["method_review_evidence_scope"] == "submission_comment_citation_only"
        assert (
            row["reviewed_mechanism"]
            == "conflicting_reported_competitive_vs_cited_refolding_assembly"
        )
        assert row["reviewed_detector"] == "conflicting_reported_fluorescence_vs_cited_elisa"
        assert row["method_review_cited_pmid"] == "12135423"

    assert (
        by_iri.loc["http://www.iedb.org/assay/1607390", "method_review_status"]
        == "source_detector_conflict_submission_scope"
    )
    assert (
        by_iri.loc["http://www.iedb.org/assay/1810181", "method_review_status"]
        == "source_detector_conflict_study_scope"
    )
    assert (
        by_iri.loc["http://www.iedb.org/assay/9900001", "method_review_status"]
        == "not_source_reviewed"
    )
    assert (
        by_iri.loc["http://www.iedb.org/assay/9900002", "method_review_status"]
        == "not_source_reviewed"
    )
    assert (
        by_iri.loc["http://www.iedb.org/assay/1715999", "method_review_status"]
        == "upstream_tag_changed_requires_rereview"
    )
    assert by_iri.loc["http://www.iedb.org/assay/1715999", "reviewed_detector"] == ""
    assert (
        by_iri.loc["http://www.iedb.org/assay/9900003", "method_review_status"]
        == "uncataloged_citation_requires_review"
    )

    summary = summarize_binding_method_reviews(reviewed)
    assert summary["n_rows"] == len(sample)
    assert summary["by_status"]["source_method_conflict_submission_scope"] == 3
    assert summary["by_status"]["upstream_tag_changed_requires_rereview"] == 1
    assert summary["by_status"]["uncataloged_citation_requires_review"] == 1

    # Flows through prepare_affinity_training_data without converting KD to IC50
    sel = prepare_affinity_training_data(sample, species="Homo sapiens")
    assert len(sel.accepted) == len(sample)
    assert set(sel.accepted["method_review_status"]) == set(reviewed["method_review_status"])
    assert (
        sel.accepted.loc[
            sel.accepted["assay_iri"] == "http://www.iedb.org/assay/1715424", "measurement_kind"
        ].iloc[0]
        == "kd"
    )


@pytest.mark.integration
def test_real_corpus_binding_method_reviews_match_pinned_counts_and_keep_loci_untouched():
    from hitlist.observations import is_binding_built, load_binding

    if not is_binding_built():
        pytest.skip("Binding corpus not built")

    bdf = load_binding()
    reviewed = review_binding_assay_methods(bdf)
    summary = summarize_binding_method_reviews(reviewed)
    assert summary["n_rows"] == len(bdf)
    assert summary["by_status"]["source_method_conflict_submission_scope"] == 18_723
    assert summary["by_status"]["source_detector_conflict_submission_scope"] == 9_150
    assert summary["by_status"]["source_detector_conflict_study_scope"] == 18
    assert summary["by_status"].get("upstream_tag_changed_requires_rereview", 0) == 0
    assert summary["by_status"].get("uncataloged_citation_requires_review", 0) == 0

    loci_mask = reviewed["assay_comments"].fillna("").astype(str).str.contains("LOCI", case=False)
    assert int(loci_mask.sum()) >= 20_000
    assert (reviewed.loc[loci_mask, "method_review_status"] == "not_source_reviewed").all()
