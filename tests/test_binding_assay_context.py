"""Binding protocol assertions retain their source and reporting scope."""

import json

import pytest

from hitlist import binding_assay_context as review
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
