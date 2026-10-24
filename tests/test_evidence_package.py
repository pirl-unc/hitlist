"""A version shared by two historical producers must not choose the wrong reader."""

import json

import pytest

from hitlist import evidence_store, external_successor
from hitlist.evidence_package import adjudication_manifest_path, package_family


@pytest.mark.parametrize(
    "envelope,family,review_path",
    [
        ({"schema_version": 7}, "reference", None),
        ({"schema_version": 8, "review_revision": "pinned"}, "adjudication", "manifest.json"),
        ({"schema_version": 8, "appended_sources": {"pxd009092": {}}}, "external", None),
        (
            {"schema_version": 9, "appended_sources": {}, "parent_kind": "adjudication"},
            "external",
            "manifests/adjudication.json",
        ),
    ],
)
def test_store_uses_family_specific_verifier(tmp_path, monkeypatch, envelope, family, review_path):
    (tmp_path / "manifest.json").write_text(json.dumps(envelope))
    calls = []

    def verifier(name):
        def verify(directory):
            assert directory == tmp_path
            calls.append(name)
            return envelope

        return verify

    monkeypatch.setattr(evidence_store, "verify_reference_evidence_package", verifier("reference"))
    monkeypatch.setattr(
        evidence_store, "verify_assay_adjudication_package", verifier("adjudication")
    )
    monkeypatch.setattr(external_successor, "verify_external_successor", verifier("external"))
    monkeypatch.setattr(
        evidence_store.EvidenceStore,
        "__init__",
        lambda self, directory, manifest: setattr(self, "manifest", manifest),
    )
    assert evidence_store.EvidenceStore.open(tmp_path).manifest == envelope
    assert calls == [family]
    assert package_family(envelope) == family
    assert adjudication_manifest_path(envelope) == review_path


@pytest.mark.parametrize(
    "envelope",
    [
        {"schema_version": 8},
        {"schema_version": 8, "appended_sources": {}, "review_revision": "ambiguous"},
        {"schema_version": 9, "review_revision": "wrong_version"},
        {"schema_version": 9},
        {"schema_version": 100},
    ],
)
def test_ambiguous_or_unknown_envelope_fails_before_reading_tables(tmp_path, envelope):
    (tmp_path / "manifest.json").write_text(json.dumps(envelope))
    with pytest.raises(evidence_store.EvidenceQueryError, match="Unknown or ambiguous"):
        evidence_store.EvidenceStore.open(tmp_path)


@pytest.mark.parametrize(
    "family,revision",
    [
        ("hap1_quantitative", "shapiro2025_hap1_quantitative_v1"),
        ("precursor_context", "pxd009360_precursor_context_v1"),
    ],
)
def test_schema_twenty_keeps_explicit_source_families_separate(
    tmp_path, monkeypatch, family, revision
):
    from hitlist import hap1_quantitative, precursor_context_package

    envelope = {"schema_version": 20, "package_family": family, "revision": revision}
    (tmp_path / "manifest.json").write_text(json.dumps(envelope))
    calls = []

    def verify(expected):
        def callback(directory):
            assert directory == tmp_path
            calls.append(expected)
            return envelope

        return callback

    monkeypatch.setattr(
        hap1_quantitative, "verify_hap1_quantitative_package", verify("hap1_quantitative")
    )
    monkeypatch.setattr(
        precursor_context_package, "verify_precursor_context_package", verify("precursor_context")
    )
    monkeypatch.setattr(
        evidence_store.EvidenceStore,
        "__init__",
        lambda self, directory, manifest: setattr(self, "manifest", manifest),
    )
    assert evidence_store.EvidenceStore.open(tmp_path).manifest == envelope
    assert calls == [family]
    assert package_family(envelope) == family
    assert adjudication_manifest_path(envelope) is None
    changed = {
        **envelope,
        "package_family": "precursor_context"
        if family == "hap1_quantitative"
        else "hap1_quantitative",
    }
    with pytest.raises(ValueError, match="Unknown or ambiguous"):
        package_family(changed)
