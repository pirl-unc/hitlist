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
