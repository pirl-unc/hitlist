"""Measured HAP1 reference RNA and honest KO provenance (#358)."""

import gzip
import hashlib
from importlib.resources import files

import pandas as pd
import pytest
import yaml

from hitlist import downloads
from hitlist.builder import build_line_expression
from hitlist.conditions import engineered_material_mask
from hitlist.curation import load_pmid_overrides
from hitlist.export import generate_sample_expression_table
from hitlist.line_expression import (
    load_line_expression,
    load_line_expression_anchors,
    resolve_sample_expression_anchor,
    write_line_expression_index,
)


@pytest.fixture(autouse=True)
def empty_optional_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path)


def test_every_curated_hap1_ko_uses_parent_rna():
    study = load_pmid_overrides()[40113210]
    assert len(study["ms_samples"]) == 12
    for sample in study["ms_samples"]:
        anchor = resolve_sample_expression_anchor(sample["sample_label"])
        is_control = sample["condition_knockout_genes"] == "none"
        assert anchor.expression_match_tier == (1 if is_control else 2), sample["sample_label"]
        assert anchor.expression_key == "HAP1"
        assert anchor.expression_parent_key == (None if is_control else "HAP1")
        assert anchor.source_ids == ("DepMap_24Q4_HAP1_gene",)
    rows = load_line_expression(line_key="HAP1", granularity="gene")
    assert len(rows) == 19193
    assert rows.tpm.notna().all() and rows.tpm.ge(0).all()


def test_packaged_hap1_matches_its_build_report():
    # Catches an edited CSV whose recorded provenance was not regenerated.
    data = files("hitlist.data.line_expression")
    report = yaml.safe_load(data.joinpath("apm_rna_build_report.yaml").read_text())
    (entry,) = report["sources"]
    raw = data.joinpath(entry["file"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
    assert hashlib.sha256(gzip.decompress(raw)).hexdigest() == entry["content_sha256"]
    rows = load_line_expression(source_id=entry["source_id"])
    assert len(rows) == entry["n_genes"]
    assert int(rows.tpm.gt(0).sum()) == entry["n_positive_genes"]
    assert rows.tpm.sum() == pytest.approx(entry["sum_tpm"])


def test_downloaded_bundle_adds_hap1_transcripts_without_duplicating_genes(tmp_path):
    frames = {
        "depmap_models": pd.DataFrame(
            {"ModelID": ["ACH-002475", "ACH-001"], "StrippedCellLineName": ["HAP1", "HELA"]}
        ),
        "depmap_profiles": pd.DataFrame(
            {
                "ProfileID": ["PR-QtHaIL", "PR-hela"],
                "ModelID": ["ACH-002475", "ACH-001"],
                "Datatype": ["rna", "rna"],
                "Stranded": [True, False],
            }
        ),
        "depmap_default_profiles": pd.DataFrame(
            {
                "ModelID": ["ACH-002475", "ACH-001"],
                "ProfileID": ["PR-QtHaIL", "PR-hela"],
                "ProfileType": ["rna", "rna"],
            }
        ),
        "depmap_rna": pd.DataFrame({"": ["ACH-002475", "ACH-001"], "TP53 (7157)": [20.0, 3.0]}),
        "depmap_rna_transcript": pd.DataFrame(
            {"": ["PR-QtHaIL", "PR-hela"], "TP53 (ENST00000269305)": [4.0, 2.0]}
        ),
    }
    for key, frame in frames.items():
        path = tmp_path / f"{key}.csv"
        frame.to_csv(path, index=False)
        downloads.register(key, path)
    rows = build_line_expression()

    hap1 = rows[rows.line_key.eq("HAP1")]
    genes = hap1[hap1.granularity.eq("gene")]
    assert len(genes) == 19193
    assert genes.source_id.eq("DepMap_24Q4_HAP1_gene").all()
    assert not genes.tpm.eq(2**20 - 1).any()
    # Metadata is stamped from sources.yaml, not carried in the packaged CSV.
    assert genes.backend.eq("depmap_rna").all()
    assert genes.license.eq("CC BY 4.0 (Broad DepMap)").all()
    transcripts = hap1[hap1.granularity.eq("transcript")]
    assert transcripts.source_id.tolist() == ["DepMap_24Q4_transcript"]
    assert transcripts.tpm.tolist() == [15.0]
    # Lines that list the downloaded gene source still receive it.
    hela = rows[rows.line_key.eq("HeLa") & rows.granularity.eq("gene")]
    assert hela.source_id.tolist() == ["DepMap_24Q4_gene"]

    anchor = resolve_sample_expression_anchor("HAP1 wildtype")
    assert anchor.expression_backend == "depmap_rna"
    assert anchor.source_ids == ("DepMap_24Q4_HAP1_gene", "DepMap_24Q4_transcript")


# ── Engineered samples never take parental exact-line RNA (#576) ────────────


@pytest.fixture
def every_registered_source(tmp_path):
    """An index holding rows for every source the registry names.

    The guard below must hold whichever optional data a user installed, so it
    runs against the most that could be installed: every (line, source) pair
    the resolver can ever find.
    """
    pairs = {
        (str(entry["expression_key"]), str(source_id))
        for entry in load_line_expression_anchors()
        if entry.get("expression_key")
        for source_id in entry.get("source_ids") or []
    }
    write_line_expression_index(pd.DataFrame(sorted(pairs), columns=["line_key", "source_id"]))
    return pairs


def test_engineered_label_on_a_parental_alias_uses_parent_rna(every_registered_source):
    parental = resolve_sample_expression_anchor("HeLa-CIITA (Mock)")
    assert parental.expression_match_tier == 1
    engineered = resolve_sample_expression_anchor("HeLa-CIITA (Mock)", engineered=True)
    assert engineered.expression_match_tier == 2
    assert engineered.expression_key == "HeLa"
    assert engineered.expression_parent_key == "HeLa"
    # Same data, honest provenance: only the tier and parent change.
    assert engineered.expression_backend == parental.expression_backend
    assert engineered.source_ids == parental.source_ids
    assert engineered.matched_alias == parental.matched_alias


def test_curated_engineering_is_the_default(every_registered_source):
    """Not opt-in: ``pmid`` + ``sample_label`` of a curated arm is enough.

    An external caller that never heard of ``engineered`` still gets honest
    provenance for a curated engineered arm, and an explicit value overrides.
    """
    engineered = resolve_sample_expression_anchor("HeLa-CIITA (Mock)", pmid=36215666)
    assert engineered.expression_match_tier == 2
    assert engineered.expression_parent_key == "HeLa"
    parental = resolve_sample_expression_anchor("HeLa naive (HLA-A*02:01)", pmid=19748539)
    assert parental.expression_match_tier == 1
    # Introduced MHC is engineering too: a K562 DPB1 transfectant is not K562.
    transfectant = resolve_sample_expression_anchor(
        "K562 transfectant DPB1*01:01/DPA1*02:01", pmid=32350084
    )
    assert transfectant.expression_match_tier == 2
    assert transfectant.expression_parent_key == "K562"
    forced = resolve_sample_expression_anchor("HeLa-CIITA (Mock)", pmid=36215666, engineered=False)
    assert forced.expression_match_tier == 1


@pytest.mark.parametrize("pmid", [None, float("nan"), pd.NA])
def test_missing_pmid_means_no_curated_lookup(every_registered_source, pmid):
    """An exported frame's missing PMID must not raise; main ignored pmid entirely."""
    anchor = resolve_sample_expression_anchor("HeLa cells", pmid=pmid)
    assert anchor.expression_match_tier == 1


def test_unresolved_arm_in_an_engineered_study_does_not_claim_exact_rna(
    every_registered_source, monkeypatch
):
    """A row ``_consensus_meta`` could not assign to one arm is not known to be wild type.

    HAP1's study has one wild-type arm and eleven knockouts, so an unattributed
    row's blanked condition block means "unknown".  A study with no engineered
    arms keeps its exact-line anchor for the same kind of row.
    """
    from hitlist.export import _attach_peptide_origin

    monkeypatch.setattr(
        "hitlist.mappings.load_peptide_mappings",
        lambda peptide=None, columns=None, **_: pd.DataFrame(
            columns=["peptide", "gene_name", "gene_id", "protein_id"]
        ),
    )
    rows = pd.DataFrame(
        {
            "peptide": ["AAAAAAAAA", "CCCCCCCCC", "DDDDDDDDD"],
            "sample_label": ["", "HAP1 wildtype", ""],
            "pmid": [40113210, 40113210, 19748539],
            "study_label": ["S", "S", "T"],
            "cell_name": ["HAP1", "HAP1", "HeLa"],
            "condition_knockout_genes": ["", "none", ""],
        }
    )
    out = _attach_peptide_origin(rows).set_index("peptide")

    unresolved = out.loc["AAAAAAAAA"]
    assert unresolved.expression_match_tier == 2
    assert unresolved.expression_parent_key == "HAP1"
    assert out.loc["CCCCCCCCC"].expression_match_tier == 1  # the resolved wild-type arm
    assert out.loc["DDDDDDDDD"].expression_match_tier == 1  # no engineered arms in 19748539
    reason = resolve_sample_expression_anchor("", cell_name="HAP1", pmid=40113210).reason
    assert "arm unresolved in a study that includes engineered arms" in reason


def test_engineered_sample_never_resolves_at_tier_1(tmp_path, monkeypatch):
    """Not even through a derivative entry with RNA registered under it.

    Registry derivative entries are catch-alls — ``HAP1-KO`` covers eleven
    knockouts, ``C1R-HLA`` every C1R transfectant — so RNA registered under
    one would pose as exact RNA for all the others.  The hit resolves at tier
    2, naming the registered material as the reference.
    """
    from hitlist import line_expression as le

    monkeypatch.setattr(
        le,
        "_load_anchors_yaml",
        lambda: [
            {
                "name": "HeLa",
                "aliases": ["hela"],
                "parent_line": None,
                "line_family": "tumor_line",
                "expression_backend": "depmap_rna",
                "expression_key": "HeLa",
                "source_ids": ["DepMap_24Q4_gene"],
            },
            {
                "name": "HeLa-CIITA",
                "aliases": ["hela-ciita"],
                "parent_line": "HeLa",
                "line_family": "tumor_line",
                "expression_backend": "packaged_rnaseq",
                "expression_key": "HeLa-CIITA",
                "source_ids": ["Pearson_BLCL_panel"],
            },
        ],
    )
    write_line_expression_index(
        pd.DataFrame(
            {
                "line_key": ["HeLa", "HeLa-CIITA"],
                "source_id": ["DepMap_24Q4_gene", "Pearson_BLCL_panel"],
            }
        ),
    )

    anchor = resolve_sample_expression_anchor("HeLa-CIITA (Mock)", engineered=True)
    assert anchor.expression_match_tier == 2
    assert anchor.expression_key == "HeLa-CIITA"
    assert anchor.expression_parent_key == "HeLa-CIITA"


def test_no_curated_engineered_sample_resolves_at_tier_1(every_registered_source):
    table = generate_sample_expression_table()
    engineered = engineered_material_mask(table)
    offenders = table.loc[engineered & table.expression_match_tier.eq(1)]
    assert offenders.empty, offenders[["pmid", "sample_label", "expression_key"]]
    # Not vacuous: without their curated engineering these labels would
    # claim the reference line's RNA as their own — knockouts, transfected
    # genes and introduced MHC alike.
    label_only = {
        label
        for label in table.loc[engineered, "sample_label"]
        if resolve_sample_expression_anchor(label).expression_match_tier == 1
    }
    assert {
        "SaOS-2 + TP53 R175H",
        "HeLa-CIITA + T6BP siRNA",
        "THP-1 TAP1 knockout + mock infection",
        "K562 transfectant DPB1*01:01/DPA1*02:01",
        "K562 transfected with DLA-88*501:01",
        "HeLa-sHLA-HLA-A*02:01",
    } <= label_only
    hap1 = table.loc[table.pmid.eq(40113210)]
    assert len(hap1) == 12
    assert hap1.expression_match_tier.eq(hap1.condition_knockout_genes.ne("none") + 1).all()
