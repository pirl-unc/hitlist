# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for hitlist.apm — APM perturbation classifier."""

from __future__ import annotations

from hitlist.apm import APM_GENES, apm_columns_for_sample, classify_apm_perturbations


def test_classify_apm_returns_stable_keys():
    """Output dict has one key per APM gene, even when nothing matched."""
    out = classify_apm_perturbations("nothing relevant here")
    assert set(out.keys()) == set(APM_GENES.keys())
    assert all(v is False for v in out.values())


def test_classify_apm_recognizes_crispr_kos():
    """v1.30.16: Shapiro 2025-style perturbation strings."""
    out = classify_apm_perturbations("B2M CRISPR KO (beta-2-microglobulin — complete HLA-I loss)")
    assert out["b2m"] is True
    # Don't fire on neighbors that share a prefix.
    assert out["ciita"] is False


def test_classify_apm_handles_alternate_names():
    """ERp57 ↔ PDIA3, calreticulin ↔ CALR, tapasin ↔ TAPBP."""
    assert classify_apm_perturbations("ERp57 knockout")["pdia3"] is True
    assert classify_apm_perturbations("calreticulin shRNA")["calr"] is True
    assert classify_apm_perturbations("tapasin deficient")["tapbp"] is True


def test_classify_apm_invariant_chain_word_boundary():
    """The 'Ii' abbreviation must NOT match 'HLA-II' / 'class II' /
    'glucosidase II' (false positives that polluted the v1.30.15
    sweep). Only literal 'CD74' or the full phrase 'invariant chain'
    should fire the cd74 flag."""
    # False-positive cases — must NOT fire.
    assert classify_apm_perturbations("HLA-I and HLA-II profiled")["cd74"] is False
    assert classify_apm_perturbations("class II peptides")["cd74"] is False
    assert classify_apm_perturbations("GANAB (glucosidase II alpha)")["cd74"] is False
    # True-positive cases — must fire.
    assert classify_apm_perturbations("CD74 knockout")["cd74"] is True
    assert classify_apm_perturbations("invariant chain deficient")["cd74"] is True


def test_classify_apm_proteasome_inhibitors():
    """v1.30.16: proteasome inhibitor drug names route to the
    proteasome_inhibitor flag rather than any specific subunit."""
    drugs = ["bortezomib", "MG132", "MG-132", "epoxomicin", "carfilzomib", "lactacystin"]
    for drug in drugs:
        out = classify_apm_perturbations(f"treated with {drug} 24h")
        assert out["proteasome_inhibitor"] is True, drug


def test_classify_apm_proteasome_subunits():
    """LMP7 ↔ PSMB8 etc — alternate names for the immunoproteasome
    catalytic subunits."""
    assert classify_apm_perturbations("LMP7 KO")["psmb8"] is True
    assert classify_apm_perturbations("PSMB9 shRNA")["psmb9"] is True
    assert classify_apm_perturbations("MECL-1 deficient")["psmb10"] is True


def test_classify_apm_tap_inhibitors():
    """ICP47 (HSV) and US6 (HCMV) are viral TAP blockers — they
    perturb antigen presentation without touching the gene."""
    assert classify_apm_perturbations("ICP47-expressing cells")["tap_inhibitor"] is True
    assert classify_apm_perturbations("US6 transfectant")["tap_inhibitor"] is True


def test_classify_apm_t2_cells_routes_to_tap_deficient_line():
    """T2 / RMA-S are genomically TAP-deficient cell lines. Captured
    under the dedicated tap_deficient_line flag so consumers can
    pivot on 'any TAP deficiency' without joining a separate
    cell-line lookup."""
    assert classify_apm_perturbations("T2 cells (TAP-deficient)")["tap_deficient_line"] is True
    assert classify_apm_perturbations("RMA-S model")["tap_deficient_line"] is True
    # T2 inside a longer non-cell-name word should not fire.
    assert classify_apm_perturbations("PT2 cohort")["tap_deficient_line"] is False


def test_classify_apm_cytokines():
    """IFN-gamma / alpha / beta and TNF-alpha are APM modulators (they
    induce class-I expression). Captured separately from gene KOs so
    consumers can distinguish 'deficient' from 'induced' samples.
    The Greek letter form is the standard biology notation in many
    papers, so the regex matches both Unicode gamma and the
    spelled-out form."""
    assert classify_apm_perturbations("IFN-γ stimulation 24h")["ifn_gamma"] is True  # noqa: RUF001
    assert classify_apm_perturbations("IFN-gamma 100 U/mL")["ifn_gamma"] is True
    assert classify_apm_perturbations("interferon-alpha treatment")["ifn_alpha"] is True
    assert classify_apm_perturbations("LPS 100 ng/mL")["lps"] is True


def test_apm_columns_for_sample_union_flag():
    """The union flag is True iff any individual gene flag is True."""
    cols = apm_columns_for_sample("ERAP1 CRISPR KO", ["ERAP1 CRISPR/Cas9 knockout"])
    assert cols["apm_perturbed"] is True
    assert cols["apm_genes_perturbed"] == "erap1"
    assert cols["apm_erap1_perturbed"] is True


def test_apm_columns_for_sample_unperturbed_returns_all_false():
    """Clean unperturbed sample → union False, genes string empty."""
    cols = apm_columns_for_sample("unperturbed — standard culture", [])
    assert cols["apm_perturbed"] is False
    assert cols["apm_genes_perturbed"] == ""


def test_apm_columns_for_sample_multi_gene_concatenates():
    """A study perturbing several genes lists them all in
    apm_genes_perturbed (semicolon-joined, lowercase, sorted by the
    APM_GENES dict order — convenient for grep + group-by)."""
    cols = apm_columns_for_sample(
        condition="B2M + TAP1 double knockout",
        perturbations=["B2M CRISPR/Cas9 knockout", "TAP1 CRISPR/Cas9 knockout"],
    )
    assert cols["apm_perturbed"] is True
    assert "b2m" in cols["apm_genes_perturbed"]
    assert "tap1" in cols["apm_genes_perturbed"]


def test_apm_columns_propagate_through_ms_samples_table(monkeypatch):
    """End-to-end: generate_ms_samples_table emits the apm_*
    columns straight from the YAML and the apm_only filter narrows
    to perturbed rows."""
    from hitlist import export

    fake_overrides = {
        100: {
            "study_label": "Clean",
            "ms_samples": [
                {"sample_label": "wt", "condition": "unperturbed", "mhc_class": "I"},
            ],
        },
        200: {
            "study_label": "ERAP1 KO",
            "perturbations": ["ERAP1 CRISPR/Cas9 knockout"],
            "ms_samples": [
                {
                    "sample_label": "erap1_ko",
                    "condition": "ERAP1 CRISPR/Cas9 knockout",
                    "mhc_class": "I",
                    "n_samples": 1,
                },
            ],
        },
    }
    monkeypatch.setattr("hitlist.export.load_pmid_overrides", lambda: fake_overrides)

    df = export.generate_ms_samples_table()
    assert "apm_perturbed" in df.columns
    assert "apm_erap1_perturbed" in df.columns
    assert "apm_genes_perturbed" in df.columns

    perturbed = df[df["apm_perturbed"]]
    assert len(perturbed) == 1
    assert perturbed.iloc[0]["sample_label"] == "erap1_ko"
    assert perturbed.iloc[0]["apm_genes_perturbed"] == "erap1"

    # apm_only=True returns the same row.
    df_apm = export.generate_ms_samples_table(apm_only=True)
    assert len(df_apm) == 1
    assert df_apm.iloc[0]["sample_label"] == "erap1_ko"
