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

"""Tests for hitlist.cell_name_parser (#261).

Coverage:
- Pure cell-line names + registry lookup
- Synonyms (293T / 293-T / HEK 293T)
- "<line> cells" suffix stripping (JY cells / C1R cells)
- Pure cell-type names (primary-cell rows)
- Hybrid "<line>-<type>" strings
- Genetic modifications (CALR KO, wildtype, CRISPR)
- Engineering constructs (MAPTAC, Strep-tag II, sHLA)
- Donor ID extraction from attributed_sample_label
- Uninformative inputs (empty, "Other", "Cell found in tissue")
- src_cell_line hint for unresolvable strings
"""

from __future__ import annotations

import pytest

from hitlist.cell_name_parser import (
    CellNameInfo,
    known_cell_lines,
    known_cell_types,
    parse_cell_name,
)

# ── Pure cell-line names ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw, expected_canonical, expected_type",
    [
        ("Expi293F", "Expi293F", "Epithelial cell"),
        ("HeLa", "HeLa", "Epithelial cell"),
        ("Raji", "Raji", "B cell"),
        ("JY", "JY", "B cell"),
        ("MDA-MB-231", "MDA-MB-231", "Epithelial cell"),
        ("HCT 116", "HCT 116", "Epithelial cell"),
        ("A549", "A549", "Epithelial cell"),
        ("K562", "K562", "Myeloid cell"),
        ("HAP1", "HAP1", "Myeloid cell"),
    ],
)
def test_pure_cell_line_resolves_to_registry_canonical(raw, expected_canonical, expected_type):
    out = parse_cell_name(raw)
    assert out.is_cell_line is True
    assert out.cell_line_name == expected_canonical
    assert out.cell_type == expected_type
    assert out.genetic_modification == ""
    assert out.raw_cell_name == raw


# ── Synonyms ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "synonym, canonical",
    [
        ("293T", "HEK293T"),
        ("293-T", "HEK293T"),
        ("HEK 293T", "HEK293T"),
        ("HEK293-T", "HEK293T"),
        ("293/T", "HEK293T"),
        ("THP1", "THP-1"),
        ("THP 1", "THP-1"),
        ("MCF7", "MCF-7"),
        ("HCT116", "HCT 116"),
        ("HCT-116", "HCT 116"),
        ("K-562", "K562"),
        ("HAP-1", "HAP1"),
        ("HROG-17", "HROG17"),
    ],
)
def test_synonyms_normalize_to_canonical(synonym, canonical):
    out = parse_cell_name(synonym)
    assert out.cell_line_name == canonical
    assert out.cell_line_input == synonym
    assert out.is_cell_line is True


def test_line_lookup_is_case_insensitive():
    """Different IEDB rows record the same line at different casings."""
    out = parse_cell_name("hela")
    assert out.cell_line_name == "HeLa"


# ── "<line> cells" suffix stripping ───────────────────────────────────


@pytest.mark.parametrize(
    "raw, canonical",
    [
        ("JY cells", "JY"),
        ("C1R cells", "C1R"),
    ],
)
def test_trailing_cells_suffix_does_not_block_lookup(raw, canonical):
    out = parse_cell_name(raw)
    assert out.cell_line_name == canonical
    assert out.is_cell_line is True


# ── Pure cell-type names ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw",
    [
        "B cell",
        "T cell",
        "NK cell",
        "Splenocyte",
        "Thymocyte",
        "Lymphocyte",
        "Lymphoblast",
        "Dendritic cell",
        "Melanocyte",
        "Fibroblast",
        "Hepatocyte",
        "Glial cell",
        "Epithelial cell",
        "Monocyte",
        "PBMC",
    ],
)
def test_pure_cell_type_string_is_primary_cell(raw):
    out = parse_cell_name(raw)
    assert out.is_cell_line is False
    assert out.cell_line_name == ""
    assert out.cell_type == raw
    assert out.genetic_modification == ""


# ── Hybrid <line>-<cell_type> ─────────────────────────────────────────


@pytest.mark.parametrize(
    "raw, canonical, cell_type",
    [
        ("K562-Myeloid cell", "K562", "Myeloid cell"),
        ("C1R cells-B cell", "C1R", "B cell"),
        ("HeLa cells-Epithelial cell", "HeLa", "Epithelial cell"),
        ("MDA-MB-231-Epithelial cell", "MDA-MB-231", "Epithelial cell"),
        ("HCT 116-Epithelial cell", "HCT 116", "Epithelial cell"),
        ("A549-Epithelial cell", "A549", "Epithelial cell"),
        ("JY cells-B cell", "JY", "B cell"),
        ("THP-1-Monocyte", "THP-1", "Monocyte"),
        ("LM-MEL-44-Melanocyte", "LM-MEL-44", "Melanocyte"),
        ("LM-MEL-33-Melanocyte", "LM-MEL-33", "Melanocyte"),
        ("MAVER-1-Lymphoblast", "MAVER-1", "Lymphoblast"),
        ("MCF-7/LY2-Epithelial cell", "MCF-7/LY2", "Epithelial cell"),
    ],
)
def test_hybrid_string_splits_into_line_and_cell_type(raw, canonical, cell_type):
    out = parse_cell_name(raw)
    assert out.is_cell_line is True
    assert out.cell_line_name == canonical
    assert out.cell_type == cell_type


def test_hybrid_case_difference_normalizes_via_registry():
    """IEDB records "Glial cell" vs "Glial Cell" inconsistently;
    registry-derived cell_type normalizes."""
    out = parse_cell_name("HROG17-Glial Cell")
    assert out.cell_line_name == "HROG17"
    # Registry says "Glial cell" lowercase — used as canonical.
    assert out.cell_type == "Glial cell"


# ── Genetic modifications ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw, expected_mod",
    [
        ("HAP1 wildtype", "wildtype"),
        ("HAP1 WT", "wildtype"),
        ("HAP1 wild-type", "wildtype"),
        ("HAP1 CALR KO", "CALR KO"),
        ("HAP1 CANX KO", "CANX KO"),
        ("HAP1 SPPL3 KO", "SPPL3 KO"),
        ("HAP1 B2M KO", "B2M KO"),
        ("HAP1 TAP1 KO", "TAP1 KO"),
    ],
)
def test_hap1_variants_extract_genetic_modification(raw, expected_mod):
    out = parse_cell_name(raw)
    assert out.cell_line_name == "HAP1"
    assert out.is_cell_line is True
    assert out.genetic_modification == expected_mod


# ── Engineering systems (monoallelic_host) ────────────────────────────


def test_maptac_construct_recognized_via_monoallelic_host():
    out = parse_cell_name("", monoallelic_host="MAPTAC")
    assert out.is_cell_line is True
    assert out.genetic_modification == "MAPTAC"
    assert out.cell_line_name == ""  # not a conventional line


def test_strep_tag_construct_recognized_via_monoallelic_host():
    out = parse_cell_name("", monoallelic_host="Strep-tag II")
    assert out.is_cell_line is True
    assert out.genetic_modification == "Strep-tag II"


def test_shla_construct_recognized_via_monoallelic_host():
    out = parse_cell_name("", monoallelic_host="sHLA (VLDLr-tagged, secreted)")
    assert out.is_cell_line is True
    assert out.genetic_modification == "sHLA"


def test_monoallelic_host_resolves_to_real_line_when_applicable():
    """C1R / 721.221 / K562 in monoallelic_host are real cell lines
    used as mono-allelic hosts — not engineering constructs."""
    out = parse_cell_name("", monoallelic_host="C1R")
    assert out.is_cell_line is True
    assert out.cell_line_name == "C1R"
    assert out.genetic_modification == ""  # the host is the LINE itself


def test_monoallelic_host_721_221():
    out = parse_cell_name("", monoallelic_host="721.221")
    assert out.cell_line_name == "721.221"
    assert out.cell_type == "B cell"


# ── Donor ID extraction ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "label, expected_donor",
    [
        ("MEL2 (13240-005)", "13240-005"),
        ("GBM11 (H4512 BT145)", "H4512 BT145"),
        ("CLL C (DFCI-5283)", "DFCI-5283"),
        ("OV1 (CP-594_v1)", "CP-594_v1"),
        ("just_a_label", "just_a_label"),  # no parens → use whole label
        ("", ""),  # empty input → empty donor
    ],
)
def test_donor_id_extraction(label, expected_donor):
    out = parse_cell_name("", attributed_sample_label=label)
    assert out.donor_id == expected_donor


# ── Uninformative / empty inputs ──────────────────────────────────────


@pytest.mark.parametrize(
    "raw",
    ["", "Other", "Cell found in tissue", "Unknown", "N/A", "n/a"],
)
def test_uninformative_cell_name_returns_empty_fields(raw):
    """Uninformative IEDB cell_name values shouldn't yield a fake
    cell_line_name or cell_type."""
    out = parse_cell_name(raw)
    assert out.cell_line_name == ""
    assert out.cell_line_input == ""
    assert out.cell_type == ""
    assert out.genetic_modification == ""
    assert out.raw_cell_name == raw


def test_uninformative_with_src_cell_line_true_marks_as_line():
    """When the obs build said src_cell_line=True but cell_name is
    "Other", preserve is_cell_line=True for downstream filtering."""
    out = parse_cell_name("Other", src_cell_line=True)
    assert out.is_cell_line is True
    assert out.cell_line_name == ""


# ── Unknown line + src_cell_line hint ────────────────────────────────


def test_unknown_line_with_src_cell_line_hint_preserves_input():
    """An unrecognized line name + src_cell_line=True preserves the
    raw input as cell_line_input so downstream can still distinguish.
    """
    out = parse_cell_name("SomeNovelLine-X42", src_cell_line=True)
    assert out.is_cell_line is True
    assert out.cell_line_name == ""  # not in registry
    assert out.cell_line_input == "SomeNovelLine-X42"


def test_unknown_string_without_src_hint_treated_as_cell_type():
    """Without src_cell_line=True and without a registry match, the
    parser treats the string as a primary-cell category label."""
    out = parse_cell_name("Lymph node cells")
    assert out.is_cell_line is False
    assert out.cell_type == "Lymph node cells"


# ── known_* helpers ───────────────────────────────────────────────────


def test_known_cell_lines_returns_sorted_canonical_names():
    lines = known_cell_lines()
    assert "HAP1" in lines
    assert "K562" in lines
    assert "HEK293T" in lines
    assert lines == sorted(lines)


def test_known_cell_types_includes_common_categories():
    types = set(known_cell_types())
    for expected in ("B cell", "Epithelial cell", "Myeloid cell", "Melanocyte"):
        assert expected in types


# ── End-to-end: real corpus top-N strings ────────────────────────────


def test_real_corpus_top_strings_decompose_correctly():
    """Smoke test against the top observed cell_name strings, including
    hybrids that mix line + type + (sometimes) genetic modification."""
    cases: list[tuple[str, dict]] = [
        (
            "C1R cells-B cell",
            {"is_cell_line": True, "cell_line_name": "C1R", "cell_type": "B cell"},
        ),
        (
            "K562-Myeloid cell",
            {"is_cell_line": True, "cell_line_name": "K562", "cell_type": "Myeloid cell"},
        ),
        (
            "HAP1 wildtype",
            {"is_cell_line": True, "cell_line_name": "HAP1", "genetic_modification": "wildtype"},
        ),
        (
            "HAP1 CALR KO",
            {"is_cell_line": True, "cell_line_name": "HAP1", "genetic_modification": "CALR KO"},
        ),
        (
            "HeLa cells-Epithelial cell",
            {"is_cell_line": True, "cell_line_name": "HeLa", "cell_type": "Epithelial cell"},
        ),
        (
            "B cell",
            {"is_cell_line": False, "cell_type": "B cell"},
        ),
        (
            "Other",
            {"is_cell_line": False, "cell_line_name": "", "cell_type": ""},
        ),
    ]
    for raw, expected in cases:
        out = parse_cell_name(raw)
        for field, expected_value in expected.items():
            actual = getattr(out, field)
            assert actual == expected_value, (
                f"{raw!r}.{field}: got {actual!r}, expected {expected_value!r}"
            )


# ── Dataclass shape ───────────────────────────────────────────────────


def test_cell_name_info_is_frozen_dataclass():
    out = parse_cell_name("K562")
    with pytest.raises((AttributeError, TypeError)):
        out.cell_line_name = "mutated"  # type: ignore[misc]


def test_cell_name_info_field_set_is_stable():
    """The set of fields on CellNameInfo is part of the public contract;
    changes need to bump callers (#262, scanner integration)."""
    expected_fields = {
        "is_cell_line",
        "cell_line_name",
        "cell_line_input",
        "cell_type",
        "donor_id",
        "genetic_modification",
        "raw_cell_name",
    }
    actual = set(CellNameInfo.__dataclass_fields__)
    assert actual == expected_fields
