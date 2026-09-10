"""Every packaged YAML loads through one duplicate-key-rejecting loader (#454).

PyYAML keeps the last value for a duplicated mapping key and discards the
first, silently. ``load_curation_yaml`` turns that into a load-time error;
these tests pin the helper's behaviour, prove every packaged file is clean
under it, and guard against a new loader bypassing it.
"""

from __future__ import annotations

import re
from importlib.resources import files
from pathlib import Path

import pytest
import yaml

import hitlist
from hitlist.curation_yaml import UniqueKeyLoader, load_curation_yaml

PACKAGE_ROOT = Path(hitlist.__file__).resolve().parent
PACKAGED_YAML = sorted(PACKAGE_ROOT.glob("data/**/*.yaml"))

# Every way PyYAML can be asked to parse a document. The leaf module is the
# only place any of them may appear.
_DIRECT_YAML_PARSE = re.compile(r"\byaml\.(?:safe_load|load|full_load|unsafe_load|load_all)\(")


def test_duplicate_top_level_key_is_rejected(tmp_path):
    path = tmp_path / "dup.yaml"
    path.write_text("synonyms: [a]\nsynonyms: [b]\n")
    with pytest.raises(yaml.constructor.ConstructorError, match="duplicate key 'synonyms'"):
        load_curation_yaml(path)


def test_duplicate_nested_key_is_rejected(tmp_path):
    """The shape that loses data on a real file: one record, one key twice."""
    path = tmp_path / "dup.yaml"
    path.write_text("- name: C1R\n  endogenous_alleles: [HLA-C*04:01]\n  endogenous_alleles: []\n")
    with pytest.raises(yaml.constructor.ConstructorError, match="endogenous_alleles"):
        load_curation_yaml(path)


def test_plain_safe_load_would_have_kept_the_last_value():
    """Documents why the guard exists: this is PyYAML's default behaviour."""
    assert yaml.safe_load("k: first\nk: last\n") == {"k": "last"}


def test_accepts_str_path_and_traversable(tmp_path):
    path = tmp_path / "ok.yaml"
    path.write_text("gene_sets:\n  demo:\n    genes: [PRAME]\n")
    expected = {"gene_sets": {"demo": {"genes": ["PRAME"]}}}

    assert load_curation_yaml(str(path)) == expected
    assert load_curation_yaml(path) == expected
    traversable = files("hitlist.data") / "gene_sets.yaml"
    assert "gene_sets" in load_curation_yaml(traversable)


def test_empty_document_is_none_like_safe_load(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("")
    assert load_curation_yaml(path) is None


def test_packaged_yaml_inventory_is_complete():
    """The parametrized check below is only as good as the glob feeding it."""
    names = {p.relative_to(PACKAGE_ROOT).as_posix() for p in PACKAGED_YAML}
    assert {
        "data/pmid_overrides.yaml",
        "data/cell_lines.yaml",
        "data/monoallelic_lines.yaml",
        "data/tissue_categories.yaml",
        "data/condition_vocabulary.yaml",
        "data/line_expression_anchors.yaml",
        "data/supplementary.yaml",
        "data/data_assets.yaml",
        "data/gene_sets.yaml",
        "data/bulk_proteomics/sources.yaml",
        "data/line_expression/sources.yaml",
    } <= names


@pytest.mark.parametrize(
    "path", PACKAGED_YAML, ids=[p.relative_to(PACKAGE_ROOT).as_posix() for p in PACKAGED_YAML]
)
def test_every_packaged_yaml_is_duplicate_free(path):
    """A duplicate key anywhere in a shipped registry fails here, not in an audit."""
    load_curation_yaml(path)


def test_no_module_parses_yaml_without_the_guard():
    """The next registry gets the guard by default, not by remembering."""
    offenders = []
    for module in sorted(PACKAGE_ROOT.rglob("*.py")):
        if module.name == "curation_yaml.py":
            continue
        for lineno, line in enumerate(module.read_text().splitlines(), 1):
            if _DIRECT_YAML_PARSE.search(line):
                offenders.append(f"{module.relative_to(PACKAGE_ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "parse YAML through hitlist.curation_yaml.load_curation_yaml so duplicate keys "
        "are rejected:\n" + "\n".join(offenders)
    )


def test_loader_class_keeps_its_original_home():
    """``UniqueKeyLoader`` was public on ``hitlist.curation`` before #454."""
    from hitlist.curation import UniqueKeyLoader as legacy_name

    assert legacy_name is UniqueKeyLoader
