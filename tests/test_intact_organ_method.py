"""A reported search policy can support missing fields; missingness alone cannot."""

import json

import openpyxl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hitlist import intact_degradomics as source
from hitlist import intact_organ_method as method
from hitlist import intact_organ_transfer as strict
from hitlist.evidence_recipe import RecipeError, verify_recipe_package
from hitlist.evidence_store import EvidenceStore
from hitlist.provenance import file_digest
from tests import test_intact_degradomics, test_intact_organ_transfer

original_source = test_intact_degradomics.original
organ_store = test_intact_organ_transfer.organ_store
read = test_intact_organ_transfer.rows
ARTICLE = """<article><front><article-id>10.1073/pnas.2419607122</article-id></front>
<body><sec id="s6"><title>Methods</title><sec id="s15">
<title>ID-MS Data Analysis and Statistics.</title><p>
ClipsMS P37840 10 ppm 4 amino acids b-, y-, or by-.
No additional modifications were included.</p></sec></sec></body></article>"""


@pytest.fixture
def method_store(organ_store, original_source, tmp_path):
    assets, _, contract = original_source
    workbook = assets / list(contract["search_workbooks"])[2]
    book = openpyxl.load_workbook(workbook)
    for sheet in book:
        for row in sheet.iter_rows(min_row=2):
            row[10].value = None
    book.save(workbook)
    (assets / "article.xml").write_text(ARTICLE)
    contract["assets"].append(
        {
            "filename": "article.xml",
            "source_url": "https://example.invalid/article.xml",
            "role": "synthetic_software_fixture",
            "license_status": "test_fixture",
            **file_digest(assets / "article.xml"),
        }
    )
    for asset in contract["assets"]:
        asset.update(file_digest(assets / asset["filename"]))
    source.CONTRACT.write_text(json.dumps(contract))
    source.write_intact_degradomics_package(assets, tmp_path / "method_source")
    return EvidenceStore.open(tmp_path / "method_source")


def recipe():
    return {"schema_version": 1, "name": "method_test", "registered_recipe": method.RECIPE_ID}


def test_method_policy_is_explicit_provenance_and_preserves_raw_nulls(method_store, tmp_path):
    original_hash = file_digest(method_store.directory / "manifest.json")
    output = tmp_path / "method_recipe"
    method_store.materialize(recipe(), output)
    manifest = verify_recipe_package(output, source_package=method_store.directory)
    assert not manifest["coverage"]["support_passed"]  # Small software fixture.
    assert file_digest(method_store.directory / "manifest.json") == original_hash
    policies = read(output / "typed", "search_policies")
    assert len(policies) == 1
    policy = policies[0]
    assert policy["source_record_id"] is None
    assert policy["assertion_value"] == "none"
    assert policy["assertion_status"] == "reported_method"
    assert (
        policy["source_asset_sha256"] == file_digest(output / "source_assets/article.xml")["sha256"]
    )
    examples = read(output / "typed", "examples")
    assert len(examples) == 35
    assert {r["modification_policy"] for r in examples} == {
        "reported_method_no_additional_modifications"
    }
    assert all(
        json.loads(r["modification_assertion_ids_json"]) == [policy["claim_id"]] for r in examples
    )
    source_rows = read(output / "source_tables", "assignments")
    lookup = {r["assignment_id"]: r for r in source_rows}
    assert all(
        json.loads(lookup[r["assignment_id"]]["modifications_json"])["Unlocalized Mod"] is None
        for r in examples
    )
    covariates = read(output / "typed", "covariates")
    assert len(covariates) == len(examples)
    assert not set(covariates[0]) & method.LABEL_FIELDS
    assert "modification_assertion_ids_json" in covariates[0]
    assert file_digest(output / "source_tables/assignments.parquet") == file_digest(
        method_store.directory / source.ROOT / "typed/assignments.parquet"
    )
    strict.compile_kinetics(method_store.directory / source.ROOT / "typed", tmp_path / "strict")
    assert read(tmp_path / "strict", "examples") == []  # Original gate unchanged.


@pytest.mark.parametrize(
    "replacement",
    [
        ARTICLE.replace(
            "No additional modifications were included", "Search settings were not reported"
        ),
        ARTICLE.replace("10.1073/pnas.2419607122", "other.study"),
        ARTICLE.replace('id="s15"', 'id="different"'),
        ARTICLE.replace("P37840", "unrelated protein"),
    ],
)
def test_blank_fields_need_the_qualified_study_declaration(method_store, tmp_path, replacement):
    # Mutate a compiler-input software fixture; do not claim this is a verified
    # source package after mutation. Public-package tampering is checked below.
    (method_store.directory / source.ROOT / "source_assets/article.xml").write_text(replacement)
    method.compile_kinetics(
        method_store.directory / source.ROOT / "typed", tmp_path / "unsupported"
    )
    assert not read(tmp_path / "unsupported", "examples")
    assert not read(tmp_path / "unsupported", "search_policies")


@pytest.mark.parametrize(
    "values",
    [
        {"Localized Mod": "1", "Term Mod": "0", "Unlocalized Mod": None},
        {"Localized Mod": "0", "Term Mod": "42", "Unlocalized Mod": None},
        {"Localized Mod": "0", "Term Mod": "0", "Unlocalized Mod": "Oxidation"},
        {"Localized Mod": None, "Term Mod": "0", "Unlocalized Mod": None},
    ],
)
def test_contradictory_or_missing_other_fields_cannot_be_overridden(values):
    eligible, metadata = method.modification_eligibility(
        {"modifications_json": json.dumps(values)}, [{"claim_id": "reported-method"}]
    )
    assert not eligible
    assert metadata["modification_policy"] == "unsupported"


def test_original_explicit_row_does_not_need_method_inference():
    eligible, metadata = method.modification_eligibility(
        {
            "modifications_json": json.dumps(
                {"Localized Mod": "0", "Term Mod": "0", "Unlocalized Mod": "None"}
            )
        },
        [],
    )
    assert eligible and metadata == {
        "modification_policy": "explicit_row",
        "modification_assertion_ids_json": "[]",
    }


@pytest.mark.parametrize("mutation", ["article", "policy"])
def test_rehashed_method_evidence_tampering_is_rejected(method_store, tmp_path, mutation):
    output = tmp_path / "recipe"
    method_store.materialize(recipe(), output)
    if mutation == "article":
        path = output / "source_assets/article.xml"
        path.write_text(ARTICLE.replace("No additional modifications", "Additional modifications"))
    else:
        path = output / "typed/search_policies.parquet"
        table = pq.read_table(path)
        records = table.to_pylist()
        records[0]["assertion_status"] = "invented"
        pq.write_table(pa.Table.from_pylist(records, schema=table.schema), path)
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["files"][str(path.relative_to(output))] = file_digest(path)
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RecipeError, match=r"source asset differs|rows differ"):
        verify_recipe_package(output, source_package=method_store.directory)
