import re

import pandas as pd
import pytest

import hitlist
from hitlist.binding_assay_context import (
    CENSOR_REVIEW_COLUMNS,
    CONTEXT_AVAILABILITY_FIELDS,
    binding_protocol_registry_contract,
    review_binding_censoring,
    summarize_binding_censoring_reviews,
)
from hitlist.binding_context_availability import (
    REPLICATE_GROUP_COLUMNS,
    _endpoint_family,
    _status_from_claim,
    audit_binding_context_availability,
    build_binding_replicate_groups,
    classify_binding_endpoint_family,
    summarize_binding_context_availability,
)
from hitlist.evidence_bindings import materialized_bindings
from hitlist.evidence_catalog import SCHEMA_VERSION, describe_evidence
from hitlist.evidence_package import package_family
from hitlist.model_training import prepare_affinity_training_data, prepare_stability_training_data
from hitlist.schema_conventions import canonical_assay_columns, normalize_assay_frame


@pytest.mark.parametrize(
    "endpoint,family",
    [
        ("half maximal inhibitory concentration (IC50)", "literal_ic50"),
        ("dissociation constant KD (~IC50)", "kd_ic50"),
        ("equilibrium dissociation constant", "other_binding"),
    ],
)
def test_binding_endpoint_families_stay_separate(endpoint, family):
    assert _endpoint_family(endpoint) == family


def test_study_claim_status_never_becomes_row_value():
    claim = {"claim_status": "reported_study_scope"}
    assert _status_from_claim(claim, reviewed=True) == "study_scope_only"
    assert (
        _status_from_claim(
            {"claim_status": "reported_alternatives_per_assay_unresolved"}, reviewed=True
        )
        == "unresolved"
    )
    assert _status_from_claim(claim, reviewed=False) == "unreviewed"


def test_binding_context_availability_is_a_registered_successor_relation():
    envelope = {
        "schema_version": 14,
        "parent_kind": "binding_review",
        "revision": "binding_assay_context_availability_v1",
    }
    assert package_family(envelope) == "binding_context_availability"
    assert SCHEMA_VERSION == 40
    fields = {
        field["name"] for field in describe_evidence("binding_context_availability")["fields"]
    }
    assert {"observation_id", "temperature_status", "kd_conversion_status"} <= fields
    assert "binding_context_availability" in materialized_bindings(14)


def test_protocol_registry_contract_and_public_exports():
    contract = binding_protocol_registry_contract()
    assert contract["revision"] == "binding_assay_protocol_registry_v1"
    sha_re = re.compile(r"^[0-9a-f]{64}$")
    pmids = set()
    for proto in contract["protocols"]:
        pmids.add(proto["source"]["pmid"])
        assert sha_re.match(proto["source"]["source_xml_sha256"])
        assert set(proto["field_availability"]) == set(CONTEXT_AVAILABILITY_FIELDS)
    assert {28481943, 24130917, 30828331, 33521695} <= pmids

    # Deprecated top-level shims still resolve with DeprecationWarning during 2.x (#986)
    for fn_name in (
        "binding_protocol_registry_contract",
        "review_binding_censoring",
        "summarize_binding_censoring_reviews",
        "classify_binding_endpoint_family",
        "build_binding_replicate_groups",
        "audit_binding_context_availability",
        "summarize_binding_context_availability",
        "write_binding_context_availability_package",
        "verify_binding_context_availability_package",
    ):
        with pytest.warns(DeprecationWarning):
            assert getattr(hitlist, fn_name) is not None, f"missing deprecated shim {fn_name}"


@pytest.mark.parametrize(
    "endpoint,expected",
    [
        ("half maximal inhibitory concentration (IC50)", "literal_ic50"),
        ("dissociation constant KD (~IC50)", "kd_ic50"),
        ("dissociation constant KD (~EC50)", "kd_ec50"),
        ("dissociation constant KD", "literal_kd"),
        ("half maximal effective concentration (EC50)", "ec50"),
        ("half life", "half_life"),
        ("stability_t_half", "half_life"),
        ("50% dissociation temperature", "tm"),
        ("off rate", "off_rate"),
        ("on rate", "on_rate"),
        ("qualitative binding", "qualitative_binding"),
        ("unknown metric", "other_binding"),
    ],
)
def test_symmetric_endpoint_family_classification(endpoint, expected):
    assert classify_binding_endpoint_family(endpoint) == expected


def test_review_binding_censoring_flags_heaps_without_relabeling_raw_values():
    """#721: separate blank vs explicit '=', flag suspected upper-limit heaps, never mutate raw bounds."""
    frame = pd.DataFrame(
        [
            # reference/1005810-style negative upper heap at 20,000 nM (3 exact '=' Negative rows + 1 lower binder)
            {
                "reference_iri": "http://www.iedb.org/reference/1005810",
                "pmid": "",
                "mhc_restriction": "HLA-A*01:01",
                "mhc_class": "I",
                "assay_method": "purified MHC/direct/fluorescence",
                "response_measured": "dissociation constant KD (~EC50)",
                "quantitative_value": 150.0,
                "measurement_units": "nM",
                "measurement_inequality": "=",
                "qualitative_measurement": "Positive",
            },
            *[
                {
                    "reference_iri": "http://www.iedb.org/reference/1005810",
                    "pmid": "",
                    "mhc_restriction": "HLA-A*01:01",
                    "mhc_class": "I",
                    "assay_method": "purified MHC/direct/fluorescence",
                    "response_measured": "dissociation constant KD (~EC50)",
                    "quantitative_value": 20000.0,
                    "measurement_units": "nM",
                    "measurement_inequality": "=",
                    "qualitative_measurement": "Negative",
                }
                for _ in range(3)
            ],
            # Same-value right-bound heap: exact 50,000 beside explicit '> 50,000'
            {
                "reference_iri": "http://www.iedb.org/reference/9990001",
                "pmid": "99999999",
                "mhc_restriction": "HLA-A*02:01",
                "mhc_class": "I",
                "assay_method": "purified MHC/competitive/radioactivity",
                "response_measured": "half maximal inhibitory concentration (IC50)",
                "quantitative_value": 50000.0,
                "measurement_units": "nM",
                "measurement_inequality": "",
                "qualitative_measurement": "Negative",
            },
            {
                "reference_iri": "http://www.iedb.org/reference/9990001",
                "pmid": "99999999",
                "mhc_restriction": "HLA-A*02:01",
                "mhc_class": "I",
                "assay_method": "purified MHC/competitive/radioactivity",
                "response_measured": "half maximal inhibitory concentration (IC50)",
                "quantitative_value": 50000.0,
                "measurement_units": "nM",
                "measurement_inequality": ">",
                "qualitative_measurement": "Negative",
            },
            # Plain blank inequality (not a heap) vs repeated round positive values (not a negative heap!)
            {
                "reference_iri": "http://www.iedb.org/reference/9990002",
                "pmid": "99999998",
                "mhc_restriction": "HLA-B*07:02",
                "mhc_class": "I",
                "assay_method": "purified MHC/competitive/radioactivity",
                "response_measured": "half maximal inhibitory concentration (IC50)",
                "quantitative_value": 20000.0,
                "measurement_units": "nM",
                "measurement_inequality": "",
                "qualitative_measurement": "Positive-Low",
            },
            # PMID 30828331 Class II documented upper ceiling (> 100,000 nM)
            {
                "reference_iri": "http://www.iedb.org/reference/30828331",
                "pmid": "30828331",
                "mhc_restriction": "HLA-DRB1*01:01",
                "mhc_class": "II",
                "assay_method": "purified MHC/competitive/fluorescence",
                "response_measured": "dissociation constant KD (~IC50)",
                "quantitative_value": 100000.0,
                "measurement_units": "nM",
                "measurement_inequality": ">",
                "qualitative_measurement": "Negative",
            },
        ]
    )

    reviewed = review_binding_censoring(frame)
    # Raw columns never altered
    assert reviewed["measurement_inequality"].tolist() == frame["measurement_inequality"].tolist()
    assert reviewed["quantitative_value"].tolist() == frame["quantitative_value"].tolist()
    assert set(CENSOR_REVIEW_COLUMNS) <= set(reviewed.columns)

    assert reviewed.loc[0, "censor_review_status"] == "explicit_exact_unflagged"
    for idx in (1, 2, 3):
        assert (
            reviewed.loc[idx, "censor_review_status"] == "suspected_upper_limit_heap_negative_max"
        )
        assert reviewed.loc[idx, "censor_heap_n_rows"] == 3
        assert reviewed.loc[idx, "censor_heap_negative_fraction"] == pytest.approx(1.0)
        assert reviewed.loc[idx, "limit_applicability_status"] == "suspected_limit_unconfirmed"
    assert reviewed.loc[4, "censor_review_status"] == "suspected_upper_limit_heap_same_value_bound"
    assert reviewed.loc[5, "censor_review_status"] == "explicit_right_censored"
    assert reviewed.loc[6, "censor_review_status"] == "blank_inequality_unreviewed"
    assert reviewed.loc[7, "censor_review_status"] == "explicit_right_censored"
    assert reviewed.loc[7, "limit_applicability_status"] == "source_confirmed_study_limit"

    summary = summarize_binding_censoring_reviews(reviewed)
    assert summary["n_rows"] == len(frame)
    assert summary["n_suspected_heap_rows"] == 4
    assert summary["n_suspected_heap_cells"] == 2
    assert summary["n_blank_inequality_unreviewed_rows"] == 1
    assert summary["n_explicit_right_censored_rows"] == 2


def test_audit_binding_context_availability_symmetric_across_assays_and_no_row_multiplication():
    """#785/#721: symmetric 1:1 audit across IC50, KD(~IC50), stability, IEDB/CEDAR/NetMHCstabpan."""
    frame = pd.DataFrame(
        [
            # PMID 24130917 HLA-A*02:01 fluorescence polarization (exact tracer ALMDKVLKV, RT)
            {
                "source": "iedb",
                "pmid": "24130917",
                "reference_iri": "http://www.iedb.org/reference/24130917",
                "peptide": "KLNDWDFVV",
                "mhc_restriction": "HLA-A*02:01",
                "mhc_class": "I",
                "assay_method": "purified MHC/competitive/fluorescence",
                "response_measured": "half maximal inhibitory concentration (IC50)",
                "quantitative_value": 110.0,
                "measurement_units": "nM",
                "measurement_inequality": "=",
                "qualitative_measurement": "Positive",
                "assay_comments": "",
            },
            # Duplicate replicate of the same pMHC in PMID 24130917 -> same replicate_group_id, no row multiplication
            {
                "source": "iedb",
                "pmid": "24130917",
                "reference_iri": "http://www.iedb.org/reference/24130917",
                "peptide": "KLNDWDFVV",
                "mhc_restriction": "HLA-A*02:01",
                "mhc_class": "I",
                "assay_method": "purified MHC/competitive/fluorescence",
                "response_measured": "half maximal inhibitory concentration (IC50)",
                "quantitative_value": 125.0,
                "measurement_units": "nM",
                "measurement_inequality": "=",
                "qualitative_measurement": "Positive",
                "assay_comments": "replicate 2",
            },
            # PMID 24130917 radioligand panel -> routes to s2n_radioligand (temperature unresolved)
            {
                "source": "iedb",
                "pmid": "24130917",
                "reference_iri": "http://www.iedb.org/reference/24130917",
                "peptide": "SYAMCTNTF",
                "mhc_restriction": "HLA-A*24:02",
                "mhc_class": "I",
                "assay_method": "purified MHC/competitive/radioactivity",
                "response_measured": "dissociation constant KD (~IC50)",
                "quantitative_value": 14.0,
                "measurement_units": "nM",
                "measurement_inequality": "=",
                "qualitative_measurement": "Positive",
                "assay_comments": "",
            },
            # CEDAR unreviewed IC50 row
            {
                "source": "cedar",
                "pmid": "39999999",
                "reference_iri": "http://www.iedb.org/reference/39999999",
                "peptide": "SIINFEKL",
                "mhc_restriction": "HLA-A*02:01",
                "mhc_class": "I",
                "assay_method": "purified MHC/competitive/radioactivity",
                "response_measured": "half maximal inhibitory concentration (IC50)",
                "quantitative_value": 50.0,
                "measurement_units": "nM",
                "measurement_inequality": "",
                "qualitative_measurement": "Positive",
                "assay_comments": "",
            },
            # NetMHCstabpan stability half-life row (symmetric schema!)
            {
                "source": "netmhcstabpan",
                "pmid": "27106578",
                "reference_iri": "",
                "peptide": "GILGFVFTL",
                "mhc_restriction": "HLA-A*02:01",
                "mhc_class": "I",
                "assay_method": "scintillation proximity assay",
                "response_measured": "half life",
                "quantitative_value": 18.5,
                "measurement_units": "h",
                "measurement_inequality": "=",
                "qualitative_measurement": "Positive",
                "assay_comments": "",
            },
        ]
    )

    audited = audit_binding_context_availability(frame)
    assert len(audited) == len(frame)
    # Protocol routing by (pmid, method, allele)
    assert (
        audited.loc[0, "protocol_id"] == "iedb:pmid:24130917:binding_protocol:s2o_hla_a2_ig_fp:v1"
    )
    assert audited.loc[0, "tracer_status"] == "study_scope_only"
    assert audited.loc[0, "temperature_status"] == "study_scope_only"
    assert audited.loc[0, "kd_conversion_status"] == "not_performed"

    assert audited.loc[2, "protocol_id"] == "iedb:pmid:24130917:binding_protocol:s2n_radioligand:v1"
    assert audited.loc[2, "endpoint_family"] == "kd_ic50"
    assert audited.loc[2, "temperature_status"] == "unresolved"
    assert audited.loc[2, "tracer_status"] == "unreported"

    # Unreviewed CEDAR & stability rows
    assert audited.loc[3, "source_review_status"] == "not_source_reviewed"
    assert audited.loc[3, "context_scope"] == "row_native_only"
    assert audited.loc[4, "endpoint_family"] == "half_life"
    assert audited.loc[4, "kd_conversion_status"] == "not_applicable"

    # Replicate group grain prevents row multiplication
    assert audited.loc[0, "replicate_group_id"] == audited.loc[1, "replicate_group_id"]
    assert audited.loc[0, "replicate_group_n_observations"] == 2
    assert audited.loc[2, "replicate_group_n_observations"] == 1

    groups = build_binding_replicate_groups(frame)
    assert tuple(groups.columns) == REPLICATE_GROUP_COLUMNS
    assert len(groups) == 4
    dup = groups[groups["n_observations"] == 2].iloc[0]
    assert dup["aggregation_policy"] == "multi_observation_do_not_multiply_weights"
    assert dup["value_min"] == pytest.approx(110.0)
    assert dup["value_max"] == pytest.approx(125.0)

    summary = summarize_binding_context_availability(audited)
    assert summary["n_observations"] == 5
    assert summary["n_replicate_groups"] == 4
    assert summary["n_multi_observation_groups"] == 1
    assert summary["by_endpoint_family"] == {"half_life": 1, "kd_ic50": 1, "literal_ic50": 3}
    assert summary["by_source_dataset"] == {"cedar": 1, "iedb": 3, "netmhcstabpan": 1}
    assert summary["kd_conversion_performed"] is False
    assert summary["native_rows_unchanged"] is True

    from hitlist.schema_conventions import ASSAY_SCHEMA_CONVENTIONS

    # Canonical schema normalization works symmetrically
    norm_audit = normalize_assay_frame(audited, assay_type="binding_context_audit")
    assert "measurement_value" in norm_audit.columns
    assert "binding_context_audit" in ASSAY_SCHEMA_CONVENTIONS
    assert "binding_replicate_group" in ASSAY_SCHEMA_CONVENTIONS
    assert "endpoint_family" in canonical_assay_columns("binding_context_audit")


def test_model_training_exports_carry_censor_review_columns_symmetrically():
    """Both affinity and stability training exports carry CENSOR_REVIEW_COLUMNS without altering labels."""
    base = pd.DataFrame(
        [
            {
                "evidence_kind": "binding",
                "source": "iedb",
                "pmid": "30828331",
                "reference_iri": "http://www.iedb.org/reference/30828331",
                "assay_iri": "http://www.iedb.org/assay/1",
                "peptide": "SIINFEKL",
                "mhc_restriction": "HLA-A*02:01",
                "mhc_class": "I",
                "mhc_species": "human",
                "assay_method": "purified MHC/competitive/radioactivity",
                "response_measured": "half maximal inhibitory concentration (IC50)",
                "quantitative_value": 120.0,
                "measurement_units": "nM",
                "measurement_inequality": "=",
                "qualitative_measurement": "Positive",
            },
            {
                "evidence_kind": "binding",
                "source": "iedb",
                "pmid": "27106578",
                "reference_iri": "http://www.iedb.org/reference/27106578",
                "assay_iri": "http://www.iedb.org/assay/2",
                "peptide": "GILGFVFTL",
                "mhc_restriction": "HLA-A*02:01",
                "mhc_class": "I",
                "mhc_species": "human",
                "assay_method": "scintillation proximity assay",
                "response_measured": "half life",
                "quantitative_value": 12.5,
                "measurement_units": "h",
                "measurement_inequality": "=",
                "qualitative_measurement": "Positive",
            },
        ]
    )
    aff = prepare_affinity_training_data(base)
    stab = prepare_stability_training_data(base)
    assert len(aff.accepted) == 1
    assert len(stab.accepted) == 1
    for sel in (aff, stab):
        for col in CENSOR_REVIEW_COLUMNS:
            assert col in sel.accepted.columns
        assert sel.accepted.iloc[0]["censor_review_status"] == "explicit_exact_unflagged"
