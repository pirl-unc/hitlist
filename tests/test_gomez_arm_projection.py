import pyarrow as pa
import pyarrow.parquet as pq

from hitlist.gomez_arm_projection import (
    ARM_DEFINITIONS,
    _arm_for_native,
    _project_context,
)


def test_native_source_arm_mapping_is_explicit():
    assert _arm_for_native("JY", "s3_method_comparison", "", "300ms_JY_None") == (
        "gomez:jy_wild_type"
    )
    assert _arm_for_native("JY", "s5_spike_experiment", "S1", "JY_S") == ("gomez:jy_spike_s1")
    assert _arm_for_native("Raji", "s5_spike_experiment", "S2", "Raji_S") == ("gomez:raji_spike_s2")
    assert len(ARM_DEFINITIONS) == 9


def test_context_projection_is_source_specific(tmp_path):
    parent = tmp_path / "parent"
    (parent / "typed").mkdir(parents=True)
    schema = pa.schema(
        [
            ("observation_id", pa.string()),
            ("arm_id", pa.string()),
            ("arm_link_status", pa.string()),
            ("sample_label", pa.string()),
            ("sample_group", pa.string()),
            ("sample_attribution", pa.string()),
            ("condition_id", pa.string()),
            ("condition_status", pa.string()),
            ("condition_evidence", pa.string()),
            ("condition_reference", pa.string()),
            ("condition_material", pa.string()),
            ("arm_resolution", pa.string()),
            ("sample_mhc", pa.string()),
        ]
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"observation_id": "gomez:one", **dict.fromkeys(schema.names[1:])},
                {"observation_id": "other:one", **dict.fromkeys(schema.names[1:])},
            ],
            schema=schema,
        ),
        parent / "typed/assay_context.parquet",
    )
    output = tmp_path / "projection.parquet"
    counts = _project_context(
        parent,
        output,
        [
            {
                "derived_observation_id": "gomez:one",
                "material_id": "JY",
                "n_supported_arms": 1,
                "supported_arm_ids_json": '["gomez:jy_wild_type"]',
                "source_phases": "s3_method_comparison",
                "n_source_memberships": 1,
                "n_source_result_rows": 1,
                "unique_arm_id": "gomez:jy_wild_type",
                "arm_projection_status": "unique_source_arm",
                "projection_basis": "test",
            }
        ],
    )
    assert counts["n_changed_context_rows"] == 1
    projected = pq.read_table(output).to_pylist()
    assert len(projected) == 1
    assert projected[0]["arm_id"] == "gomez:jy_wild_type"
    assert projected[0]["sample_attribution"] == "source_unique_arm"
