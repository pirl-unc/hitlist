"""Explicit local comparison against the unchanged strict-F3 implementation."""
import subprocess
import types
import pyarrow.parquet as pq
from hitlist import intact_degradomics as source
from hitlist import intact_organ_transfer as current
from tests import test_intact_degradomics, test_intact_organ_transfer
original_source = test_intact_degradomics.original
organ_store = test_intact_organ_transfer.organ_store

def test_f3_reference_output_parity(organ_store, tmp_path):
    code = subprocess.check_output(["git", "show", "7c427e390daafdfb457581c511de88c82af3efbd:hitlist/intact_organ_transfer.py"])
    reference = types.ModuleType("hitlist._frozen_f3_reference")
    exec(compile(code, "pinned_F3_reference.py", "exec"), reference.__dict__)
    assert reference.CONTRACT == current.CONTRACT
    assert reference.OUTPUT_SCHEMAS == current.OUTPUT_SCHEMAS
    tables = organ_store.directory / source.ROOT / "typed"
    assert reference.compile_kinetics(tables, tmp_path / "old") == current.compile_kinetics(tables, tmp_path / "new")
    for name in reference.OUTPUT_SCHEMAS:
        assert pq.read_table(tmp_path / "old" / (name + ".parquet")).equals(pq.read_table(tmp_path / "new" / (name + ".parquet")))
