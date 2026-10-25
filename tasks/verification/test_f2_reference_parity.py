"""Run explicitly: compare F2 fixture outputs against the pinned pre-change source."""

import hashlib
import subprocess
import types

import pyarrow.parquet as pq

from hitlist import intact_degradomics as source
from hitlist import intact_kinetics as current
from tests import test_intact_degradomics, test_intact_kinetics

original_source = test_intact_degradomics.original
temporal = test_intact_kinetics.temporal


def test_f2_reference_output_parity(temporal, tmp_path):
    code = subprocess.check_output([
        "git", "show", "d76c225a0aa741a1a1b0aee707bff328ca467c7c:hitlist/intact_kinetics.py"
    ])
    reference = types.ModuleType("hitlist._frozen_f2_reference")
    exec(compile(code, "pinned_F2_reference.py", "exec"), reference.__dict__)
    assert reference.CONTRACT == current.CONTRACT
    assert reference.OUTPUT_SCHEMAS == current.OUTPUT_SCHEMAS
    tables = temporal.directory / source.ROOT / "typed"
    old_summary = reference.compile_kinetics(tables, tmp_path / "old")
    new_summary = current.compile_kinetics(tables, tmp_path / "new")
    assert old_summary == new_summary
    for name in reference.OUTPUT_SCHEMAS:
        assert pq.read_table(tmp_path / "old" / (name + ".parquet")).equals(
            pq.read_table(tmp_path / "new" / (name + ".parquet")))
    print("Pinned source SHA256:", hashlib.sha256(code).hexdigest())
