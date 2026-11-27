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

"""The curated core top-level API + deprecated compatibility shims (#986 Phase 2b)."""

import importlib

import pytest

import hitlist
from hitlist import _DEPRECATED_API, _PUBLIC_API


def test_top_level_core_entry_points_importable():
    from hitlist import (  # noqa: F401
        DatasetError,
        InsufficientSpaceError,
        OfflineError,
        ProteomeIndex,
        audit_splits,
        build_observations,
        fetch,
        generate_ms_samples_table,
        generate_observations_table,
        generate_training_table,
        load_all_evidence,
        load_binding,
        load_lineage,
        load_ms_observations,
        load_observations,
        load_other_assays,
        prepare_affinity_training_data,
        prepare_ms_training_data,
        prepare_stability_training_data,
        sample_alleles_for_pmid,
    )

    assert callable(build_observations)
    assert callable(fetch)
    assert callable(load_observations)
    assert callable(generate_training_table)


def test_public_api_is_small_and_discoverable():
    assert "build_observations" in dir(hitlist)
    assert "load_observations" in hitlist.__all__
    assert "generate_training_table" in hitlist.__all__
    assert "EvidenceStore" not in hitlist.__all__
    assert "VersionedDatasetRegistry" not in hitlist.__all__
    # Guardrail (#986 Phase 2b): keep the top-level API focused (~37 core symbols).
    assert len(hitlist.__all__) <= 40
    assert set(_PUBLIC_API).isdisjoint(_DEPRECATED_API)


def test_unknown_attribute_raises_attributeerror():
    missing = "this_does_not_exist"
    with pytest.raises(AttributeError):
        getattr(hitlist, missing)


@pytest.mark.parametrize("name", sorted(_PUBLIC_API))
def test_every_public_api_entry_resolves_without_warning(name):
    """Every `_PUBLIC_API` key must import cleanly from the module it names."""
    attribute = getattr(hitlist, name)
    module = importlib.import_module(_PUBLIC_API[name], package="hitlist")
    assert attribute is getattr(module, name)
    assert name in hitlist.__all__
    assert name in dir(hitlist)


@pytest.mark.parametrize("name", sorted(_DEPRECATED_API))
def test_every_deprecated_api_entry_warns_and_resolves(name):
    """Legacy top-level names remain importable with DeprecationWarning during 2.x."""
    assert name not in hitlist.__all__
    assert name not in dir(hitlist)
    with pytest.warns(DeprecationWarning, match=r"deprecated.*hitlist 3\.0\.0"):
        attribute = getattr(hitlist, name)
    module = importlib.import_module(_DEPRECATED_API[name], package="hitlist")
    assert attribute is getattr(module, name)
