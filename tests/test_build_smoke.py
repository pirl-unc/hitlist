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

"""End-to-end ``build_observations`` smoke test that runs in CI.

The full production build reads ~15 GB of IEDB/CEDAR exports + proteome data
that aren't available in CI — but the bulk of the build *orchestration*
(scanner → ``classify_ms_row`` → dedup → supplement merge → categorical
compression → atomic parquet write, plus the packaged bulk-proteomics and
line-expression indexes) runs entirely off **packaged data** plus a tiny
synthetic IEDB CSV, with ``build_mappings=False`` skipping the only
proteome/pyensembl-dependent step.

This gives the data build real CI coverage (#176/#63) without shipping any
source data, and guards the build-time columns (#261 ``cell_type`` split,
#263 categoricals) against regressions in a true build rather than just the
scanner unit tests. Marked ``integration`` so it runs on the single
build-heavy CI job, not every Python version.
"""

from __future__ import annotations

import csv

import pandas as pd
import pytest

from hitlist import downloads

# IEDB "field" header (row 2); row 1 is the category grouping row. Mirrors the
# 21-column subset the scanner resolves by name (see tests/test_scanner.py).
_IEDB_FIELD_HEADER = [
    "Assay IRI",
    "Reference IRI",
    "PMID",
    "Submission ID",
    "Title",
    "Epitope | Name",
    "Epitope | Source Organism",
    "Epitope | Species",
    "Host",
    "Host Age",
    "Host | Process Type",
    "Host | Disease",
    "Host | Disease Stage",
    "Antigen Processing Comments",
    "Qualitative Measurement",
    "Assay Comments",
    "Effector Cells | Source Tissue",
    "Effector Cells | Cell Name",
    "Assay | Culture Condition",
    "MHC Restriction | Name",
    "MHC Allele Class",
]


def _write_synthetic_iedb(path) -> None:
    rows = []
    for i in range(20):
        r = [""] * 21
        r[0] = f"http://iedb.org/assay/{1000 + i}"
        r[1] = f"http://iedb.org/reference/{i % 5}"
        r[2] = str(33858848 + i % 3)
        r[5] = f"PEPTIDE{i:03d}AA"  # Epitope | Name
        r[6] = "Homo sapiens"
        r[7] = "Homo sapiens"
        # Alternate a hybrid cell-line row and a pure cell-type row so the
        # build exercises the #261 cell_name → cell_line_name + cell_type split.
        if i % 2:
            r[17] = "K562-Myeloid cell"
            r[18] = "Cell Line / Clone"
        else:
            r[17] = "B cell"
            r[18] = "Direct Ex Vivo"
        r[19] = "HLA-A*02:01"
        r[20] = "I"
        rows.append(r)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] * 21)  # category header row
        w.writerow(_IEDB_FIELD_HEADER)  # field header row
        w.writerows(rows)


@pytest.mark.integration
def test_build_observations_from_packaged_data(tmp_path, monkeypatch):
    """A full ``build_observations(build_mappings=False)`` produces a
    well-formed ``observations.parquet`` from packaged data + a synthetic
    IEDB CSV, with no network and no proteome dependency."""
    from hitlist.builder import build_observations

    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path)
    csv_path = tmp_path / "iedb.csv"
    _write_synthetic_iedb(csv_path)
    downloads.register("iedb", csv_path)

    out = build_observations(
        build_mappings=False,
        with_flanking=False,
        fetch_missing_proteomes=False,
        force=True,
    )

    assert out.exists()
    df = pd.read_parquet(out)
    assert len(df) > 0

    # Core schema present after a real build.
    assert {"peptide", "mhc_restriction", "source", "cell_line_name", "cell_type"}.issubset(
        df.columns
    )

    # #261: the synthetic hybrid "K562-Myeloid cell" row survives the full
    # build and is split into a clean line + cell type.
    hybrid = df[df["cell_name"] == "K562-Myeloid cell"]
    assert not hybrid.empty
    assert set(hybrid["cell_line_name"]) == {"K562"}
    assert set(hybrid["cell_type"]) == {"Myeloid cell"}

    # #263: low-cardinality columns are categorical on the built parquet.
    assert isinstance(df["mhc_restriction"].dtype, pd.CategoricalDtype)
    assert isinstance(df["cell_type"].dtype, pd.CategoricalDtype)


# Two tiny proteins for a UniProt-registry species (Sarcophilus harrisii →
# UP000007648). The test peptides below are substrings of these.
_FASTA_PROT1 = "MKTAYIAKQRSLYNTVATLPEPTIDEGILGFVFTLKQWERTY"
_FASTA_PROT2 = "ACDEFGHIKLMNPQRSTVWYSLLQHLIGLAAAAA"


@pytest.mark.integration
def test_build_peptide_mappings_offline(tmp_path, monkeypatch):
    """The proteome-dependent ``peptide_mappings`` build runs offline in CI.

    Normally the mapping pass fetches a reference proteome per species
    (pyensembl GTF or a UniProt FASTA download). ``fetch_species_proteome``
    uses an already-cached FASTA if one is on disk, so pre-placing a tiny
    synthetic FASTA at a UniProt-registry species' cache path lets the full
    pipeline — proteome resolution → ``from_fasta`` → the int-encoded
    ``ProteomeIndex`` (#250/#273) → ``map_peptides`` — run with no network.
    ``HITLIST_BUILD_WORKERS=1`` forces the sequential path (no
    ProcessPoolExecutor in the test).
    """
    from hitlist.mappings import build_peptide_mappings

    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path)
    monkeypatch.setenv("HITLIST_BUILD_WORKERS", "1")

    # Pre-place the FASTA where fetch_species_proteome("Sarcophilus harrisii")
    # looks for it → no download.
    proteomes_dir = tmp_path / "proteomes"
    proteomes_dir.mkdir(parents=True, exist_ok=True)
    (proteomes_dir / "sarcophilus_harrisii.fasta").write_text(
        f">sp|T1|prot1\n{_FASTA_PROT1}\n>sp|T2|prot2\n{_FASTA_PROT2}\n"
    )

    obs = pd.DataFrame(
        {
            "peptide": ["SLYNTVATL", "GILGFVFTL", "SLLQHLIGL", "NOTINPROT"],
            "source_organism": ["Sarcophilus harrisii"] * 4,
            "mhc_species": ["Sarcophilus harrisii"] * 4,
            "pmid": [1, 1, 2, 2],
        }
    )

    out = build_peptide_mappings(obs_override=obs, fetch_missing=False, verbose=False, force=True)

    m = pd.read_parquet(out)
    # Only the three peptides present in the FASTA map; NOTINPROT is dropped.
    assert set(m["peptide"]) == {"SLYNTVATL", "GILGFVFTL", "SLLQHLIGL"}
    assert {"protein_id", "position", "n_flank", "c_flank", "gene_name"}.issubset(m.columns)

    # Positions are correct against the source protein (validates the
    # int-encoded index end-to-end, not just presence).
    row = m[m["peptide"] == "SLYNTVATL"].iloc[0]
    pos = int(row["position"])
    assert _FASTA_PROT1[pos : pos + len("SLYNTVATL")] == "SLYNTVATL"
