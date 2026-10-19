import json

import pyarrow as pa
import pyarrow.parquet as pq
from scripts.allele_candidate_audit import audit


def test_audit_distinguishes_rows_paper_peptides_global_peptides_and_blanks(tmp_path):
    rows = []
    for pmid, peptide, restriction, before in [
        ("999", "SHARED", "HLA-DRB1", ""),
        ("999", "SHARED", "HLA-DRB1", ""),
        ("998", "SHARED", "HLA-DRB1", ""),
        ("999", "EXISTING", "HLA-DRB1", ""),
        ("999", "EXISTING", "HLA-DRB1*12:01", "HLA-DRB1*12:01"),
        ("999", "BLANK", "", ""),
        ("999", "MISMATCH", "HLA-DQ", ""),
    ]:
        rows.append(
            {
                "pmid": pmid,
                "peptide": peptide,
                "mhc_restriction": restriction,
                "mhc_class": "II",
                "host_mhc_types": "HLA-DRB1*12:01",
                "mhc_allele_set": before,
            }
        )
    path = tmp_path / "observations.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    output = tmp_path / "audit"
    result = audit(path, output, batch_size=2)
    assert result["gene_locus"]["n_rows_gained"] == 4
    assert result["gene_locus"]["n_peptides_gained"] == 2
    assert result["corpus_candidate_peptides"]["n_peptides_gained"] == 1
    assert result["blank"]["n_rows"] == 1
    assert result["blank"]["n_rows_with_typing"] == 1
    assert result["blank"]["n_rows_gained"] == 0
    papers = json.loads((output / "per_paper.json").read_text())
    paper = next(p for p in papers if p["pmid"] == "999" and p["scope"] == "gene_locus")
    assert paper["n_peptides_gained"] == 2
    assert paper["corpus_candidate_peptides"]["n_peptides_gained"] == 1
    assert json.loads((output / "manifest.json").read_text())["input"]["n_rows"] == 7
