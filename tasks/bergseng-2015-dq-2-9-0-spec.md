# Spec: `hitlist==2.9.0` — Bergseng 2015 (`PMID 25502872`, `PMC4297300`) DQ Run-Level Ion Volumes, Sample Hierarchy & Typing Ambiguity (#859)

## Goals

1. **Pinned CC-BY 4.0 Source Contract (`hitlist/data/evidence_revisions/bergseng_2015_dq.json`):**
   - Pins official EuropePMC OA archive (`PMC4297300/supplementaryFiles`) member `251_2014_819_MOESM3_ESM.xlsx` (`4,693,608` bytes, MD5 `f655ec4556d1a3b74bbb878026e0be06`, SHA-256 `30d6206813788ffd05eabded3f276d950605b923e4798aaabba0be6d5a50dd37`).
   - Pins Table 1 DNA typing for all 8 IHWG-number-matched cell lines (`9037 SWEIG`, `9089 BOB`, `9064 AMALA` preserving `DQA1*05:01/05:03` + `DQB1*03:01` ambiguity, `9088 PF04015`, `9087 STEINLIN`, `9051 PITOUT`, `9047 PLH`, `9050 MOU`) while explicitly leaving header group `RN` unresolved (`unresolved_alias` — never mapped to `CD114` by elimination).
   - Preserves MaxQuant `1.3.0.5` search parameters (3-min RT alignment / match-between-runs, variable modifications without modified-sequence column, 134 PEP > 0.1 rows retained without silent filtering, unspecified low-total-signal experiment exclusion caveat, allelic cell-line comparison scope).

2. **Adapter, Package Builder & Uniform Catalog Registration (`hitlist/bergseng_2015_dq.py`, `hitlist/_dataset_catalog.py`, `hitlist/__init__.py`):**
   - `fetch_bergseng_2015_source()`, `bergseng_2015_source_contract()`, `parse_bergseng_2015_workbook()`, `write_bergseng_2015_dq_package()`, `verify_bergseng_2015_dq_package()`.
   - Separates `peptides.parquet` (12,712), `runs.parquet` (54), and `measurements.parquet` (686,448) with exact source-cell coordinates (`Sheet1!L3`), raw MS1 ion peak volumes vs per-run sum-normalized relative fractions, explicit zeros (`521,368`) vs positives (`165,080`) vs missing (`0`), and explicit separation from aggregate IEDB detections.
   - Registers `"bergseng-2015-dq"` in `hitlist/_dataset_catalog.py` so `hitlist data fetch bergseng-2015-dq` and `hitlist.fetch("bergseng-2015-dq")` work uniformly.
