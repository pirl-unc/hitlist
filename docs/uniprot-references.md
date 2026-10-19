# Versioned UniProt references

Hitlist packages a small release catalog and downloads FASTAs on demand. Protein
sequences are outside the wheel and source distribution. The `human` collection
contains all human UniProtKB Swiss-Prot and TrEMBL canonical entries plus reviewed
isoforms; it is broader than the representative/reference-proteome subset.

```bash
hitlist data uniprot list --json
hitlist data uniprot fetch --release 2015_10
hitlist data uniprot fetch --release 2026_03
hitlist data uniprot path --release 2026_03
hitlist data uniprot info --release 2026_03 --verify
hitlist data uniprot remove --release 2026_03
```

`fetch` and `path` return a plain FASTA usable by existing mapping and digest
APIs. Omitting `--release` uses the catalog's **pinned default**, currently
`2026_03`. There is no mutable `latest` lookup. Record an explicit release in
reproducible workflows. Removal requires an explicit release. Unknown releases
fail rather than falling back to another version.

The 2026_03 asset is **127,856,511 bytes (121.93 MiB)** and contains 232,837
sequences: 20,431 reviewed canonical entries, 22,131 reviewed isoforms and
190,275 unreviewed canonical entries. The source transfer used 53,402,733
compressed bytes; installation retains the plain FASTA, its license notice and small receipts.

The historical **2015_10** asset is **82,758,858 bytes (78.93 MiB)** and contains
170,921 sequences: 20,196 reviewed canonical entries, 21,935 reviewed isoforms
and 128,790 unreviewed canonical entries. It was extracted from the complete
official release archives, with their published size and MD5 checks verified
before publication. The receipt records source and output SHA256s. Users
download only this human FASTA, not the 31.8 GB all-species source archive.
An independent exact-sequence audit matches all **661,142 distinct peptides**
across the 14 packaged Bekker-Jensen scopes (2,047,003 observation rows), keeping
I/L distinct. Reviewed canonical sequences alone explain 649,785; adding reviewed
isoforms explains 657,123. The complete human selection explains all 661,142.
The [audit receipt and script](https://github.com/pirl-unc/hitlist/releases/tag/uniprot-references-v1)
record the input hashes, comparison and bounded resource usage.
Sequence compatibility does not establish the original Bekker-Jensen search
database or its unobserved search space ([#654](https://github.com/pirl-unc/hitlist/issues/654)).

## Python

```python
from hitlist import (
    fetch_uniprot_reference,
    list_uniprot_references,
    uniprot_info,
    uniprot_path,
)
from hitlist.proteome import ProteomeIndex

path = fetch_uniprot_reference("2015_10")
reference = uniprot_info("2015_10", verify=True)
index = ProteomeIndex.from_fasta(path)

# Offline operations: no download, directory creation, or repair.
path = uniprot_path("2015_10")  # requires existing, checksum-verified bytes
releases = list_uniprot_references(verify=True)
```

`info` includes taxonomy, selection, release, sequence counts, source/build
provenance, attribution, URL, exact file size, SHA256 and local state. Default
inspection checks size only and reports `verified: false`; `--verify` hashes
the current bytes. Every `fetch` cache hit and `path` call verifies SHA256.
Corrupt files require explicit `fetch --force`; a failed replacement retains
the installed file. Copying an exact catalog FASTA elsewhere preserves its
content identity: detectability exports recognize its size and digest and
include the UniProt descriptor alongside the existing search-reference hash.
The caller must still supply the study-specific search-space contract.

## Storage bounds

Releases coexist at
`<data directory>/uniprot/<collection>/<release>/<filename>.fasta`.
`HITLIST_DATA_DIR` and `set_data_dir()` relocate this cache along with built
data. `hitlist data dirs` reports it; the general cache inventory recognizes
installed releases and their trusted hashes. Existing species/proteome caches
continue to use their existing paths and download behavior.

Downloads default to **256 MiB per asset** and **1 GiB total cache bytes**:

```bash
hitlist data uniprot fetch --release 2026_03 \
    --max-asset-bytes 268435456 --max-cache-bytes 1073741824
```

Python accepts `max_asset_bytes` and `max_cache_bytes`. Preflight counts all
regular files in the UniProt subtree, including hidden partials and manual
files, then reserves the entire incoming FASTA plus 2 MiB for transfer/control
headroom. Replacement keeps the old FASTA until verification succeeds and
therefore needs room for both. Concurrent writers share a lock. Historical
versions are never evicted automatically; exceeding the budget requires
explicit removal or a larger limit. Reusing existing verified bytes does not
need extra space and never evicts files, even if the configured budget is now
smaller than the existing cache.

These limits count logical file bytes, not filesystem blocks or quotas, and
apply only to the UniProt subtree. Mapping/digest indexes and other Hitlist
datasets have separate storage requirements. POSIX downloads retain bounded
resumable partials. New transfers currently require a POSIX local filesystem
(Linux/macOS); catalog inspection and existing verified FASTA reads are portable.
Managed paths cannot contain symlinks. Removal deletes
only the selected catalog FASTA, license notice, receipt and resumable bytes; unrelated files,
other releases and permanent coordination locks remain.

## Maintaining the catalog

`scripts/build_uniprot_reference.py` builds a compact human asset and a JSON
receipt. For the currently served REST release:

```bash
python scripts/build_uniprot_reference.py --source rest \
    --release 2026_03 --output-dir /tmp/uniprot-2026_03
```

The builder requires matching UniProt release headers, checks canonical counts
against the independent search endpoint, validates gzip completion, and bounds
input/output bytes and sequence-record memory. For `2015_10`, `--source archive`
streams official all-species archives, validates their published sizes/MD5s,
records SHA256s, and compares human counts with official release statistics.
It never stores or expands the full archive on disk. The complete 2015_10
archive extraction is validated; annotated and wrapped taxonomy records have
regression coverage, including an original historical UniSave entry.

Publish the verified FASTA and receipt as immutable named assets; add their
URL, size, SHA256 and provenance to `hitlist/data/uniprot_references.yaml`.
Retain existing release definitions and their bytes. Updating the pinned default
is an explicit catalog change, never a response to an unavailable release.
The FASTA is attributed to the UniProt Consortium under
[CC BY 4.0](https://www.uniprot.org/help/license), with Hitlist's selection and
format conversion recorded in the receipt.
