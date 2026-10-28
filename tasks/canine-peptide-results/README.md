# Released canine peptide consumer acceptance

`acceptance.json` records Hitlist's offline consumption of the public ORFdata
`pfizer-atlas-dog-peptide-evidence` release `2026-10-27.1`.

The complete 24,710,258,688-byte artifact was verified against the registry pin
and independently snapshotted without changing its bytes. This preserves all
70,280,798 retained native rows and their complete reference/provenance. Common
counts, all 44 sample contexts and all 15,726 group relations match the original artifact. The explicitly
named peptide query checks every field of 374 measurements and 13 mapping sets.
Sockets and DNS were disabled during curation and queries.

This is a consumer/artifact acceptance check. The full vendor extraction and
source reconciliation belong to the upstream
[data release](https://github.com/openvax/orfdata/releases/tag/data-pfizer-dog-peptide-evidence-2026-10-27.1);
Hitlist did not perform another vendor extraction or assert a new full vendor audit.
The two Pfizer generations describe one study. Neither provides positive DLA
observations or independently positive dog counts.

Reproduce with Python 3.11+ and the pinned dataset already installed in an ORFdata
cache (no acquisition is performed):

```sh
python scripts/verify_canine_peptide_evidence.py \
  --cache-root /path/to/openvax-cache \
  --output /path/to/new-canine-bundle \
  --receipt /path/to/acceptance.json
```

The output directory must not exist. This creates the independent 24.71 GB
snapshot and checks its full hash; allow corresponding space on filesystems
without copy-on-write clones. The selected query compares every native
measurement field and every mapping alternative; the script does not iterate
all 200 million logical quantitative cells.
