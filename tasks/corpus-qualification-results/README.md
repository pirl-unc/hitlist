# Independent corpus build receipts

Current run: `2d794b88453e75e4ae548326c4ffaaea05748d7df81140fcd24dbf0c8053ff71`.
Producer: released Hitlist 1.76.0, datacache 1.27.1, Python 3.12, Linux.
These are downloaded, independently hash-verified stage receipts. They are not
training-readiness or held-out performance reports.

| Receipt | Meaning | Runtime (seconds) | SHA-256 |
| --- | --- | ---: | --- |
| [environment.json](environment.json) | Historical first run; reference stage failed | 2.21 | `e59d519eb2f05e461976df5c7096ea1c6188437f79bc095fb375a54bb2e23784` |
| [environment-posix.json](environment-posix.json) | Current environment; pip check passed | 2.55 | `2e672bd00a59e00f2c137627132bd011b4039c866f39299ac3a6cd3159a5bcc9` |
| [reference.json](reference.json) | Current human Ensembl-112 reference; complete | 218.19 | `05a5331266bfda09e5ea3c178feccbd9948f227bdd64adc407087f42d9ad78b2` |
| [sources.json](sources.json) | Current source registration; complete | 72.02 | `fdd32b334856db03202fa717506c72c2b6cbbcbda8037fe8e81d7d8efeb1477a` |
| [indexes.json](indexes.json) | Current independent indexes; complete | 2026.60 | `c26fa421b55bff5bb90114260f09a81a07a5c45e1b812decbd56f2e7c8185823` |
| [mappings.json](mappings.json) | Human Ensembl-112 mappings; complete | 582.58 | `bb3177fea10370a13ff101f8babec125252896e450c84e5300968994ef7f02c0` |
| [snapshot.json](snapshot.json) | Complete portable generation; verified | 26.28 | `e438474a95dc3be347e3142f8faef3d610cce9a9e1c2e1be1ff4f1102f1fe0fd` |

The source receipt pins both original CSVs (14,719,407,889 bytes total) and 26
paper assets (including citation descriptions and public URLs). The source export
dates are unverified; local file times do not establish them. Optional DepMap
matrices are not registered. Source files and products remain in shared datacache.
The reference receipt pins four Ensembl downloads and the complete eight-file
indexed bundle. Its SQLite database was built on local POSIX scratch, closed and
published as an immutable datacache bundle; mapping workers consume verified local
copies. Modal Volume hard-link support is tracked in openvax/datacache#125.

The historical `environment.json` belongs to run
`fb42a56f489d79964fb689abcb7706547a16cf5d3c26524024646a2fcbb0d82b`.
Its subsequent reference build failed on the volume's unsupported hard link. That
failure is preserved remotely in the original run's reference receipt; its
successful environment receipt does not qualify the failed reference output.

The independent indexes completed successfully: 4,486,991 MS observations,
814,268 binding rows and 26,537 other assay rows. The generation also contains
bulk proteomics, line expression and the full observation-contributor table.
These are build counts, not qualified independent training examples. The index
stage took 2,026.60 seconds; observed peak runner RSS was 32,400,773,120 bytes
(worker processes are excluded from that metric). The index receipt was downloaded
and independently hash-verified. Human-reference mappings completed over
1,506,664 unique peptides from 5,327,266 assay rows. All ten public-facing files
were downloaded from the cloud, checked against the snapshot receipt, published
as [ci-corpus-v3](https://github.com/pirl-unc/hitlist/releases/tag/ci-corpus-v3),
then independently downloaded again through the pinned datacache installer.
[public-corpus.json](public-corpus.json) lists every public URL, byte count and
SHA-256. The public manifest is
`7f7f4035d8ec080b26385710d447633dba36ac5cce49008e48e14cf8894f20d5`.

The test and release workflows now install that complete manifest and validate
hashes on both cold downloads and cache hits. CI compatibility remains a release
gate. Actual cohort partitions and observation-weighted coverage/leakage reports
remain pending; the public build snapshot is not a qualified training split.
