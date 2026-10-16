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

The independent observation indexes are building. Mappings, a complete public
portable generation, actual cohort partitions and coverage/leakage reports are
still pending. No incomplete index generation is published as a corpus.
