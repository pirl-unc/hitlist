# Reconcile the concurrent TMAB source correction and native response recipe

PR794 merged as7649ffe while PR793's independent test suite was running. PR793's
merge was rejected with conflicts; its prior frozen source replay remains valid
only for ddc6581, not for the merged implementation. Upstream release run40071541566
is already live, so do not duplicate or cancel it. Preserve issue792's closure.

Prospective resolution before further source changes: merge current main into
PR793, retain upstream raw handling/mass/acquisition fields, and retain the strict
parser, position-bearing modifications, noncolliding context keys, full compound
names and explicit source eligibility verified in the independent replay. Publish
these together with the registered native-response recipe as2.0.17, sourcev3 /
schema17 / catalog22. Reject older semantic packages rather than silently relabel
them. The original v2 package versions are pinned historical evidence only.

The merged parser still strips pyroE's residue, drops oxidation positions, accepts
arbitrary annotation-letter extraction and conflates bestatin methyl ester with
bestatin. Its context key omits handling annotations and can overwrite them.
These are source corrections in the shared importer, not Presto-local repairs.
Keep the original raw columns and all values; do not reinterpret unknown identities.

Candidate reuse is a source-independent cell/sequence/charge/mass/first-five-ratio
fingerprint with at least two finite numeric values, including unidentified peaks
as possible candidates. Full source-independent keys expose different treatment
labels for adjudication; no specimen independence is asserted. Preserve the new
upstream candidate_reuse_key field and original raw source fields. Per-slot lineage
IDs distinguish treated, control and mean roles/replicates within each candidate
row group. Previously identified830known-sequence groups must remain a subset;
additional unidentified groups require separate reported counts.

The existing prospective native-response recipe spec remains in force. Its source
revision becomesv3. New full-source and reference replay must be registered and
independently verified before biological fitting. No broad claim based on tests
from the superseded branch, and no claim that source qualification is predictive
success. The overall APC availability/concentration goal remains active.

## Local verification review

The reconciled source and registered recipe pass format/lint (287 files unchanged)
and all 3,414 normal local tests (630.18 seconds, one worker). The frozen full-source
0259 Presto admission preserves all 10,564 source rows and 61,908 measurement
slots; 1,736,376 original-field invariants pass. It retains 830 identified and
553 unidentified possible reuse groups without asserting specimen identity.

The public recipe admits 2,737 response groups / 10,114 effective exact/bounded
observations, preserving all source contributors and dispositions. Independent
whole-string grammar, Aho–Corasick reference matching, bipartite graph traversal
and target/weight reconstruction pass for all 42,562 reference proteins, 1,155
occurrences, split components and observations. The first verifier's valid-header
terminator bug and its correction are preserved; source and recipe are unchanged.

The separate first biological baselines have completed but do not qualify APC
concentrations; their prediction/metric closure is separate. New-head hosted
checks, merge, clean-main release and PyPI verification remain outstanding.
Upstream PR794's existing release-build run40071541566 passed; it was not repeated.
