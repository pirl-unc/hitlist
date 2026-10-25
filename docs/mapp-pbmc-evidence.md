# Primary-PBMC MAPP observations

> The public MAPP API is now the [canonical captured-fragment MAPP package](mapp-canonical.md).
> `pbmc_mapp_*` and `external_pbmc_proteasome_fragment_quantity` were removed;
> this page retains PXD009438-specific source and cohort notes.

`PBMCMAPPAdapter` imports the deposited PXD009438 MaxQuant search tables as a
separate, source-faithful MAPP panel. It retains 985 peptide rows, 8,851 evidence
rows, 283 protein-group rows and 14 source acquisitions. It is captured
proteasome-associated fragment signal, not a concentration, MHC ligand,
cleavage-rate label or training example.

The deposit's `summary.txt` labels six acquisitions `CTRL_1` through `CTRL_6`
and eight `LUPUS_1` through `LUPUS_8`; each label is tied to one listed raw
file. The adapter represents that as project → source group → source sample
label → acquisition. A source sample label is deliberately a distinct context
kind: it can be queried without pretending it is a donor, a physical specimen,
or an aliquot. All fourteen specimen and aliquot identities therefore remain
null, and no cross-patient, cross-modal or other-project pairing is created.

The publication reports six healthy participants and eight participants with
SLE. Those are cohort facts, not additional run/specimen mappings. The literal
deposited labels remain `CTRL` and `LUPUS`; consumers that require a normalized
clinical meaning need a separately qualified source relation. The project
metadata says `blood serum`, while the publication describes PBMCs. Both claims
are retained as source assertions with their conflict visible rather than
silently choosing one as a run-level specimen label.

All original selected table rows and raw fields survive in source-record
relations. Repeated peptide sequences stay distinct when their deposited peptide
row IDs differ. Each peptide/run source intensity is replayed from its evidence
rows; source zeros, no reported evidence, partially reported evidence and
missing intensity retain different statuses. Matching-between-runs, direct
MS/MS, PEP, reverse and contaminant fields are also independent original claims.

To materialize the public panel, extract exactly the received archive members
and freeze the project/file-inventory JSON at the paths and hashes in
`mapp_pbmc_assets.json`. Then append it together with PXD009092 and the
controlled adapter, as shown in the [canonical MAPP guide](mapp-canonical.md#query).
Appending PXD009438 alone retains a verified source receipt but deliberately
does not expose a partial public MAPP relation set.

Filter the common `mapp_*` relations to `source_dataset=PXD009438` for this
panel. Its primary-PBMC rows remain distinct through that field, while the
identical relation names permit cross-source queries when appropriate. The
material graph makes its source samples queryable without claiming physical
identity; every original source field remains preserved through the canonical
base/extension relation and retained source table.

## Conditional observed-composition recipe

`pxd009438_observed_baseline_composition_v1` is an explicit, narrow consumer
contract, available through `EvidenceStore.materialize` and the normal recipe CLI.
It keeps CTRL and LUPUS separately weighted and preserves every peptide/run
admission decision. Source admission alone still does not authorize training.

Eligibility requires positive finite original intensity, canonical sequence,
explicitly blank peptide flags, peptide PEP in [0, 0.01), per-run `By MS/MS`, at
least one positive direct-MS/MS evidence row, all linked intensities reported,
and one distinct supplied parent accession. Every otherwise eligible occurrence
of a duplicated sequence within an accession/run is excluded, without summing or
choosing by intensity. Groups need at least three distinct remaining sequences.

Targets are within-group centered log10 intensity and observed-set intensity
fraction. Source-wide sequence/accession connected components include excluded
bridges before eligibility. Each clinical source group gets equal component,
then accession/run group, then peptide weights. This external-evaluation recipe
has no new train/validation split. Its support flags require at least 20 components,
20 groups and 100 observations per cohort; materialization retains the data and
failed support flags rather than silently relaxing the policy.

The output includes examples, label-free covariates, decisions, groups and all
component nodes. Its manifest binds all twelve source relations and the source
generation; verification independently recompiles the derivation from those
snapshots. Original-source parsing is a separate audit.

On the pinned deposit, CTRL has 6 components / 27 groups / 234 observations and
LUPUS has 7 / 20 / 161. Both fail the component support requirement. These counts
do not constitute a model-transfer result. Parent origin/species remains a project
assertion because the original full search FASTA is unavailable. The deposited
6–40-residue nonspecific-search scope, capture recovery, peptide-specific MS
response and unresolved physical/cross-modal identities still rule out interpreting
these quantities as free-peptide concentrations or unobserved peptides as negatives.
