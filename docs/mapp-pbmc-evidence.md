# Primary-PBMC MAPP observations

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

To append the panel, extract exactly the received archive members plus freeze
the project and file-inventory JSON at the paths and hashes in
`mapp_pbmc_assets.json`, then supply that root explicitly:

```sh
hitlist evidence append-sources PREDECESSOR OUTPUT \
  --source-root pxd009438=PXD009438_ASSETS
```

The catalog exposes `pbmc_mapp_*` relations and
`external_pbmc_proteasome_fragment_quantity`. This dedicated surface avoids
mixing primary-PBMC rows with HEK293 or inhibitor/capture-control MAPP rows;
the common typed relation shapes and material graph make all three queryable by
the same EvidenceStore interface. Every physical column remains available as
`native.<field>`.

No endpoint-specific recipe is registered for this panel. The deposited 6–40
residue nonspecific-search scope, MAPP capture recovery, peptide-specific MS
response, unresolved physical identity, and unresolved cross-modal links rule
out treating unobserved peptides as negatives or admitting these rows as
training data by default.
