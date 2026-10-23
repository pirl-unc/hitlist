# Admission amendment before any model fitting

The original three-peptide recipe materialized 544 peptides / 65 full parent
sequences, with train/validation/test parent counts 51/7/7. Independent replay
from all original TSV/XLSX inputs, reference hits and all-source graph confirms
these values. The prespecified ten-parent-per-split gate failed. No model was
fitted, no held-out prediction or model score was inspected, and the original
recipe/export remains immutable.

Register one extension: `mapp_captured_composition_min2_v1` admits parent sets
with at least two eligible measured peptides. Two categories are the mathematical
minimum for a nontrivial conditional composition and condition-redistribution
comparison. The original three-category restriction is not an identification or
confidence criterion. This extends the usable biological observations; it does
not relax PEP, direct MS/MS, replicated positive quantities, sequence ambiguity,
source-leading coordinates, canonical residues or source provenance.

Critically, preserve the original all-source component graph and original split
salt `mapp_captured_composition_v1`. A new recipe name must not reseed the same
observations. Every existing parent stays in its original partition, with identical
original candidate quantities/targets. Only newly eligible two-peptide parents
can be added. Verify this nested relationship independently. The ten-parent gate,
all model settings, three seeds, baselines, objective, metrics and test-use policy
remain unchanged. If this extension fails the support gate, leave fitting stopped
and acquire another biological source; no further membership or split tuning in
this family. Record two-peptide groups separately in evaluation.
