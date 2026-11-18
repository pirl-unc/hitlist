# Final qualification of APC curation PR #909

Preserve the eight-study curation scope and original source statements. Reconcile
with export/filter releases and use version 2.1.15 after those ship.

## Primary-source qualification

Verify intervention background, concurrent factors, exact genotype and material
against each paper. Critically distinguish source categories combining enrichment
and uniqueness from genuine nondetection. PMID 31222486's 417 inhibitor-specific
and 1,198 control-specific peptides combine unique and >=4-fold-enriched detections;
the IEDB category text cannot justify blanket negative comparator memberships.
Keep positive membership and raw statements, remove unsupported exclusions, and
document the original quantitative matrix as the evidence needed to refine them.

Reconcile DP constructs against the original supplementary table (#921), which
lists 13 pairs and omits deposited DPB1*19:01. Preserve the latter restriction
without asserting its unverified construct. Add both cultured and subcutaneous
tumor EL4 arms (#922), blank material consensus on unresolved EL4 observations,
and correct all three Hilton source references to PMC5510751 (#923).

## Expansion API qualification

Review the new long-form condition projection for study-scoped identity,
ambiguous arm metadata, missing metadata/IDs, contradictory memberships, and
preservation of source provenance. Require explicit errors where an invalid join
would otherwise attach an arbitrary biological arm. Keep ordinary empty inputs
cheap and usable; do not invent controls, quantities, or negative binding labels.

## Verification

Add regression cases for source corrections and any confirmed expansion defects.
Use the existing source fixtures and corpus tests; retain row-count conservation.
Run format/lint and guarded test.sh, isolated metadata/dependency audits and
hosted final-head checks. Record exact outcomes before merge/deploy.
