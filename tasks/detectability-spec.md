# Detectability datasets — #361 and source controls #654

## User scope

Build theoretical-digest/observed-peptide training datasets for Presto, with
search-space, parent-protein observation and acquisition-depth controls. Keep
Hitlist responsible for evidence, not vaccine assembly. The user confirmed:
report first-seen depth only within demonstrably comparable protocols; retain
protocol-scoped observed/not-observed labels.

## Verified source findings

A bounded HTTP-range audit retrieved PXD004452 SearchResults.zip/parameters.txt
and summary.txt; no full archive download. Original files and audit script are
in /private/tmp/hitlist-bekker-search-audit and
/private/tmp/hitlist-search-metadata-audit.py. The archive is ~21.7 GB; summary
is 587345 bytes (117207 compressed). The deposit directory contains raw files,
SDRFs, README and SearchResults.zip, but no search FASTA or mqpar.xml.

- Correct paper: PMID 28601559, not the currently recorded 28591648 (an unrelated
  zebrafish review). https://pubmed.ncbi.nlm.nih.gov/28601559/
- Original parameters: MaxQuant 1.5.3.19; minimum peptide length 7; fixed
  Carbamidomethyl(C); MBR enabled; local HUMAN.fasta path only. Exact reference
  identity and unrecorded search limits must not be invented from a current
  proteome. Paper methods differ (1.5.3.6 and excluding MBR downstream).
- Summary: Trypsin/P 3 missed cleavages; Chymotrypsin+ 4; GluC;D.P 3; LysC/P 2;
  Specific enzyme mode. Four variable modifications: Oxidation(M), protein
  N-terminal acetylation, Gln->pyro-Glu, phosphorylation(STY); multi-modification
  Deamidation(NQ). Preserve modifications/search settings, not just enzyme names.
- Cox Lab distinguishes Trypsin/P (permits cleavage before P) from classical
  Trypsin (excludes before P). Existing Hitlist aliases conflate them incorrectly.
  https://cox-labs.github.io/coxdocs/andromeda_enzymes.html
- Current ingest collapses Tryp-46fracs and HeLa-46fracs-IT-E1/E2, aliases their
  replicate IDs to E1/E2, and loses the original experiment labels. Its CSVs
  retain only leading-razor protein/position, not all parent alternatives.
- The 14-fraction data were reused from an earlier study. Fraction count alone
  therefore cannot establish protocol comparability or a dose-response.
- Peptide CSV detections mean nonzero intensity, potentially including MBR;
  they must not be relabeled as direct MS/MS identifications. Protein abundance
  is a same-arm sum of intensities assigned to leading-razor proteins, not an
  independent measurement proving every alternative parent was expressed.

## Design constraints

1. Expose `build_detectability_training_set` in bulk_proteomics, with the requested
   cell line/enzyme/depth, missed-cleavage and length controls, default parent
   observation requirement and documented acquisition fields. Add a streaming
   iterator so a proteome-scale result need not be retained as a whole frame.
2. Require the actual searched FASTA (or an explicit search-reference contract),
   hash it and capture exact search constraints/provenance. Never silently use a
   current reference as though it were the historical search database. Missing
   essential search metadata fails before assigning absence labels. Allow
   explicit caller-supplied verified metadata; default known source fields only
   from captured evidence. Document any unavailable historical inputs.
3. Candidates are individual peptide occurrences with parent ID, gene, 1-based
   inclusive start/end, flanks and missed cleavages. Preserve repeated occurrences
   and alternative proteins. Refactor digest generation to share cleavage logic
   with the existing set-returning helper; distinguish Trypsin and Trypsin/P.
   Validate other enzyme definitions from primary evidence, without guessing
   GluC;D.P semantics from the name. Enforce the declared search length, mass and
   missed-cleavage envelope before labeling; query limits may narrow it.
4. Join observation by sequence within the selected documented source/search
   scope. A shared peptide detected under another parent must not become a
   negative just because a razor assignment chose a different parent. Parent
   observation and abundance come from the same scope and carry their assignment
   basis. Missing reference parents/sequence mismatches require explicit handling.
5. Keep source, protocol/search IDs, enzyme, enrichment, fractionation pH, depth,
   detection basis and abundance basis in output. Do not manufacture experimental
   resolution lost by the old ingest. Mixed protocol pools must be explicit;
   unresolved replicate counts are nullable with a reason, never false precision.
6. First-seen depth is conditional on a validated comparison group with all
   non-depth protocol controls equal. No group/insufficient evidence => nullable.
   Distinguish absence in the selected search from intrinsic undetectability.
7. Bound candidate generation and temporary/output storage; fail explicitly at
   caller-configurable limits, never silently return a truncated training set.
   Stream source files/proteins and avoid loading the entire bulk parquet before
   filtering. Preserve all input/provenance hashes and parameters in result attrs
   or a companion manifest for any export.
8. Correct confirmed source metadata defects with #654 and add cache/read
   migration where required. Keep raw source provenance inspectable. Do not
   regenerate unrelated MS indexes or mutate the shared environment.

## Verification

- Tiny deterministic search references: positives, same-protein negatives,
  unobserved parents, observed shared sequences, repeated positions, search-
  excluded candidates, reference mismatch and empty results.
- Correct Trypsin vs Trypsin/P K/R-P boundary behavior, missed-cleavage counts,
  terminals and coordinate/flank reconstruction.
- Reject mixed cell/enzyme/depth/enrichment/pH/search scopes; no cross-protocol
  first-seen labels; independent replicated evidence must not be double-counted.
- Streaming/DataFrame equivalence, deterministic ordering, limit/failure cleanup,
  immutable inputs and pandas 2/3 compatibility.
- Bounded real source audit with clear scope, full required format/lint/test/CI,
  separate versioned PR, merge and verified PyPI publication.

## Current release dependency

#644 is PR #653, head e526e96861b6dc1f77cd9120f81f5cc191e5fa56, version 1.66.0.
This branch starts there while its CI runs; rebase only this branch's commits
onto the merged main before opening the independent #361 PR. The implementation is in progress. Detailed implementation choices should be tightened as the
remaining primary search-reference/control limitations are resolved.


## Additional primary enzyme evidence

Cox Lab's own XML at commit 25bdb094d6b23c1f4c3e07aa4c766e43fa4bee5b,
PluginTutorial/conf/enzymes.xml, is captured locally as cox-enzymes.xml in the
source audit directory. Its explicit specificity pairs prove:

- Trypsin excludes K/R-P; Trypsin/P includes them.
- Chymotrypsin is F/W/Y; Chymotrypsin+ is F/W/Y/L/M; both include P after the cut.
- GluC cleaves E including E-P. D.P is ONLY the DP pair, so GluC;D.P means E or
  D-before-P, not E/D except before P. Existing Hitlist's GluC helper was wrong.
- LysC excludes K-P; LysC/P includes it.

The digest refactor now starts from these exact search names. The contradictory
legacy long Trypsin/P label must request an explicit unambiguous choice rather
than silently changing its meaning. Biological labels in historical bulk CSVs
need a source-specific mapping to the recorded search definitions; they must
not determine a search rule by name alone. The source adapter now records 14 curated scopes and the validated dataset builder is implemented.


## Reference search result and implementation review

The original archive inventory, README, current SDRF, PRIDE API and original
ProteomeXchange XML have no FASTA or reference release identifier. Linked
PXD013455 explicitly uses June 2017 UniProt (71,591 sequences), which cannot
identify the original pre-publication search reference. RPXD056882 and Winnow
also provide reanalyses, not evidence identifying the original HUMAN.fasta.
Do not substitute any of these references under original labels. #654 stays
open for recovery/re-ingestion; #361's generic API can consume verified custom
searches, and its historical adapter requires the missing contract explicitly.

Implemented an explicit search contract, occurrence-preserving shared digest,
reference/parent/sequence validation, sequence-level observed labels with
parent occurrences retained, comparable-only depth labels, chunked historical
source adapter, stable schemas, iterator, materializer, and atomic Parquet +
manifest export with a hard storage cap. Historical pooled replicate counts
are nullable; source evidence and limitations are retained in curated YAML.


Validation checkpoint: 150 focused tests pass on pandas 3, with the earlier
148-test set also passing on pandas 2. Format and lint pass. The full test script
was invoked but its unchanged memory guard refused 0.71/0.68 GiB available
against 2.5 GiB required; full CI and release checks must pass before merge.
A chunked real-source audit covers all 2,047,003 peptide rows across 14 curated
scopes (25,000-row parser batches). It is a source-control audit, not a historical
training export with an invented FASTA. A real searched reference was found in
PXD013455, Human-ReferenceProteome-Canonical-Isoform-71591.fasta; its observations
and mqpar-celllines.xml are distinct from the original search.
