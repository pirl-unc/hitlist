# Shared controlled MAPP panels and run-level recipe

Date: 2026-10-23. Codex / GPT-6. Owner: Hitlist759. Previous goal turn: progress
(schema collision fixed, real predecessor/source/recipe conservation verified).
Base:46df42d, current main0a21661 (2.0.4). This feature reserves2.0.6, building on
unreleased2.0.5 source work. Preserve the other checkout's unstaged documentation.

## Scientific scope and hypothesis

The immediate question is whether a directly perturbed processing panel has a
repeatable treatment response that sequence features can predict beyond a common
treatment shift. Admit PXD009508 (epoxomicin/capture controls) and PXD007263
(inflammation time course) from the previously recovered original bytes. Both
are captured-product observations. They cannot separately identify precursor
abundance, production, clearance, capture and MS response, or free concentration.
Do not redefine the full availability goal as fitting these observations.

## Shared source contract

1. Register one study-panel adapter, `wolflevy2018_controlled`, containing both
   projects with explicit `source_dataset` on every original/result/context row.
   Homogeneous tables and column meanings apply to both projects; source-native
   labels, identifiers, fields and all flagged rows remain available.
2. Pin both project/file metadata records, the ten original selected MaxQuant
   tables, their archive/member hashes, and the two existing later human model
   references. Reuse the reviewed reference reader/mapping relations. Original
   Human_UP_2016_02.fasta remains unavailable; later mappings are alternatives,
   not a claim about the original search or uniquely measured precursor.
3. Materialize source records; peptide/group rows; acquisition channels; evidence;
   peptide/run results; reported and reconstructed quantities; contributor edges;
   source-label aliases; context/protocol assertions; full reference alternatives.
   Preserve all6,060/3,146 peptide rows,19,581/27,196 evidence rows and18/12 runs.
4. Require exact reciprocal peptide/evidence/group links, evidence/summary raw-file
   agreement and unique full-label matches. Controls need the already verified
   hyphen/underscore normalization; retain both literal names and the derived
   alias basis. Time-course labels match exactly. Reject ambiguous aliases.
5. Replay all109,080/37,752 deposited peptide/run intensities against summed
   evidence with rtol1e-4/atol1. Preserve missing, zero, positive and partial sums
   separately. A source zero is never a measured biological absence. A missing
   evidence intensity must not silently become a measured zero. Keep direct
   MS/MS versus match-between-runs evidence and confidence levels distinct.
6. Keep capture (`alpha6`, `iso`, `gap`) separate from live-cell treatment
   (`ut`, `epoxo`), source replicate number and processing/preservation protocol.
   Literal alpha6 is not silently relabeled as a source-supplied PSMA1 identifier.
   Paper PSMA1 capture and epoxomicin1µM/4h context get explicit source scopes;
   gene KO and measured residual activity remain unknown. HEK293 cell identity
   is paper-level, human species is project-level; specimen/aliquot/pair IDs are
   unavailable. Numbered repeats do not create biological pairing.
7. For PXD007263 preserve literal TNF alongside paper TNF-alpha+IFN-gamma and
   0/1/2/6h context. Do not create an assured single-cytokine treatment default.
   Parameters are source evidence, including PSM FDR.01, protein FDR1, MBR and
   6–40 search scope. Use the full Methods, not just project descriptions.

## Canonical interface and recipe

Use catalog13 with the existing schema9 envelope, preserving catalog12 and both
historical schema8 readers. Extend optional-source presence through an explicit
catalog attribute rather than another growing source-name special case. Add
shared `processing_*` relations with normal aliases and explicit one/many links.
Existing reference, adjudication, MAPP and calibration outputs must be conserved.

Register `mapp_controlled_run_intensity_v1` through EvidenceStore materialize and
verify-recipe. It exposes all source run quantities plus a complete disposition
ledger, raw/later-reference identity and context, direct-MS/MS coverage, source
confidence and contributor IDs. Eligibility for a positive-response analysis:
nonreverse/noncontaminant; finite peptidePEP<=.01; one source group; valid
sequence; positive reported and positive reconstructed value. Do not discard
nonbinding peptides or impose an MHC-length filter. Preserve zeros and other
excluded measurements in the source/ledger. This is a run-level observation
recipe, not an implicit normalized composition or a concentration label.

Derive outcome-independent connected peptide/parent components from all exact
matches in the later references, including source-unlisted alternatives. Freeze
train/validation/test assignments from component hashes and report their scope:
sequence/parent separation, not homology or independent physical specimens.
The downstream signal diagnostic may construct a fully observed alpha6 contrast
from this recipe, with three positive values per arm and direct-MS/MS evidence
in each arm. Report all exclusions. No numerical pairing of replicates.

## Verification and execution sequence

- [ ] Implement source/metadata/aliases/reference integration and fixtures.
- [ ] Implement canonical queries and registered run recipe, full dispositions.
- [ ] Register and freeze the real admission experiment before generation.
- [ ] Independently replay original tables, aliases, quantities, all context,
      reference mappings, recipe memberships/targets and unchanged prior sources.
- [ ] Run required format/lint/test/docs, respecting the normal memory guard.
- [ ] On training components only, measure within-arm repeatability and the
      epoxomicin effect versus run variation; preserve totals and composition as
      distinct endpoints. Do not inspect held-out outcomes to choose features.
- [ ] Freeze the biological objective and matched exact/sampled comparison after
      source qualification, before held-out evaluation. Retain the mean/no-change
      references; compare sequence effects to a common treatment shift.
- [ ] Publish actual evidence, update759, close registry and preserve the full
      availability goal. No author outreach, cloud training or environment changes.

This is the next source and diagnostic step. The later minimal cleavage-kinetics
exact/Bernoulli-partition model remains separately specified in Presto; use a
purified known-substrate kinetic source for its initial rate-identifiability test
rather than treating cytokine exposure time as time since substrate synthesis.
