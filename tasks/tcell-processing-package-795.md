# Standalone T-cell and processing evidence package (#795)

## Specification

The existing schema-5 T-cell evidence generation and the controlled MAPP
processing relations are source-faithful but are exposed through different
appendage/registry paths. This successor introduces one package envelope and
query contract without merging scientific target families. Functional T-cell
responses, pMHC multimer binding, receptor evidence, direct processing
measurements and controlled MAPP products remain distinct; missing source
families remain explicit availability results rather than synthetic negatives.

## Plan

- [x] Audit the current T-cell v2 and controlled-processing manifests, source
      assets, relation schemas and registry paths; freeze row/count/hash anchors.
- [x] Define schema-18/catalog-23 package-family metadata and the common source,
      assay-context, reported-measurement, molecular-input, lineage and
      eligibility contract.
- [x] Add typed family projections for functional response, multimer binding,
      receptor/repertoire evidence, processing measurements and MAPP products;
      preserve source values, units, qualifiers, context, contributor rows and
      unknown/not-applicable/absent states.
- [x] Register one standalone EvidenceStore/query package while retaining
      predecessor manifests and preventing family-specific joins from
      multiplying observations.
- [x] Add availability and no-direct-processing audit relations, source-linked
      examples, replay/count/hash tests, and explicit donor/restriction policy.
- [x] Run hosted gates, merge, and deploy 2.0.18.

## Review

The package replay built from the pinned T-cell v2 and PXD009092 generations
is 208 MB and preserves 55,844 T-cell examples, 578,529 assay rows, 192,114
MAPP quantity results and 362,882 MAPP quantity measurements. The family
ledger records receptor and direct-processing availability as explicit
unmaterialized/unobserved states. Local focused tests and standalone
EvidenceStore alias queries pass. `./format.sh` and `./lint.sh` pass, and the
host-level one-worker `./test.sh` run passes all 3,418 non-integration tests.
The package manifest verifies as schema 18/catalog 23; its EvidenceStore
replay returns 47,421 functional responses, 8,423 multimer bindings, 192,114
processing measurements, 362,882 MAPP measurements, five family rows and five
availability rows.

Release update: PR #796 merged as `fda1eff79d7a77f195e6a5f3a4c5abbee5cfd7c4`;
clean-main release workflow `40079964064` passed. Hitlist 2.0.18 is on PyPI
with wheel SHA-256
`eece32b864313a5bddd45113a48717d7e86da37ea4d55fb5c3f5be487c7ea065` and sdist SHA-256
`8177479c633b5e870a89a4bc682d53b2c8e208e8f7cf5fb1c0cd3366ab36fafb`.
