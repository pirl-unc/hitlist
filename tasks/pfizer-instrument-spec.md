# Pfizer canine instrument provenance — #670

The publication reports Fusion Eclipse acquisition and mentions Lumos in its
real-time-search description; the ProteomeXchange deposit lists Fusion Lumos.
The flat protein table cannot settle those reports. Exposing Eclipse without
qualification can mislead acquisition/detectability consumers.

Keep the original asset identities, source IDs and bundle schema unchanged.
Add structured, cited instrument reports and an explicit conflict status to the
Pfizer catalog. Set its normalized instrument to a readable unresolved value
containing both reported models. Existing import code already snapshots the
complete source metadata and puts the source instrument in normalized views;
no new inference or migration of old frozen bundles is needed. Definitive raw/run
metadata reconciliation stays in #668.

Add a failing offline regression before changing the catalog. Prove both reports,
their URLs and the conflict status survive into a curated manifest, and that
normalized protein rows cannot present Eclipse alone as resolved. Document the
uncertainty and frozen-snapshot behavior. Bump to 1.71.1, run format/lint/tests
and final-head CI, then merge after 1.71.0 is published. Deploy from clean main
and verify public PyPI bytes against the tested artifacts.
