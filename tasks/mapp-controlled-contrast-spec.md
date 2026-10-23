# Shared complete-case inhibitor contrast

Implement the already frozen Presto controlled-mapp-exact-sampled-fit contract
through the canonical Hitlist materialize/verify API before fitting. No changed
source selection, split, quantity or biological interpretation. The base run
recipe and its outputs must remain unchanged.

New registered ID: mapp_controlled_complete_alpha6_response_v1. Reuse the ten
verified source tables and base run admissions. For every source peptide retain
a contrast disposition. Only PXD009508 alpha6 runs qualify. Require all three
positive admitted UT and epoxo runs and at least one direct MS/MS identification
in each arm, without specimen pairing. Keep all six individual measurements,
IDs, inherited component/split, source sequence, and arithmetic mean intensity
per arm. These are captured-signal means, not concentrations or imputed values.
Keep the base run examples and every base run exclusion in the portable output;
original source tables retain contexts and contributors. No normalization or
pooled ratio in curation: the model constructs its conditional distribution on
its declared support. The treatment-arm positivity selection must remain explicit.

Verification: regress base recipe unchanged; fixture missing-zero exclusion,
positive six-run admission and correct arithmetic means/contributors; matching-
only arm excluded; corrupted contrast target rejected even after outer hash
repair; generic canonical materialize/verify dispatch and source links tested.
On real data independently reproduce all 59/14/14 members, six source quantities,
means, source IDs, contexts, splits and zero held-out selection changes. Stage
before training; do not tune any eligibility based on response values.
