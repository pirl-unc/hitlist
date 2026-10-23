# PR 10 — bounded, provenance-bound release artifact verification (#755)

## Problem and invariant

`scripts/release_artifacts.py verify` validates clean main, local distribution
hashes, the successful release workflow and its unique named artifact, then
streams `gh api .../artifacts/<id>/zip` without a timeout. This child hung for
over 23 minutes during the 1.82.23 release. A release check must terminate
within a documented bound while retaining the comparison between the local
receipt and `release.json` downloaded from the exact successful main run.

The verifier must reject modified local distributions, a forged local receipt,
a different workflow run or artifact, a changed artifact identity, a failed or
timed-out download, and incomplete or corrupt downloaded contents. No failure
path may authorize publication or leave a trusted-looking partial download.

## Implementation

1. Keep the existing clean-main, version, local distribution, corpus-manifest,
   workflow and single nonexpired artifact checks. Give every GitHub metadata
   command a 30-second timeout.
2. Download by exact run ID and unique artifact name using
   `gh run download <run-id> --name <name> --repo pirl-unc/hitlist --dir <temp>`.
   Bound the subprocess to 120 seconds, with a clear failure on timeout or
   nonzero exit. Use a temporary directory for all downloaded files.
3. Require exactly the release receipt, wheel and source archive in that
   directory. Parse the trusted receipt; compare its JSON value exactly
   with the local JSON manifest and check the downloaded distributions against
   the receipt's SHA-256/size pairs. Re-query the run's artifact inventory and
   require that the same unique nonexpired artifact ID is still present.
4. Before implementation, add a focused failing test that requires an exact
   `gh run download` command and finite timeout. Extend the release fixture to
   write actual downloaded files, then cover metadata timeout, download
   timeout/failure, missing or corrupt receipt, altered downloaded bytes,
   forged local receipt and artifact replacement. Keep existing release tests.
5. Bump 2.0.1 to 2.0.2, run format/lint/full tests, check the wheel and
   source archive, review final diff, open PR, pass final-head CI, merge,
   run the clean-main release build and verify the exact published hashes.

## Review questions

- Does every network-dependent child process have a finite timeout?
- Can `gh run download` select anything outside the verified run and name?
- Does a name-preserving artifact replacement get rejected after download?
- Are local and trusted receipts still compared exactly, and are both local
  and downloaded distribution bytes tied to their receipt digests?
- Can any failed path leave a partial output or publish the release?
