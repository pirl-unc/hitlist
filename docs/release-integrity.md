# Release publication and interrupted gates

A release is shipped only after both public PyPI files match the tested wheel
and source archive. Merging, building, uploading one file, and trusting PyPI's
metadata alone are insufficient.

## Publish a hosted build

Dispatch the **Release build** workflow on clean main. It runs formatting, lint,
both complete test phases, the distribution/license checks, and records the
source commit, dependency revisions, corpus hashes and distribution hashes in
`release.json`. Download its `hitlist-release-<commit>` artifact, then run from a
clean main checkout at that exact commit:

```sh
./deploy.sh --artifacts /path/to/downloaded-artifact
```

The publication command verifies the successful manual main workflow, fetches
its artifact independently, and compares the manifest and both distributions.
It snapshots the verified bytes for upload, checks any existing PyPI files,
uploads only missing matching-version files, then downloads and hashes both
public files. It writes `pypi-<version>-verified.json` beside the artifact directory.

A matching partial upload can be resumed with the same command. A partial pair
is never accepted as a completed release. A conflicting file, unexpected file,
yanked file, or failed metadata/download request stops publication. In particular,
403/429/500 responses are not treated as an unused version. Never use
`--skip-existing` to mask a collision. A conflicting published version requires a
new version and its own complete gate; public files are immutable.

The existing `./deploy.sh` local path still runs lint and both complete test
phases before building. Its publication step requires clean main and uses the
same collision, exact-file upload and public-byte verification checks.
`./deploy.sh --build-only` never uploads. The hosted artifact path is preferable
when local memory admission refuses a full corpus test run.

## Failure evidence

The hosted gate saves `gate.log` and `gate.json`, including pre/post memory and
cgroup snapshots, process CPU/peak RSS, child return status, and signals received
by the wrapper. It forwards shutdown signals to the gate's process group and
preserves failure status. Diagnostic artifacts are retained for 30 days, including
failed gates when the runner survives to execute the upload. Release artifacts
are uploaded only after success. A runner that is forcibly shut down may never
execute an `always()` step; live Actions logs remain part of the evidence.

Exit status 143 alone does not establish an OOM or identify a signal sender.
Compare runner messages and memory/cgroup evidence; report missing evidence as
unknown. Peak RSS describes a process high-water mark, not summed simultaneous
memory across workers.

## Historical incidents

### #844: 2.0.38 collision and the missing 2.0.39 replacement

The original 2.0.38 PyPI pair did not match the validated build of `82aa4aa`.
Those files were left unchanged. [PR #843](https://github.com/pirl-unc/hitlist/pull/843)
produced the version-only replacement at
`4f4e398ca1618ddf2418d61b10eb254f61d2ae25`.
Its [manual clean-main release gate](https://github.com/pirl-unc/hitlist/actions/runs/40177039295)
succeeded, but the replacement was absent from PyPI when this audit checked both
the version endpoint and the complete release inventory.

The retained artifact was independently downloaded and verified against that
clean historical main checkout, its successful workflow, manifest and exact file
inventory. The pair was then published as
[2.0.39](https://pypi.org/project/hitlist/2.0.39/) and both actual public downloads
were verified. The checked receipt is
[`tasks/reliability-results/pypi-2.0.39-verified.json`](https://github.com/pirl-unc/hitlist/blob/main/tasks/reliability-results/pypi-2.0.39-verified.json).
This establishes the replacement's provenance; it does not retroactively assign
that provenance to 2.0.38.

### #754: release runner shutdown

The [first attempt](https://github.com/pirl-unc/hitlist/actions/runs/39836089683/job/120024709056)
at `1a70e5f910099e9d6331c02b3828f00403b2f9a9` passed the unit phase and reached
32 of 96 integration tests. At `2026-10-22T19:20:41.1622984Z`, the retained log
explicitly reported: “The runner has received a shutdown signal.” The shell
reported 143; post-checkout and post-Python cleanup and release upload were skipped.
The retained annotations add the exit status but no initiating actor or OOM event.

The [second attempt at the identical commit](https://github.com/pirl-unc/hitlist/actions/runs/39836089683/job/120036050566)
succeeded, including all 96 integration tests in 1025.93 seconds. Its largest
process peak was 15,349,668 KiB. This confirms a runner shutdown in attempt one
and a complete successful retry; the original sender and underlying trigger
remain unknown. Available memory near the prior measured peak was a reason to
add headroom, not proof of an OOM.

The existing #768 mitigation supplies bounded swap and per-test progress. The
additional retained diagnostics make future surviving-runner failures easier to
investigate without weakening memory admission, retrying failed tests, or allowing
partial test runs to approve an artifact. This closes the bounded historical
investigation without inventing a more specific cause.
