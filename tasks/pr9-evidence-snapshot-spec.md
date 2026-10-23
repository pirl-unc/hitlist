# PR 9 — independent evidence-generation snapshots (#748)

## Problem and invariant

`mhc_evidence._link_or_copy` hard-links when source and destination share a
filesystem. MHC native assets, cohort/curation inputs and predecessor files
then remain mutable through the original path. A hash check detects later
damage but cannot restore a published generation. Every file named by an
evidence-package manifest must hold its own inode and the original verified
bytes after its input or predecessor path is edited in place.

The 2.0 numeric and mouse-reference writers already use independent copies
or APFS clones. Keep their package schemas, artifact paths and byte contents
unchanged. This PR changes only the publication operation and its tests; it
does not alter evidence labels, selectors, source counts or rights claims.

## Implementation

1. Add one `snapshot_file(source, target)` primitive. Require a new target,
   create its parent directory, follow a source symlink to the verified file
   bytes, and always check that source and target have different inodes.
   Prefer a copy-on-write clone (`cp -c` on macOS; `cp --reflink=auto` on
   Linux) for large artifacts. If cloning is unsupported or fails, remove
   any partial target and use `shutil.copy2`. Never call `os.link`.
2. Replace `_link_or_copy` throughout the MHC, context, supporting, external
   and T-cell writers. Use the same primitive for the already-independent
   numeric and reference successors so one invariant governs all seven
   generations. Keep atomic staging/rename, source rechecks, manifest format
   and all typed materializations unchanged.
3. Reproduce the reported failure before the change: mutate a v4 input after
   writing an MHC package and mutate a predecessor file after writing a
   context package. After the change, both published packages must retain
   original bytes, have distinct inodes and still verify. Exercise clone
   failure fallback and an existing-target refusal without a large fixture.
4. Bump to 2.0.1, run format/lint/full tests and focused package tests;
   review the diff and any real-package or storage check needed to resolve
   remaining risks. Open a PR against current main, require final-head CI,
   merge, run the clean-main release build and verify exact PyPI artifacts.

## Review questions

- Do all input and predecessor snapshot call sites now use the common helper?
- Can any fallback leave a partial target, mutate the source or share its inode?
- Are the original package bytes, counts and source manifest chain unchanged?
- Does a source mutation after publication leave the child generation valid?
- Is the large-corpus path a filesystem clone where supported and a bounded
  full copy only where necessary?
