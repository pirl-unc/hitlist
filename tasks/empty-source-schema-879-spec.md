# Empty source schemas — #879

The released `df.iloc[:0].copy()` pattern keeps full Arrow string buffers alive.
A bounded three-row probe retains the same 3,000,000-byte buffer in its zero-row
copy. The draft source result list can extend the lifetime to both source schemas
on the serial path. Correct this before shipping that lifetime extension.

Construct owned, empty columns with the original dtypes while preserving the
empty frame's index, column order and metadata. Apply this to serial source
results before retaining them for the next scan. The parallel IPC path already
transports empty schema values without the original buffers; leave that execution
path unchanged. Preserve categorical metadata because it is part of the schema.

Regression checks must compare frames, dtypes, index/column names and metadata,
and inspect actual Arrow buffer capacities, not just logical memory_usage/nbytes.
Keep existing actual-worker serial/parallel equivalence tests, full local tests,
format/lint and final-head hosted release gates.

The running full-build experiment remains bound to its immutable e88891a producer.
Before using its timings for the release, verify both candidate runs actually use
the unchanged parallel path without a serial retry. Explicitly document the later
serial-only ownership correction and do not claim a new measured serial memory or
speed improvement from the small retained-buffer test. If the experiment exercises
the changed serial path, requalification is required.

The conditional reuse of e88891a timings was superseded by #880: that candidate
retried serially and its experiment was cancelled. The replacement full producer
includes the owned-schema correction and compressed IPC, so final measurements
will use the complete current implementation.
