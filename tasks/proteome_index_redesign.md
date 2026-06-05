# ProteomeIndex redesign — int-encoded columnar k-mer index

Closes the converging perf trio: **#250** (cold-build hot loop), **#248**
(warm-load format), **#176/#63** (build profiling). Replaces the
`index: dict[str, int | np.ndarray]` (k-mer string → packed-int64 postings)
with a per-length **int-encoded columnar** structure.

## Why (benchmark, Macaca L9, 11M k-mers)

- pickle warm load: 23s; of which raw load 4.75s + **Python dict rebuild 8.5s**.
- The dict rebuild is unavoidable while the lookup API needs `dict[str, …]`.
- Eliminating the Python dict (mmap'd columnar arrays + binary search) is the
  only path to #248's <5s **and** #250's faster build **and** kills the
  `dict.get` hot path (#176's #1 cProfile entry, 1.17B calls).

## Encoding (verified safe for real data)

- All callers build with `lengths=(8,9,10,11)` → **max k = 11**.
- Real proteome alphabet = 21 chars (`ACDEFGHIKLMNPQRSTVWXY`); ≤ 32.
- **5 bits/residue**, k ≤ 11 → ≤ 55 bits ≤ 63: a k-mer packs into one
  **uint64**, collision-free. `code = Σ aa_code[c_i] << 5*(k-1-i)`.
- Alphabet built dynamically from observed residues, **stored in the index**
  (for decode + determinism across build/load).
- **Safety fallback**: if `alphabet_size > 2**bits_budget` or
  `max(lengths) * bits_budget > 63`, fall back to the legacy `dict[str,…]`
  build/query path. Never triggers for real data; guarantees correctness for
  pathological inputs (odd residues, huge lengths). Keep `_build_legacy`.

## New internal structure

Per indexed length `k`, three mmap-friendly arrays (code-sorted):
- `codes[k]`: `uint64[n_k]` — unique k-mer codes, **sorted ascending**.
- `offsets[k]`: `int64[n_k + 1]` — postings slice bounds.
- `values[k]`: `int64[total_k]` — packed `(prot_idx<<32 | pos)` postings,
  grouped by code in `codes[k]` order.

Unchanged dataclass fields: `proteins`, `protein_meta`, `lengths`,
`_protein_ids`. The `index` dict is replaced by a `_PackedIndex` holding the
per-length arrays + the alphabet table (+ bits/char). `_pack`/`_unpack`/
`_PROT_BITS=32` preserved exactly.

## Build (vectorized — #250)

Per protein, per length `k` (all numpy, no Python inner loop):
1. `seq` bytes → `c = lut[np.frombuffer(seq.encode(), uint8)]` (per-residue
   5-bit codes; `lut` maps the alphabet, sentinel for out-of-alphabet →
   triggers fallback).
2. Rolling pack: `code = 0; for j in range(k): code = (code<<5) | c[j : j+L-k+1]`
   → `kcodes` (`uint64[L-k+1]`), k vectorized shifts (k ≤ 11).
3. `packed = (prot_idx<<32) | np.arange(L-k+1, dtype=int64)`.
4. Append `kcodes`, `packed` to per-length lists.

After all proteins, per length: concat → `order = np.argsort(all_codes, kind="stable")`
→ sort both → group by `np.diff` boundaries → `codes`, `offsets`, `values`.
Replaces ~100M Python dict ops with vectorized ops + one O(n log n) C sort.

## Query (preserve public API)

`lookup(peptide) -> list[tuple[protein_id, pos]]`:
1. `k = len(peptide)`; if `k not in lengths` → `[]`.
2. encode peptide → `code` (fallback path: dict lookup); out-of-alphabet → `[]`.
3. `i = np.searchsorted(codes[k], code)`; if in-range and `codes[k][i]==code`:
   `vals = values[k][offsets[k][i]:offsets[k][i+1]]`; map each via
   `_protein_ids[v>>32], v & POS_MASK`. Else `[]`.

`map_peptides` already delegates to `lookup` — unchanged. `all_kmers` →
decode every code back to a string (cached_property, rare use). `merge` →
rebuild via `_build` (unchanged call site).

## Serialization (#248)

Disk cache becomes the columnar arrays via **Arrow IPC + `memory_map`** (or
`.npz`), zero-copy load — **no Python dict rebuild**. Small sidecar (alphabet,
lengths, `_protein_ids`, `proteins`, `protein_meta`) via pickle/marshal.
Bump `_INDEX_FORMAT_VERSION` → 2 (v1 pickles evicted by the LRU cap).
Target: Macaca L9 warm load **< 5s**.

## Staging (separate PRs, each green against the 25 test_proteome.py tests)

- **PR A (#250)** — int-encoding + vectorized columnar build + query, in
  memory. Legacy fallback retained. Pickle serialization still works (arrays
  pickle fine). Delivers cold-build speedup. Contract: all 25 tests pass.
- **PR B (#248)** — Arrow IPC + mmap disk-cache format; `_INDEX_FORMAT_VERSION=2`.
  Delivers warm-load <5s.
- **PR C (#176/#63)** — re-profile a build with the new index; address any
  remaining hot spots; close #176/#63 with before/after numbers.

## Contract (must stay green)

`lookup` return shape; `map_peptides` columns + flank/`n_sources`/`unique_*`
logic; `all_kmers` frozenset; `merge` renumbering; from_fasta LRU identity;
disk-cache round-trip integrity + corrupt-file fallback. 25 tests in
`tests/test_proteome.py`.
