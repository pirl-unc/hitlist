# Downloads and cache inspection

Hitlist requires datacache 1.14.0 or later. It supplies dataset definitions,
cache locations and biological metadata; datacache handles streaming transfers,
bounded retries, decompression, integrity checks and atomic publication.

```console
hitlist data list                 # registered datasets + mirrored data assets
hitlist data list --all           # also built indexes, proteomes and other cache files
hitlist data list --verify        # check trusted mirrored-asset SHA-256 digests
hitlist data list --all --json     # complete inventory, including provenance
hitlist data dirs                 # roots and the rules selecting them
```

The inventory spans the configured data root, platform asset cache, proteome
index root, existing legacy `~/.hitlist`, and externally registered paths.
Overlapping paths appear once. Missing named files remain visible. Hidden
transfer state, temporary files and cache metadata are omitted. Directory
enumeration failures are reported; symlink directories are not traversed.
Listing never creates directories, downloads files, repairs an installation or
moves old caches. `list_datasets()` retains its registered-dataset dictionary;
`downloads.list_cache_files()` supplies the new inventory.

`available` means the file is readable and matches any supplied size metadata.
It does not imply checksum verification. `verified` requires a trusted SHA-256
checked during this inspection. A `recorded_sha256` is a historical receipt,
not a fresh integrity check. Old cache hits gain no invented provenance; new
downloads record a source URL and fetch time. Datacache omits URL credentials,
queries and fragments from those receipts. Missing, corrupt and inaccessible
files have distinct statuses. Availability is not a check for remote updates.

Mirrored assets appear in `data available` and can be addressed by filename
with `data info`, `data path` and `data fetch`. Their registry supplies trusted
sizes and SHA-256 digests. These downloads now resume automatically after an
interruption on POSIX filesystems; other platforms use ordinary atomic downloads. Corrupt completed files require an explicit `--force` refresh.

```python
from hitlist.downloads import download_to_file

path = download_to_file(
    url, "reference.download", expected_size=size_bytes,
    expected_sha256=trusted_sha256, resume=True,
)
```

Raw bytes remain the default, including gzip, ZIP and HTML. Set `decompress=True`
to expand a URL literally ending in `.gz` or `.zip` when the destination does
not retain that archive suffix. Integrity expectations describe the installed
bytes. Resume supports raw HTTP(S) files on POSIX filesystems and requires an
expected size. Prefer a trusted hash. Without a hash, every accepted response
must carry a strong ETag; this protects against mixing server versions but is
not cryptographic verification. Size-only cache reuse does not detect same-size
corruption or check remote freshness. Do not enable resume for transformed
downloads without arranging a separate raw acquisition step.

Retries use datacache's bounded policy, including transient 408/429/5xx statuses
and Retry-After handling. Socket inactivity remains bounded at 300 seconds.
Empty responses are rejected. Failed acquisition, validation or decompression
leaves an existing destination intact. `download_to_file` returns a `Path` and
wraps failures in `RuntimeError` with the original exception as its cause.
