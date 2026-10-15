"""Install the published, hash-pinned PXD013455 evidence without rebuilding."""

from pathlib import Path

import datacache
import yaml

from .downloads import data_dir
from .pxd013455_publication import verify_pxd013455_bundle

PUBLICATION = Path(__file__).parent / "data" / "pxd013455_publication.yaml"


def load_pxd013455_bundle(*, data_root=None, force=False):
    """Download/reuse the pinned portable generation in the shared data cache.

    Source reconstruction is separate. Failed downloads, extraction or inventory
    validation preserve the previous generation; verified reuse works offline.
    """
    with PUBLICATION.open() as handle:
        publication = yaml.safe_load(handle)
    root = Path(data_root or data_dir()) / "scoped-proteomics" / "pxd013455-hela-tryp46"
    return Path(
        datacache.install_archive(
            root / "published-download",
            publication["url"],
            expected_sha256=publication["sha256"],
            expected_size=publication["size_bytes"],
            max_members=64,
            max_extracted_size=256 * 2**20,
            download_options={"max_bytes": publication["size_bytes"]},
            force=force,
            validator=lambda tree: verify_pxd013455_bundle(
                tree, expected_sha256=publication["manifest_sha256"]
            ),
        )
    )
