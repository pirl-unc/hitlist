"""Check the actual release artifacts before uploading them to PyPI (#409)."""

import sys
import tarfile
from email import message_from_bytes
from pathlib import Path
from zipfile import ZipFile


def check_metadata(data: bytes) -> None:
    metadata = message_from_bytes(data)
    assert metadata["License-Expression"] == "Apache-2.0", "missing SPDX license expression"
    assert "LICENSE" in metadata.get_all("License-File", []), "missing license-file metadata"


def check_distributions(dist_dir: Path) -> None:
    expected = (Path(__file__).resolve().parents[1] / "LICENSE").read_bytes()
    wheels = list(dist_dir.glob("*.whl"))
    sdists = list(dist_dir.glob("*.tar.gz"))
    assert len(wheels) == len(sdists) == 1, "expected exactly one wheel and one sdist"

    with ZipFile(wheels[0]) as wheel:
        metadata_path = next(p for p in wheel.namelist() if p.endswith(".dist-info/METADATA"))
        check_metadata(wheel.read(metadata_path))
        license_path = metadata_path.removesuffix("METADATA") + "licenses/LICENSE"
        assert wheel.read(license_path) == expected, "wheel license differs from repository"

    with tarfile.open(sdists[0]) as sdist:
        root = sdists[0].name.removesuffix(".tar.gz")
        metadata_file = sdist.extractfile(f"{root}/PKG-INFO")
        license_file = sdist.extractfile(f"{root}/LICENSE")
        assert metadata_file is not None and license_file is not None, "sdist metadata missing"
        check_metadata(metadata_file.read())
        assert license_file.read() == expected, "sdist license differs from repository"

    print("Wheel and sdist contain Apache-2.0 metadata and the exact LICENSE text.")


if __name__ == "__main__":
    check_distributions(Path(sys.argv[1]))
