"""Hash-verified gzip transport for large pinned qualification source exports."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / "tasks/corpus-qualification-sources.json"
app = modal.App("hitlist-qualification-source-transport")
volume = modal.Volume.from_name("presto-data", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.12")


def file_pin(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            value.update(block)
    return {"sha256": value.hexdigest(), "size_bytes": Path(path).stat().st_size}


def compress(source, name, cache):
    from datacache import materialize

    pin = json.loads(PINS.read_text())[name]
    filename = pin["filename"] + ".gz"
    recipe = file_pin(__file__)["sha256"]

    def build(inputs, outputs):
        digest = hashlib.sha256()
        n_bytes = 0
        with (
            Path(source).open("rb") as raw,
            Path(outputs[filename]).open("wb") as target,
            gzip.GzipFile(
                filename="", fileobj=target, mode="wb", mtime=0, compresslevel=1
            ) as zipped,
        ):
            for block in iter(lambda: raw.read(2**20), b""):
                digest.update(block)
                n_bytes += len(block)
                zipped.write(block)
        if n_bytes != pin["size_bytes"] or digest.hexdigest() != pin["sha256"]:
            raise ValueError("Raw source differs from the declared qualification pin")
        Path(outputs["source_receipt.json"]).write_text(
            json.dumps(
                {"raw_source": pin, "gzip": file_pin(outputs[filename]), "recipe_sha256": recipe},
                indent=2,
            )
            + "\n"
        )

    paths = materialize(
        Path(cache) / pin["sha256"] / recipe,
        {"source-pins.json": {"path": str(PINS), "sha256": file_pin(PINS)["sha256"]}},
        transform={
            "version": "verified-export-gzip-v1",
            "options": {"source": name, "recipe_sha256": recipe},
        },
        outputs={filename: {}, "source_receipt.json": {}},
        builder=build,
    )
    return paths, pin, filename


@app.function(
    image=image,
    volumes={"/data": volume},
    cpu=(2, 2),
    memory=(4096, 4096),
    timeout=1800,
    max_containers=1,
    retries=0,
)
def expand(remote_gzip, gzip_pin, source_pin):
    source = Path("/data") / remote_gzip
    target = Path("/data") / source_pin["remote_path"]
    if file_pin(source) != gzip_pin:
        raise ValueError("Compressed transport failed verification")
    target.parent.parent.mkdir(parents=True, exist_ok=True)

    def existing():
        actual = file_pin(target)
        if any(actual[key] != source_pin[key] for key in ("sha256", "size_bytes")):
            raise ValueError("A different existing raw input will not be overwritten")
        return {"path": str(target), **actual, "reused": True}

    if target.exists():
        return existing()
    with tempfile.TemporaryDirectory(prefix=".expand-", dir=target.parent.parent) as directory:
        staged = Path(directory) / target.name
        with staged.open("wb") as output, gzip.open(source, "rb") as compressed:
            n_bytes = 0
            for block in iter(lambda: compressed.read(2**20), b""):
                n_bytes += len(block)
                if n_bytes > source_pin["size_bytes"]:
                    raise ValueError("Expanded source exceeds its declared size")
                output.write(block)
        actual = file_pin(staged)
        if any(actual[key] != source_pin[key] for key in ("sha256", "size_bytes")):
            raise ValueError("Expanded source failed its original content pin")
        # Atomic directory publication cannot replace a concurrent nonempty
        # directory. Canonical reuse begins at the producer's datacache install.
        try:
            os.rename(directory, target.parent)
        except OSError:
            if target.exists():
                return existing()
            raise
        volume.commit()
        return {"path": str(target), **actual, "reused": False}


@app.local_entrypoint()
def main(source: str, name: str = "cedar", cache: str = "", receipt: str = ""):
    from hitlist.downloads import data_dir

    paths, pin, filename = compress(
        source, name, cache or str(data_dir() / "corpus-qualification" / "transports")
    )
    packed = Path(paths[filename])
    packed_pin = file_pin(packed)
    remote = f"hitlist-qualification/raw-transports/{packed_pin['sha256']}/{filename}"
    try:
        with volume.batch_upload(force=False) as batch:
            batch.put_file(str(packed), remote)
    except FileExistsError:
        pass  # The remote function must still verify the exact compressed bytes.
    result = expand.remote(remote, packed_pin, pin)
    result["transport"] = {"remote_path": remote, "local_path": str(packed), **packed_pin}
    print(json.dumps(result, indent=2))
    if receipt:
        Path(receipt).write_text(json.dumps(result, indent=2) + "\n")
