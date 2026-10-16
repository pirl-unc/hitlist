"""Transport and bounded CPU execution for the isolated corpus qualification."""

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("hitlist-corpus-qualification")
volume = modal.Volume.from_name("presto-data", create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(
        str(ROOT / "tasks/corpus-qualification-requirements.txt"),
        extra_options="--require-hashes",
    )
    .run_commands("python -m pip check")
    .add_local_file(str(ROOT / "scripts/qualify_corpus.py"), "/opt/qualification/qualify_corpus.py")
    .add_local_file(str(ROOT / "scripts/ci_corpus.py"), "/opt/qualification/ci_corpus.py")
    .add_local_file(
        str(ROOT / "tasks/corpus-qualification-sources.json"), "/opt/qualification/sources.json"
    )
    .add_local_file(
        str(ROOT / "tasks/corpus-qualification-requirements.txt"),
        "/opt/qualification/requirements.txt",
    )
)


@app.function(
    image=image,
    volumes={"/data": volume},
    cpu=(8, 8),
    memory=(65536, 65536),
    ephemeral_disk=200000,
    timeout=5400,
    max_containers=1,
    retries=0,
)
def execute(stage: str):
    import subprocess
    import sys

    if stage not in ("environment", "sources", "indexes", "reference", "mappings", "snapshot"):
        raise ValueError(stage)
    try:
        subprocess.run([sys.executable, "/opt/qualification/qualify_corpus.py", stage], check=True)
    finally:
        volume.commit()


@app.local_entrypoint()
def main(stage: str = "environment"):
    execute.remote(stage)
