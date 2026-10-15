"""Admission of deposited bytes and scientific scope before label publication."""

import copy
import csv
import hashlib
import io
import json
import zlib
from pathlib import Path

import pandas as pd
import pytest
import yaml

from hitlist import pxd013455 as pxd
from hitlist import pxd013455_publication as publication
from hitlist.provenance import file_digest


def compressed_source(tmp_path, raw):
    compressor = zlib.compressobj(wbits=-15)
    path = tmp_path / "peptides.deflate"
    path.write_bytes(compressor.compress(raw) + compressor.flush())
    return path, {
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "crc32": f"{zlib.crc32(raw):08x}",
    }


@pytest.mark.parametrize(
    "raw",
    [b"a\nb\n", b"a\nb", b"a\r\nb\r\n", b"a" * 2**20 + b"\n"],
    ids=["lf", "no-final-newline", "crlf", "expansion-boundary"],
)
def test_bounded_deflate_is_complete_and_byte_preserving(tmp_path, raw):
    path, spec = compressed_source(tmp_path, raw)
    assert "".join(pxd._deflate_lines(path, spec)).encode() == raw


@pytest.mark.parametrize("damage", ["truncated", "trailing", "hash", "crc", "size", "line"])
def test_deflate_never_accepts_incomplete_or_unbounded_source(tmp_path, damage):
    path, spec = compressed_source(tmp_path, b"one\ntwo\n")
    if damage == "truncated":
        path.write_bytes(path.read_bytes()[:-2])
    elif damage == "trailing":
        path.write_bytes(path.read_bytes() + b"trailing")
    elif damage == "hash":
        spec["sha256"] = "0" * 64
    elif damage == "crc":
        spec["crc32"] = "00000000"
    elif damage == "size":
        spec["size_bytes"] -= 1
    with pytest.raises(ValueError):
        list(pxd._deflate_lines(path, spec, max_line_bytes=2 if damage == "line" else 100))


@pytest.fixture
def source_rows():
    scope = pxd._catalog()["experiment"]
    common = {
        "Sequence": "ACDEFGK",
        "Leading razor protein": "P1",
        "Proteins": "P1;P2",
        "Start position": "1",
        "End position": "7",
        "Gene names": "GENE",
        "id": "0",
        "Reverse": "",
        "Potential contaminant": "",
        "PEP": "0.001",
        "Score": "100",
        f"Intensity {scope}": "100",
        f"Identification type {scope}": "By MS/MS",
    }
    return [
        common,
        {**common, "Sequence": "LMNPQRR", "id": "1", f"Identification type {scope}": "By matching"},
        {**common, "id": "2", "Reverse": "+"},
        {**common, "id": "3", "Potential contaminant": "+"},
        {**common, "id": "4", f"Intensity {scope}": "0"},
    ]


def observation_fixture(tmp_path, rows):
    catalog = copy.deepcopy(pxd._catalog())
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
    path, spec = compressed_source(tmp_path, stream.getvalue().encode())
    catalog["members"]["peptides"].update(spec)
    catalog["expected_counts"].update(
        n_source_rows=5,
        n_positive_intensity_rows=4,
        n_reverse_or_contaminant_rows=2,
        n_observations=2,
        n_msms_observations=1,
        n_matching_observations=1,
        n_observed_parents=1,
    )
    return path, catalog


def test_observations_preserve_evidence_and_leave_replication_unresolved(tmp_path, source_rows):
    path, catalog = observation_fixture(tmp_path, source_rows)
    frame, parents, counts = pxd._observations(path, catalog)
    assert frame.peptide.tolist() == ["ACDEFGK", "LMNPQRR"]
    assert frame.identification_type.tolist() == ["By MS/MS", "By matching"]
    assert frame.evidence_row_id.tolist() == [
        f"PXD013455:txt-celllines.zip:peptides:{i}" for i in range(2)
    ]
    assert frame.source_file_sha256.eq(catalog["members"]["peptides"]["sha256"]).all()
    assert frame.n_replicates_detected.isna().all()
    assert frame.n_replicates_possible.eq(1).all()
    assert frame.fractionation_ph.eq(8.0).all()
    assert parents.n_peptides.tolist() == [2]
    assert parents.assigned_intensity_sum.tolist() == [200]
    assert counts["n_reverse_or_contaminant_rows"] == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", ""),
        ("Leading razor protein", "P1;P2"),
        ("Sequence", "ACDEXGK"),
        ("Reverse", "?"),
        ("Potential contaminant", "True"),
        ("intensity", "nan"),
        ("intensity", "-1"),
        ("identification", "inferred"),
    ],
)
def test_invalid_source_evidence_is_not_silently_curated(tmp_path, source_rows, field, value):
    scope = pxd._catalog()["experiment"]
    field = {
        "intensity": f"Intensity {scope}",
        "identification": f"Identification type {scope}",
    }.get(field, field)
    source_rows[0][field] = value
    path, catalog = observation_fixture(tmp_path, source_rows)
    old_limit = csv.field_size_limit()
    with pytest.raises(ValueError):
        pxd._observations(path, catalog)
    assert csv.field_size_limit() == old_limit


@pytest.mark.parametrize("field", ["id", "Sequence"])
def test_duplicate_observation_identity_fails(tmp_path, source_rows, field):
    source_rows[1][field] = source_rows[0][field]
    path, catalog = observation_fixture(tmp_path, source_rows)
    with pytest.raises(ValueError, match="Ambiguous observation"):
        pxd._observations(path, catalog)


def test_missing_expected_evidence_category_is_a_count_failure(tmp_path, source_rows):
    scope = pxd._catalog()["experiment"]
    source_rows[1][f"Identification type {scope}"] = "By MS/MS"
    path, catalog = observation_fixture(tmp_path, source_rows)
    with pytest.raises(ValueError, match="Unexpected n_msms_observations"):
        pxd._observations(path, catalog)


@pytest.fixture
def protocol_inputs(tmp_path):
    catalog = copy.deepcopy(pxd._catalog())
    catalog["expected_counts"]["n_raw_files"] = 2
    catalog["scope"]["n_fractions_in_run"] = 2
    scope = catalog["experiment"]
    params = {
        "Version": "1.6.0.16",
        "Fixed modifications": "Carbamidomethyl (C)",
        "Min. peptide Length": "7",
        "Max. peptide mass [Da]": "4600",
        "Match between runs": "True",
        "Fasta file": "C:\\data\\" + catalog["files"]["searched.fasta"]["original_name"],
    }
    paths = {
        name: tmp_path / name
        for name in (
            "parameters.txt",
            "summary.txt",
            "mqpar-celllines.xml",
            "experimentalDesignTemplate-celllines.txt",
        )
    }
    with paths["parameters.txt"].open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerows([("Parameter", "Value"), *params.items()])
    rows = [
        {
            "Raw file": f"run{i}",
            "Fraction": str(i),
            "Experiment": scope,
            "Enzyme": "Trypsin/P",
            "Max. missed cleavages": "2",
            "Enzyme mode": "Specific",
            "LC-MS run type": "Standard",
        }
        for i in (1, 2)
    ]
    pd.DataFrame(rows).to_csv(paths["summary.txt"], sep="\t", index=False)
    pd.DataFrame(
        [
            {"Name": row["Raw file"], "Fraction": row["Fraction"], "Experiment": scope}
            for row in rows
        ]
    ).to_csv(paths["experimentalDesignTemplate-celllines.txt"], sep="\t", index=False)
    paths["mqpar-celllines.xml"].write_text(
        "<root><maxQuantVersion>1.6.0.1</maxQuantVersion>"
        "<filePaths><string>C:\\run1.raw</string><string>C:\\run2.raw</string></filePaths>"
        f"<experiments><string>{scope}</string><string>{scope}</string></experiments>"
        "<fractions><short>1</short><short>2</short></fractions>"
        "<paramGroupIndices><int>0</int><int>0</int></paramGroupIndices>"
        "<parameterGroups><group><enzymes><string>Trypsin/P</string></enzymes>"
        "<maxMissedCleavages>2</maxMissedCleavages></group></parameterGroups></root>"
    )
    return paths, catalog


def test_protocol_preserves_versions_and_does_not_call_fractions_replicates(protocol_inputs):
    runs, report = pxd._protocol(*protocol_inputs)
    assert len(runs) == report["n_fractions"] == report["n_raw_files"] == 2
    assert report["effective_maxquant_version"] == "1.6.0.16"
    assert report["standalone_xml_maxquant_version"] == "1.6.0.1"
    assert report["biological_replicate_identity"] == "unresolved"


@pytest.mark.parametrize(
    "filename,old,new",
    [
        ("parameters.txt", "4600", "5000"),
        ("parameters.txt", "71591.fasta", "other.fasta"),
        ("summary.txt", "Specific", "Unspecific"),
        ("summary.txt", "Standard", "Dependent peptides"),
        ("mqpar-celllines.xml", "run2.raw", "run3.raw"),
        ("mqpar-celllines.xml", "<short>2</short>", "<short>1</short>"),
        ("mqpar-celllines.xml", "<string>Trypsin/P</string>", "<string>Trypsin</string>"),
        ("experimentalDesignTemplate-celllines.txt", "run2", "run3"),
    ],
)
def test_protocol_disagreement_rejects_scope(protocol_inputs, filename, old, new):
    paths, catalog = protocol_inputs
    paths[filename].write_text(paths[filename].read_text().replace(old, new))
    with pytest.raises(ValueError):
        pxd._protocol(paths, catalog)


def test_published_observations_use_complete_datacache_generation(
    tmp_path, monkeypatch, source_rows, protocol_inputs
):
    path, catalog = observation_fixture(tmp_path, source_rows)
    paths, protocol_catalog = protocol_inputs
    catalog["expected_counts"]["n_raw_files"] = 2
    catalog["scope"] = protocol_catalog["scope"]
    fasta = tmp_path / "searched.fasta"
    fasta.write_text(">sp|P1|test\nACDEFGKLMNPQRR\n")
    paths.update({"searched.fasta": fasta, "peptides.deflate": path})
    # The fixture has no external paper dependencies. Source admission itself is
    # tested separately; this exercises actual materialization/publication.
    catalog["supplement_tables"] = {}
    for name in catalog["members"]:
        receipt = tmp_path / f"{name}.receipt.json"
        receipt.write_text("{}\n")
        paths[receipt.name] = receipt
    monkeypatch.setattr(pxd, "_catalog", lambda: copy.deepcopy(catalog))
    monkeypatch.setattr(pxd, "recover_pxd013455_sources", lambda **kw: paths)
    outputs = pxd.build_pxd013455_observations(data_root=tmp_path)
    assert pd.read_parquet(outputs["observations.parquet"]).shape[0] == 2
    assert json.loads(Path(outputs["coverage.json"]).read_text())["n_observed_parents"] == 1
    monkeypatch.setattr(pxd, "_observations", lambda *args: pytest.fail("Cache hit rebuilt data"))
    assert pxd.build_pxd013455_observations(data_root=tmp_path) == outputs


@pytest.fixture
def publication_bundle(tmp_path, monkeypatch, source_rows):
    path, catalog = observation_fixture(tmp_path, source_rows)
    catalog["expected_counts"].update(
        n_raw_files=2,
        n_reference_sequences=1,
        n_candidate_occurrences=4,
        n_observed_candidate_occurrences=2,
        n_distinct_observed_candidate_sequences=2,
    )
    # The publication API has one deliberately fixed 46-fraction scope. The
    # small observation builder fixture need not reproduce its 46 raw files.
    fasta = tmp_path / "searched.fasta"
    fasta.write_text(">sp|P1|test\nACDEFGKLMNPQRR\n")
    catalog["search_space"]["fasta_sha256"] = file_digest(fasta)["sha256"]
    frame, parents, counts = pxd._observations(path, catalog)
    observed = {"searched.fasta": str(fasta)}
    for name, table in {"observations.parquet": frame, "proteins.parquet": parents}.items():
        target = tmp_path / name
        table.to_parquet(target, index=False)
        observed[name] = str(target)
    for name, data in {"source_registry.json": catalog, "coverage.json": counts}.items():
        target = tmp_path / name
        target.write_text(json.dumps(data))
        observed[name] = str(target)
    monkeypatch.setattr(publication, "build_pxd013455_observations", lambda **kw: observed)
    return publication.build_pxd013455_bundle(data_root=tmp_path)


def test_complete_publication_has_auditable_links_and_verified_portable_bytes(
    publication_bundle, tmp_path
):
    import tarfile

    root = Path(publication_bundle["manifest.json"]).parent
    manifest = publication.verify_pxd013455_bundle(
        root, expected_sha256=file_digest(root / "manifest.json")["sha256"]
    )
    assert manifest["training_readiness"] == "not_established"
    audit = publication.audit_pxd013455_candidates(root)
    assert audit["n_candidate_occurrences"] == 4
    assert audit["partition_status"] == "not_assigned"
    frame = pd.read_parquet(root / "candidates.parquet")
    assert frame.candidate_id.is_unique
    assert frame.loc[frame.observed, "source_observation_id"].notna().all()
    assert frame.loc[~frame.observed, "source_observation_id"].isna().all()
    archive = publication.export_pxd013455_archive(data_root=tmp_path)
    with tarfile.open(archive) as handle:
        assert sorted(handle.getnames()) == sorted(publication_bundle)
        assert handle.extractfile("manifest.json").read() == (root / "manifest.json").read_bytes()
    assert publication.export_pxd013455_archive(data_root=tmp_path) == archive


@pytest.mark.parametrize(
    "field,value",
    [
        ("n_flank", "WRONG"),
        ("observed", False),
        ("start_position", 2),
        ("source_observation_id", "invented"),
        ("candidate_id", "wrong"),
        ("first_seen_at_n_fractions", 12),
        ("n_replicates_detected", 1),
        ("peptide_mass_da", 1.0),
        ("fractionation_ph", 10.0),
    ],
)
def test_independent_audit_catches_mutated_labels_or_coordinates(publication_bundle, field, value):
    root = Path(publication_bundle["manifest.json"]).parent
    path = root / "candidates.parquet"
    frame = pd.read_parquet(path)
    frame.loc[frame.index[frame.observed][0], field] = value
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError):
        publication.audit_pxd013455_candidates(root)
    with pytest.raises(ValueError, match="mismatch"):
        publication.verify_pxd013455_bundle(root)


def test_publication_failure_retains_previous_generation(publication_bundle, monkeypatch, tmp_path):
    manifest = Path(publication_bundle["manifest.json"])
    before = manifest.read_bytes()

    def fail(*args):
        raise ValueError("independent audit failed")

    monkeypatch.setattr(publication, "audit_pxd013455_candidates", fail)
    with pytest.raises(ValueError, match="independent audit failed"):
        publication.build_pxd013455_bundle(data_root=tmp_path, force=True)
    assert manifest.read_bytes() == before
    assert publication.build_pxd013455_bundle(data_root=tmp_path) == publication_bundle


@pytest.mark.parametrize("damage", ["pin", "missing", "path", "symlink", "changed"])
def test_public_inventory_rejects_missing_pins_paths_and_mutation(
    publication_bundle, monkeypatch, damage
):
    root = Path(publication_bundle["manifest.json"]).parent
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    if damage == "missing":
        del manifest["artifacts"]["candidates.parquet"]
        path.write_text(json.dumps(manifest))
    elif damage == "path":
        manifest["artifacts"]["../outside"] = manifest["artifacts"].pop("candidates.parquet")
        path.write_text(json.dumps(manifest))
    elif damage == "symlink":
        target = root / "README.md"
        saved = root / "saved-readme"
        target.rename(saved)
        target.symlink_to(saved)
    elif damage == "changed":
        original = publication.datacache.validate_file

        def change_after_validation(file, *args, **kwargs):
            result = original(file, *args, **kwargs)
            if Path(file).name == "README.md":
                path.write_text(path.read_text() + " ")
            return result

        monkeypatch.setattr(publication.datacache, "validate_file", change_after_validation)
    with pytest.raises(ValueError):
        publication.verify_pxd013455_bundle(
            root, expected_sha256="0" * 64 if damage == "pin" else None
        )


def test_published_download_is_shared_pinned_and_preserves_previous_generation(
    publication_bundle, tmp_path, monkeypatch
):
    from hitlist import pxd013455_download as download

    archive = Path(publication.export_pxd013455_archive(data_root=tmp_path))
    registry = tmp_path / "published.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "url": archive.as_uri(),
                **file_digest(archive),
                "manifest_sha256": file_digest(publication_bundle["manifest.json"])["sha256"],
            }
        )
    )
    monkeypatch.setattr(download, "PUBLICATION", registry)
    directory = download.load_pxd013455_bundle(data_root=tmp_path)
    assert (
        publication.verify_pxd013455_bundle(directory)["dataset_id"] == "pxd013455-hela-tryp46-v1"
    )
    assert download.load_pxd013455_bundle(data_root=tmp_path) == directory
    original = (directory / "candidates.parquet").read_bytes()
    archive.write_bytes(b"incomplete refresh")
    with pytest.raises(ValueError):
        download.load_pxd013455_bundle(data_root=tmp_path, force=True)
    assert (directory / "candidates.parquet").read_bytes() == original
    assert download.load_pxd013455_bundle(data_root=tmp_path) == directory
