# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for UniProt proteome UPID correctness and FASTA taxon guards (#977)."""

from __future__ import annotations

import re

import pytest

import hitlist
from hitlist import downloads
from hitlist.errors import IntegrityError

_UPID_RE = re.compile(r"^UP\d{9}$")

# Known-wrong UPIDs discovered in #977 (poplar, archaeon, influenza A,
# fungus, sesame seed, wrong HPV/rhabdovirus, etc.) that must NEVER appear
# anywhere in FETCHABLE_DATASETS, SPECIES_PROTEOMES, or VIRAL_PROTEOMES.
_FORBIDDEN_WRONG_UPIDS: dict[str, str] = {
    "UP000006729": "Populus trichocarpa (was hpv16)",
    "UP000006728": "Saccharopolyspora erythraea (was hpv18)",
    "UP000002063": "Methanocaldococcus vulcanius (was htlv1)",
    "UP000126453": "Influenza A virus (was hbv)",
    "UP000009113": "Human papillomavirus type 26 (was kshv)",
    "UP000116695": "Influenza A virus (was mcpyv)",
    "UP000117312": "Influenza A virus (was canine distemper virus)",
    "UP000006706": "Aspergillus niger (was PRRSV)",
    "UP000147130": "Influenza A virus (was wobbly possum disease virus)",
    "UP000007212": "Infectious hematopoietic necrosis virus (was alphapolyomavirus muris)",
    "UP000504604": "Sesamum indicum (was Trichosurus vulpecula)",
    "UP000002254": "Deprecated Non-Reference Canis lupus proteome (0-byte UniProtKB stream)",
}


def test_public_exports_exposed():
    assert hitlist.validate_uniprot_fasta_headers is downloads.validate_uniprot_fasta_headers
    assert hitlist.audit_cached_proteomes is downloads.audit_cached_proteomes


def test_no_known_wrong_upids_in_any_registry():
    for name, ds in downloads.FETCHABLE_DATASETS.items():
        url = ds.get("url", "")
        upid = ds.get("proteome_id", "")
        for bad_upid, reason in _FORBIDDEN_WRONG_UPIDS.items():
            assert bad_upid not in url, (
                f"FETCHABLE_DATASETS[{name!r}] URL still uses {bad_upid}: {reason}"
            )
            assert upid != bad_upid, f"FETCHABLE_DATASETS[{name!r}] still uses {bad_upid}: {reason}"

    for species, entry in downloads.SPECIES_PROTEOMES.items():
        upid = entry.get("proteome_id", "")
        for bad_upid, reason in _FORBIDDEN_WRONG_UPIDS.items():
            assert upid != bad_upid, (
                f"SPECIES_PROTEOMES[{species!r}] still uses {bad_upid}: {reason}"
            )

    for virus, entry in downloads.VIRAL_PROTEOMES.items():
        upid = entry.get("proteome_id", "")
        for bad_upid, reason in _FORBIDDEN_WRONG_UPIDS.items():
            assert upid != bad_upid, f"VIRAL_PROTEOMES[{virus!r}] still uses {bad_upid}: {reason}"


def test_specific_977_corrections_and_possum_removal():
    # Brushtail possum has no UniProt proteome (UP000504604 was sesame seed).
    assert "Trichosurus vulpecula" not in downloads.SPECIES_PROTEOMES
    assert downloads.lookup_proteome("Trichosurus vulpecula") is None

    # Ascaris suum uses its own species proteome UP001319243 (OX=6253), not A. lumbricoides UP000036681.
    asuum = downloads.lookup_proteome("Ascaris suum")
    alumb = downloads.lookup_proteome("Ascaris lumbricoides")
    assert asuum is not None and alumb is not None
    assert asuum["proteome_id"] == "UP001319243"
    assert asuum["expected_taxon_id"] == 6253
    assert alumb["proteome_id"] == "UP000036681"
    assert alumb["expected_taxon_id"] == 6252

    # Canis lupus / Canis sp. use active Reference proteome UP000805418 (UP000002254 streams 0 bytes).
    for name in ("Canis lupus", "Canis sp."):
        dog = downloads.lookup_proteome(name)
        assert dog is not None
        assert dog["proteome_id"] == "UP000805418"
        assert dog["expected_taxon_id"] == 9615

    # AAV-6 and AAV-8 use organism_id queries because their UPIDs are Non-Reference (0-byte stream).
    aav6 = downloads.lookup_proteome("Adeno-associated virus 6")
    aav8 = downloads.lookup_proteome("Adeno-associated virus 8")
    assert aav6 is not None and aav6["uniprot_query"] == "organism_id:68558"
    assert aav8 is not None and aav8["uniprot_query"] == "organism_id:202813"

    # All 10 viral UPID fixes verified against UniProt REST API.
    expected_viral = {
        "Human papillomavirus type 16": ("UP000009251", 333760),
        "Human papillomavirus type 18": ("UP000009109", 333761),
        "Human T-lymphotropic virus 1": ("UP000001061", 11927),
        "Hepatitis B virus": ("UP000007930", 490133),
        "Human gammaherpesvirus 8": ("UP000000942", 868565),
        "Merkel cell polyomavirus": ("UP000170366", 493803),
        "Canine distemper virus": ("UP000007634", 11233),
        "Porcine reproductive and respiratory syndrome virus": ("UP000006687", 11049),
        "Wobbly possum disease virus": ("UP000203107", 1118369),
        "Alphapolyomavirus muris": ("UP000116380", 179241),
    }
    for organism, (upid, taxon_id) in expected_viral.items():
        entry = downloads.lookup_proteome(organism)
        assert entry is not None, f"lookup_proteome({organism!r}) returned None"
        assert entry["proteome_id"] == upid
        assert entry["expected_taxon_id"] == taxon_id


def test_all_curated_uniprot_entries_have_valid_upid_and_expected_taxon():
    for species, entry in downloads.SPECIES_PROTEOMES.items():
        if entry.get("kind") != "uniprot":
            continue
        assert _UPID_RE.match(entry["proteome_id"]), f"Bad UPID for {species!r}"
        assert isinstance(entry.get("expected_taxon_id"), int) and entry["expected_taxon_id"] > 0

    for virus, entry in downloads.VIRAL_PROTEOMES.items():
        assert _UPID_RE.match(entry["proteome_id"]), f"Bad UPID for {virus!r}"
        assert isinstance(entry.get("expected_taxon_id"), int) and entry["expected_taxon_id"] > 0

    viral_by_key = {v["key"]: v for v in downloads.VIRAL_PROTEOMES.values()}
    for name, ds in downloads.FETCHABLE_DATASETS.items():
        url = ds.get("url", "")
        if "proteome:UP" not in url:
            continue
        upid = ds.get("proteome_id")
        assert upid and _UPID_RE.match(upid), (
            f"Missing/bad proteome_id on FETCHABLE_DATASETS[{name!r}]"
        )
        assert f"proteome:{upid}" in url, (
            f"URL/proteome_id mismatch on FETCHABLE_DATASETS[{name!r}]"
        )
        assert isinstance(ds.get("expected_taxon_id"), int) and ds["expected_taxon_id"] > 0
        if name in viral_by_key:
            assert viral_by_key[name]["proteome_id"] == upid
            assert viral_by_key[name]["expected_taxon_id"] == ds["expected_taxon_id"]


def test_validate_uniprot_fasta_headers_accepts_matching_and_rejects_wrong_taxon(tmp_path):
    hpv16_good = tmp_path / "hpv16.fasta"
    hpv16_good.write_text(
        ">sp|P03126|VE6_HPV16 Protein E6 OS=Human papillomavirus 16 OX=333760 GN=E6 PE=1 SV=1\n"
        "MHQKRTAMFQDPQERPRKLPQLCTELQTTI\n"
    )
    summary = downloads.validate_uniprot_fasta_headers(
        hpv16_good,
        expected_taxon_id=333760,
        expected_organism="Human papillomavirus type 16",
        require_taxon_header=True,
    )
    assert summary["n_headers_checked"] == 1
    assert summary["observed_taxon_ids"] == [333760]
    assert summary["observed_organisms"] == ["Human papillomavirus 16"]
    assert summary["has_taxon_headers"] is True

    # Poisoned poplar proteome downloaded under hpv16 name must fail loudly.
    poplar_poison = tmp_path / "hpv16_poplar.fasta"
    poplar_poison.write_text(
        ">tr|A0A2K1X123|A0A2K1X123_POPTR Uncharacterized protein OS=Populus trichocarpa OX=3694 PE=4 SV=1\n"
        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ\n"
    )
    with pytest.raises(IntegrityError, match=r"taxon mismatch.*333760.*3694"):
        downloads.validate_uniprot_fasta_headers(
            poplar_poison,
            expected_taxon_id=333760,
            expected_organism="Human papillomavirus type 16",
        )

    # Empty or HTML error body fails even without expected_taxon_id.
    html_err = tmp_path / "error.fasta"
    html_err.write_text("<html><body>503 Service Unavailable</body></html>\n")
    with pytest.raises(IntegrityError, match="contains no FASTA headers"):
        downloads.validate_uniprot_fasta_headers(html_err, expected_taxon_id=333760)

    # Synthetic FASTA without OX= passes when require_taxon_header=False, fails when True.
    synthetic = tmp_path / "synthetic.fasta"
    synthetic.write_text(">sp|P1|SYNTH\nMKVFLVLLCL\n")
    ok = downloads.validate_uniprot_fasta_headers(synthetic, expected_taxon_id=333760)
    assert ok["n_headers_checked"] == 1
    assert ok["has_taxon_headers"] is False
    with pytest.raises(IntegrityError, match="no UniProt OX="):
        downloads.validate_uniprot_fasta_headers(
            synthetic, expected_taxon_id=333760, require_taxon_header=True
        )


def test_fetch_species_proteome_rejects_wrong_taxon_download_and_refetches_poisoned_cache(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path)

    poplar_body = (
        ">tr|A0A2K1|POPTR Uncharacterized protein OS=Populus trichocarpa OX=3694 PE=4 SV=1\n"
        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ\n"
    )
    hpv16_body = (
        ">sp|P03126|VE6_HPV16 Protein E6 OS=Human papillomavirus 16 OX=333760 GN=E6 PE=1 SV=1\n"
        "MHQKRTAMFQDPQERPRKLPQLCTELQTTI\n"
    )

    # 1. Server returns wrong organism -> rejected before landing at final path or manifest.
    def fake_bad_download(_url, dest, **_kw):
        dest.write_text(poplar_body)
        return dest

    monkeypatch.setattr(downloads, "download_to_file", fake_bad_download)
    with pytest.raises(IntegrityError, match="taxon mismatch"):
        downloads.fetch_species_proteome("Human papillomavirus type 16", verbose=False)

    proteomes_dir = tmp_path / "proteomes"
    assert not list(proteomes_dir.glob("*.fasta"))
    assert not list(proteomes_dir.glob("*.staging"))
    assert "Human papillomavirus type 16" not in downloads.list_proteomes()

    # 2. Pre-existing poisoned cache on disk: offline reuse raises IntegrityError;
    #    online reuse rejects the poisoned file and re-downloads the valid proteome.
    poisoned_dest = proteomes_dir / downloads._safe_filename("Human papillomavirus type 16")
    poisoned_dest.write_text(poplar_body)

    with pytest.raises(IntegrityError, match="taxon mismatch"):
        downloads.fetch_species_proteome(
            "Human papillomavirus type 16", verbose=False, fetch_missing=False
        )

    calls: list[str] = []

    def fake_good_download(url, dest, **_kw):
        calls.append(url)
        dest.write_text(hpv16_body)
        return dest

    monkeypatch.setattr(downloads, "download_to_file", fake_good_download)
    result = downloads.fetch_species_proteome("Human papillomavirus type 16", verbose=False)
    assert result == poisoned_dest
    assert len(calls) == 1
    assert "proteome:UP000009251" in calls[0]
    assert "OX=333760" in poisoned_dest.read_text()
    manifest_rec = downloads.list_proteomes()["Human papillomavirus type 16"]
    assert manifest_rec["expected_taxon_id"] == 333760


def test_fetch_dataset_viral_fasta_rejects_poisoned_download_and_cached_copy(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path)

    poplar_body = (
        ">tr|A0A2K1|POPTR Uncharacterized protein OS=Populus trichocarpa OX=3694 PE=4 SV=1\n"
        + ("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ\n" * 60)
    )
    hpv16_body = (
        ">sp|P03126|VE6_HPV16 Protein E6 OS=Human papillomavirus 16 OX=333760 GN=E6 PE=1 SV=1\n"
        + ("MHQKRTAMFQDPQERPRKLPQLCTELQTTI\n" * 60)
    )

    # 1. Poisoned download is deleted and raises IntegrityError.
    def fake_bad_download(_url, dest, **_kw):
        dest.write_text(poplar_body)
        return dest

    monkeypatch.setattr(downloads, "download_to_file", fake_bad_download)
    with pytest.raises(IntegrityError, match="taxon mismatch"):
        downloads.fetch("hpv16", offline=False)
    assert not (tmp_path / "hpv16.fasta").exists()
    assert "hpv16" not in downloads.list_datasets()

    # 2. Stale poisoned cached file (even if registered with matching sha256) is rejected and re-fetched.
    cached = tmp_path / "hpv16.fasta"
    cached.write_text(poplar_body)
    downloads.register("hpv16", cached)

    def fake_good_download(_url, dest, **_kw):
        dest.write_text(hpv16_body)
        return dest

    monkeypatch.setattr(downloads, "download_to_file", fake_good_download)
    fetched = downloads.fetch("hpv16", offline=False)
    assert fetched == cached
    assert "OX=333760" in fetched.read_text()


def test_audit_cached_proteomes_detects_mismatches_and_tmp_files(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path)
    proteomes_dir = tmp_path / "proteomes"
    proteomes_dir.mkdir(parents=True)

    # Valid cached species proteome
    (proteomes_dir / downloads._safe_filename("Ascaris suum")).write_text(
        ">tr|F1L001|ASCSU Putative protein OS=Ascaris suum OX=6253 PE=4 SV=1\nMKVFLVLLCL\n"
    )
    # Poisoned cached species proteome (A. lumbricoides cached under Ascaris suum would fail;
    # here test Canine distemper virus poisoned with Influenza A)
    (proteomes_dir / downloads._safe_filename("canine distemper virus")).write_text(
        ">sp|P03437|HEMA_I34A1 Hemagglutinin OS=Influenza A virus OX=211044 PE=1 SV=2\nMKANLLVLL\n"
    )
    # Poisoned FETCHABLE_DATASETS viral FASTA + leftover .tmp file
    (tmp_path / "hpv16.fasta").write_text(
        ">tr|A0A2K1|POPTR Uncharacterized protein OS=Populus trichocarpa OX=3694 PE=4 SV=1\nMKTA\n"
    )
    (tmp_path / "hpv16.fasta.tmp").write_text("partial")
    (proteomes_dir / "old.fasta.tmp").write_text("partial")

    report = downloads.audit_cached_proteomes()
    assert report["n_files_checked"] == 3
    assert report["n_valid_files"] == 1
    assert report["n_mismatched_files"] == 2
    assert report["n_leftover_tmp_files"] == 2
    mismatched_taxa = {tuple(m["observed_taxon_ids"]) for m in report["mismatches"]}
    assert mismatched_taxa == {(211044,), (3694,)}


def test_fetch_aav6_uses_organism_id_query_and_non_reference_upid_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path)

    # 1. AAV-6 uses organism_id:68558 directly.
    seen_urls: list[str] = []

    def fake_aav6(url, dest, **_kw):
        seen_urls.append(url)
        dest.write_text(
            ">tr|O56137|O56137_AAV6 Capsid protein VP1 OS=Adeno-associated virus 6 OX=68558 PE=1 SV=1\n"
            "MAADGYLPDWLEDNLSEGIREWWDLKPGAPKPKANQQKQDDGRGLVLPGYKYLGPFNGLDKGEPVNAADAAALEHDKAYDQQLKAGDNPYL\n"
        )
        return dest

    monkeypatch.setattr(downloads, "download_to_file", fake_aav6)
    aav6_path = downloads.fetch_species_proteome("Adeno-associated virus 6", verbose=False)
    assert aav6_path is not None and aav6_path.exists()
    assert seen_urls == [
        "https://rest.uniprot.org/uniprotkb/stream?query=organism_id:68558&format=fasta&compressed=false"
    ]

    # 2. fetch_proteome_by_upid falls back to organism_id:<taxon> when proteome:UP... returns 0 bytes.
    measles_urls: list[str] = []

    def fake_measles(url, dest, **_kw):
        measles_urls.append(url)
        if "proteome:UP000100252" in url:
            dest.write_text("")
        else:
            dest.write_text(
                ">sp|P0DXN6|NCAP_MEASF Nucleoprotein OS=Measles virus (strain Edmonston B) OX=70146 PE=1 SV=1\n"
                "MATLLRSLALFKRNKDKPPITSGSGGAIRGIKHIIIVPIPGDSSITTRSRLLDRLVRLIGNPDVSGPKLTGALIGILSLFVESPGQLIQR\n"
            )
        return dest

    monkeypatch.setattr(downloads, "download_to_file", fake_measles)
    monkeypatch.setattr(downloads, "_resolve_uniprot_proteome_taxon", lambda _upid: 70146)
    measles_path = downloads.fetch_proteome_by_upid(
        "UP000100252", label="Measles virus (strain Edmonston B)", verbose=False
    )
    assert measles_path is not None and measles_path.exists()
    assert len(measles_urls) == 2
    assert "proteome:UP000100252" in measles_urls[0]
    assert "organism_id:70146" in measles_urls[1]


def test_resolve_proteome_via_uniprot_skips_non_reference_and_unrelated_organisms(monkeypatch):
    import io
    import json
    import urllib.request

    payload = {
        "results": [
            {
                "id": "UP000111111",
                "proteomeType": "Non Reference proteome",
                "proteinCount": 40000,
                "taxonomy": {"scientificName": "Canis lupus familiaris", "taxonId": 9615},
            },
            {
                "id": "UP000222222",
                "proteomeType": "Reference proteome",
                "proteinCount": 12,
                "taxonomy": {"scientificName": "Influenza A virus", "taxonId": 11320},
            },
            {
                "id": "UP000805418",
                "proteomeType": "Reference proteome",
                "proteinCount": 43616,
                "taxonomy": {"scientificName": "Canis lupus familiaris", "taxonId": 9615},
            },
        ]
    }

    class _FakeResp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *_a, **_kw: _FakeResp(json.dumps(payload).encode()),
    )
    hit = downloads.resolve_proteome_via_uniprot("Canis lupus familiaris")
    assert hit is not None
    assert hit["proteome_id"] == "UP000805418"
    assert hit["taxon_id"] == 9615

    # Unrelated organism with only Influenza A / Canis results must return None, not Influenza A!
    assert downloads.resolve_proteome_via_uniprot("Trichosurus vulpecula") is None


def test_audit_cached_proteomes_allows_legitimate_strain_proteome_and_flags_replaced_upid(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(downloads, "_override_data_dir", tmp_path)
    proteomes_dir = tmp_path / "proteomes"
    proteomes_dir.mkdir(parents=True)

    # Legitimate strain-specific UniProt-resolved proteome (M. tuberculosis CDC 1551, OX=83331)
    cdc_path = proteomes_dir / "mycobacterium_tuberculosis_strain_cdc_1551_oshkosh.fasta"
    cdc_path.write_text(
        ">tr|P00001|MTBC Protein OS=Mycobacterium tuberculosis (strain CDC 1551 / Oshkosh) OX=83331 PE=4 SV=1\n"
        "MKTAYIAKQRQ\n"
    )
    # Poisoned strain-named CDV proteome fetched under old wrong UPID UP000117312 (Influenza A)
    cdv_path = proteomes_dir / "canine_distemper_virus_strain_onderstepoort.fasta"
    cdv_path.write_text(
        ">tr|Q00001|IAV Hemagglutinin OS=Influenza A virus OX=1331558 PE=4 SV=1\nMKTAYIAKQRQ\n"
    )
    with downloads.manifest_transaction() as manifest:
        manifest["proteomes"] = {
            "Mycobacterium tuberculosis (strain CDC 1551 / Oshkosh)": {
                "kind": "uniprot",
                "proteome_id": "UP000001020",
                "path": str(cdc_path),
            },
            "Canine distemper virus strain Onderstepoort": {
                "kind": "uniprot",
                "proteome_id": "UP000117312",
                "path": str(cdv_path),
            },
        }

    report = downloads.audit_cached_proteomes()
    assert report["n_files_checked"] == 2
    assert report["n_valid_files"] == 1
    assert report["n_mismatched_files"] == 1
    assert report["mismatches"][0]["path"] == str(cdv_path)
    assert report["mismatches"][0]["observed_taxon_ids"] == [1331558]
