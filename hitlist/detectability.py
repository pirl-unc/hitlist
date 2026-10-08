"""Search-scoped bulk-MS detectability candidates, preserving parent occurrences."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path

import pandas as pd

from .proteome import DEFAULT_FLANK, canonical_search_enzyme, digest_occurrences
from .provenance import file_digest
from .version import __version__

# Neutral monoisotopic residue masses; add H2O for an intact peptide.
_RESIDUE_MASS = dict(
    zip(
        "ACDEFGHIKLMNPQRSTVWY",
        (
            71.037114,
            103.009185,
            115.026943,
            129.042593,
            147.068414,
            57.021464,
            137.058912,
            113.084064,
            128.094963,
            113.084064,
            131.040485,
            114.042927,
            97.052764,
            128.058578,
            156.101111,
            87.032028,
            101.047679,
            99.068414,
            186.079313,
            163.063329,
        ),
    )
)
_SCOPE_COLUMNS = (
    "source",
    "cell_line_name",
    "digestion_enzyme",
    "n_fractions_in_run",
    "enrichment",
    "fractionation_ph",
    "protocol_id",
)
_ACQUISITION_COLUMNS = (
    "instrument",
    "fragmentation",
    "acquisition_mode",
    "labeling",
    "search_engine",
    "detection_basis",
    "protein_observation_basis",
)
_CANDIDATE_COLUMNS = (
    "peptide",
    "uniprot_acc",
    "gene_symbol",
    "start_position",
    "end_position",
    "n_flank",
    "c_flank",
    "n_missed_cleavages",
    "peptide_mass_da",
    "observed",
    "n_replicates_detected",
    "n_replicates_possible",
    "replicate_count_status",
    "first_seen_at_n_fractions",
    "first_seen_depth_status",
    "protein_observed",
    "protein_abundance_percentile",
    "search_space_id",
    *_SCOPE_COLUMNS,
    *_ACQUISITION_COLUMNS,
)


@dataclass(frozen=True)
class DetectabilitySearchSpace:
    """An explicit search contract; unknown limits must not be called unlimited.

    ``None`` for an upper bound declares that the search had no such bound.
    It does not mean unknown. Supply the actual searched FASTA's SHA256 and a
    provenance reference to the recorded search settings. The historical
    PXD004452 deposit alone does not supply a complete contract (#654).
    """

    fasta_sha256: str
    enzyme: str
    max_missed_cleavages: int
    min_peptide_length: int
    max_peptide_length: int | None
    max_peptide_mass_da: float | None
    fixed_residue_modifications: dict[str, float]
    provenance: str

    def __post_init__(self):
        if len(self.fasta_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.fasta_sha256
        ):
            raise ValueError("fasta_sha256 must be the lowercase SHA256 of the searched FASTA")
        canonical_search_enzyme(self.enzyme)
        for name, value, minimum in (
            ("max_missed_cleavages", self.max_missed_cleavages, 0),
            ("min_peptide_length", self.min_peptide_length, 1),
        ):
            _integer(name, value, minimum)
        if self.max_peptide_length is not None:
            _integer("max_peptide_length", self.max_peptide_length, self.min_peptide_length)
        if self.max_peptide_mass_da is not None and (
            not math.isfinite(self.max_peptide_mass_da) or self.max_peptide_mass_da <= 0
        ):
            raise ValueError("max_peptide_mass_da must be finite and positive")
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValueError("Search-setting provenance is required")
        for residue, mass in self.fixed_residue_modifications.items():
            if residue not in _RESIDUE_MASS or len(residue) != 1 or not math.isfinite(mass):
                raise ValueError(
                    "Fixed modifications must map standard residues to finite mass deltas"
                )

    @classmethod
    def read(cls, value):
        if isinstance(value, cls):
            return cls(**asdict(value))
        if value is None:
            raise ValueError(
                "A verified search_space contract is required for absence labels: supply "
                "DetectabilitySearchSpace, a mapping, or its JSON path. Historical source "
                "summaries omit the exact FASTA identity and some search limits (#654)."
            )
        if isinstance(value, (str, Path)):
            value = json.loads(Path(value).read_text())
        return cls(**value)

    @property
    def identifier(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_fasta(cls, path, **settings):
        """Fingerprint a searched reference; settings still require evidence."""
        return cls(fasta_sha256=file_digest(Path(path))["sha256"], **settings)


def _integer(name, value, minimum):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _fasta_records(path, *, max_protein_residues, max_reference_bytes):
    """Stream one protein at a time and reject ambiguous reference identities."""
    import gzip
    import re

    opener = gzip.open if str(path).endswith(".gz") else open
    seen = set()
    header = None
    fragments = []
    n_residues = 0
    n_bytes = 0

    def record():
        token = header.split()[0]
        accession = token.split("|")[1] if token.startswith(("sp|", "tr|")) else token
        if accession in seen:
            raise ValueError(f"Duplicate protein identifier in search FASTA: {accession}")
        seen.add(accession)
        if len(seen) > 500000:
            raise ValueError("Search reference exceeds 500000 protein identifiers")
        if not fragments:
            raise ValueError(f"Empty protein sequence: {accession}")
        gene = re.search(r"(?:^|\s)GN=([^\s]+)", header)
        sequence = "".join(fragments).removesuffix("*")
        if not sequence or any(not r.isalpha() or not r.isascii() for r in sequence):
            raise ValueError(f"Invalid protein sequence: {accession}")
        return accession, gene.group(1) if gene else "", sequence

    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        while line := stream.readline(max_protein_residues + 3):
            n_bytes += len(line.encode())
            if n_bytes > max_reference_bytes:
                raise ValueError("Decompressed FASTA exceeds max_reference_bytes")
            if len(line) > max_protein_residues + 2:
                raise ValueError("FASTA line exceeds max_protein_residues")
            if line.startswith(">"):
                if header is not None:
                    yield record()
                header = line[1:].strip()
                if not header:
                    raise ValueError("Empty FASTA header")
                if len(header) > 4096:
                    raise ValueError("FASTA header exceeds 4096 characters")
                fragments, n_residues = [], 0
            elif line.strip():
                if header is None:
                    raise ValueError("FASTA sequence precedes its header")
                sequence = line.strip().upper()
                n_residues += len(sequence)
                if n_residues > max_protein_residues:
                    raise ValueError("Protein exceeds max_protein_residues; no silent truncation")
                fragments.append(sequence)
        if header is not None:
            yield record()


def _select_scope(frame, filters):
    missing = set(_SCOPE_COLUMNS) - set(frame)
    if missing:
        raise ValueError(f"Observation scope metadata missing: {sorted(missing)}")
    result = frame
    for column, value in filters.items():
        if value is not None:
            if column == "cell_line_name":
                mask = result[column].astype("string").str.casefold().eq(value.casefold())
            elif column == "digestion_enzyme":
                names = {
                    name: canonical_search_enzyme(name) for name in result[column].dropna().unique()
                }
                mask = result[column].map(names).eq(canonical_search_enzyme(value))
            else:
                mask = result[column].eq(value)
            result = result.loc[mask]
    if result.empty:
        raise ValueError("No observed source scope matches the requested controls")
    if result[list(_SCOPE_COLUMNS)].drop_duplicates().shape[0] != 1:
        raise ValueError("Multiple acquisition scopes match; select an explicit protocol_id")
    return result


def _scope_value(frame, column):
    values = frame[column].drop_duplicates() if column in frame else pd.Series(dtype=object)
    if len(values) != 1 or pd.isna(values.iloc[0]) or str(values.iloc[0]).strip() == "":
        raise ValueError(f"Scope requires one explicit {column}")
    value = values.iloc[0]
    return value.item() if hasattr(value, "item") else value


def _sequence_evidence(frame):
    """Sequence labels never depend on an arbitrary razor-parent assignment."""
    if "peptide" not in frame or frame.peptide.isna().any():
        raise ValueError("Every observed row requires a peptide sequence")
    evidence = {}
    for peptide, rows in frame.groupby("peptide", sort=False, observed=True):
        if not isinstance(peptide, str) or not peptide or set(peptide) - _RESIDUE_MASS.keys():
            raise ValueError("Observed peptides must be unmodified uppercase standard sequences")
        counts = rows.get("n_replicates_detected", pd.Series(dtype="Int64")).dropna().unique()
        if len(counts) > 1:
            raise ValueError(f"Conflicting aggregate replicate counts for {peptide}")
        count = counts[0] if len(counts) else pd.NA
        if not pd.isna(count):
            if isinstance(count, bool) or not float(count).is_integer():
                raise ValueError("Detected replicate counts must be integers")
            count = int(count)
        if "replicate_id" in rows and rows.replicate_id.notna().all():
            count = rows.replicate_id.nunique()
        if not pd.isna(count) and count <= 0:
            raise ValueError("An observed peptide cannot have zero detected replicates")
        evidence[str(peptide)] = count
    return evidence


def _first_seen_depth(all_rows, selected, search_space_id):
    """Only a provenance-backed comparison group can relate different depths."""
    if "comparison_group" not in selected:
        return {}, "unavailable_no_comparable_protocol_group"
    groups = selected.comparison_group.dropna().unique()
    if len(groups) == 0:
        return {}, "unavailable_no_comparable_protocol_group"
    if len(groups) != 1 or not groups[0] or selected.comparison_group.isna().any():
        raise ValueError("Selected scope requires one complete comparison_group or none")
    related = all_rows.loc[all_rows.comparison_group.eq(groups[0])]
    _scope_value(related, "comparison_provenance")
    invariant = [c for c in _SCOPE_COLUMNS if c not in ("n_fractions_in_run", "protocol_id")]
    invariant += [
        *list(_ACQUISITION_COLUMNS),
        "search_space_id",
        "n_replicates_possible",
        "comparison_controls",
    ]
    for column in invariant:
        _scope_value(related, column)
    if _scope_value(related, "search_space_id") != search_space_id:
        raise ValueError("Comparison group's search_space_id differs from the selected contract")
    controls = json.loads(_scope_value(related, "comparison_controls"))
    if not isinstance(controls, dict) or any(
        not controls.get(k)
        for k in (
            "lc_gradient_minutes",
            "peptide_load_ug",
            "sample_preparation",
            "acquisition_method",
        )
    ):
        raise ValueError(
            "Comparison controls must document LC gradient, load, preparation and acquisition method"
        )
    for column in ("lc_gradient_minutes", "peptide_load_ug"):
        value = controls[column]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"Comparison {column} must be finite and positive")
    if "experiment_ids" not in related or related.experiment_ids.isna().any():
        raise ValueError("Comparison requires experiment_ids to detect reused experiments")
    protocols = related[["protocol_id", "n_fractions_in_run", "experiment_ids"]].drop_duplicates()
    if protocols.protocol_id.duplicated().any() or protocols.n_fractions_in_run.duplicated().any():
        raise ValueError("Comparison requires exactly one protocol per depth")
    seen_experiments = set()
    for record in protocols.itertuples(index=False):
        experiments = json.loads(record.experiment_ids)
        if (
            not isinstance(experiments, list)
            or not experiments
            or any(not isinstance(x, str) or not x for x in experiments)
        ):
            raise ValueError(
                "experiment_ids must be a nonempty JSON list of source-qualified identifiers"
            )
        if len(set(experiments)) != len(experiments) or seen_experiments.intersection(experiments):
            raise ValueError("Comparison group reuses experiments")
        if len(experiments) != _scope_value(related, "n_replicates_possible"):
            raise ValueError("Comparison experiment count differs from n_replicates_possible")
        seen_experiments.update(experiments)
        counts = _sequence_evidence(related.loc[related.protocol_id.eq(record.protocol_id)])
        if any(not pd.isna(value) and value > len(experiments) for value in counts.values()):
            raise ValueError("Comparison replicate count exceeds possible experiments")
    for depth in related.n_fractions_in_run.unique():
        _integer("comparison n_fractions_in_run", depth, 1)
    return related.groupby(
        "peptide", observed=True
    ).n_fractions_in_run.min().to_dict(), "comparable_protocol_group"


def _frame_digest(frame):
    """Content hash without serializing the entire observation table at once."""
    digest = hashlib.sha256()
    digest.update(json.dumps(list(frame.columns)).encode())
    for start in range(0, len(frame), 10000):
        digest.update(
            frame.iloc[start : start + 10000]
            .to_json(orient="records", double_precision=15)
            .encode()
        )
    return {
        "sha256": digest.hexdigest(),
        "n_rows": len(frame),
        "encoding": "pandas-records-json-10000",
    }


def _validate_reference(
    reference, selected, parent_rows, max_protein_residues, max_reference_bytes
):
    """Validate reported parent assignments before emitting any negative labels."""
    if "uniprot_acc" not in selected or selected.uniprot_acc.isna().any():
        raise ValueError(
            "Observed peptides require reported parent identifiers for reference validation"
        )
    assigned = selected.groupby("uniprot_acc", observed=True).peptide.agg(set).to_dict()
    missing = set(parent_rows) | set(assigned)
    for accession, _, sequence in _fasta_records(
        reference,
        max_protein_residues=max_protein_residues,
        max_reference_bytes=max_reference_bytes,
    ):
        missing.discard(accession)
        for peptide in assigned.get(accession, ()):
            if peptide not in sequence:
                raise ValueError(
                    f"Observed peptide {peptide} does not occur in reported parent {accession}"
                )
    if missing:
        raise ValueError(
            f"Search reference is missing {len(missing)} observed parent proteins, e.g. {sorted(missing)[:5]}"
        )


def iter_detectability_training_set(
    *,
    search_fasta=None,
    search_space=None,
    cell_line="HeLa",
    digestion_enzyme="Trypsin/P",
    n_fractions_in_run=46,
    max_missed=2,
    length=(7, 30),
    require_protein_observed=True,
    enrichment="none",
    fractionation_ph=10.0,
    protocol_id=None,
    peptide_observations=None,
    protein_observations=None,
    flank=DEFAULT_FLANK,
    batch_size=25000,
    max_candidates=2000000,
    max_protein_residues=1000000,
    max_reference_bytes=256 * 1024**2,
    max_observation_rows=1000000,
):
    """Yield bounded DataFrames of candidate occurrences for one search scope.

    Absence means not observed within that scope. It does not establish intrinsic
    undetectability, equal digestion efficiency, or vaccine safety. Input source
    tables may be supplied explicitly; the packaged adapter supplies their
    documented aggregate scope, including unresolved replicate/protocol details.
    """
    contract = DetectabilitySearchSpace.read(search_space)
    if search_fasta is None:
        raise ValueError(
            "search_fasta must name the actual searched reference; a current proteome is not a substitute"
        )
    for name, value, minimum in (
        ("max_missed", max_missed, 0),
        ("flank", flank, 0),
        ("batch_size", batch_size, 1),
        ("max_candidates", max_candidates, 1),
        ("max_protein_residues", max_protein_residues, 1),
        ("max_reference_bytes", max_reference_bytes, 1),
        ("max_observation_rows", max_observation_rows, 1),
    ):
        _integer(name, value, minimum)
    _integer("n_fractions_in_run", n_fractions_in_run, 1)
    if not isinstance(require_protein_observed, bool):
        raise ValueError("require_protein_observed must be a boolean")
    if len(length) != 2:
        raise ValueError("length must contain inclusive lower and upper bounds")
    lower, upper = length
    _integer("length lower bound", lower, 1)
    _integer("length upper bound", upper, lower)
    if max_missed > contract.max_missed_cleavages:
        raise ValueError("Requested missed-cleavage allowance exceeds the recorded search space")
    if lower < contract.min_peptide_length or (
        contract.max_peptide_length is not None and upper > contract.max_peptide_length
    ):
        raise ValueError("Requested length window exceeds the recorded search space")
    reference = Path(search_fasta).resolve()
    if reference.stat().st_size > max_reference_bytes:
        raise ValueError("Search FASTA exceeds max_reference_bytes")
    reference_digest = file_digest(reference)
    if reference_digest["sha256"] != contract.fasta_sha256:
        raise ValueError("Search FASTA does not match the declared search-space SHA256")
    if (peptide_observations is None) != (protein_observations is None):
        raise ValueError("Supply both peptide_observations and protein_observations, or neither")
    source_inputs = {}
    if peptide_observations is None:
        peptide_observations, protein_observations, source_inputs = _packaged_observations(
            cell_line,
            digestion_enzyme,
            n_fractions_in_run,
            enrichment,
            fractionation_ph,
            contract,
            max_observation_rows,
        )
    for frame in (peptide_observations, protein_observations):
        if len(frame) > max_observation_rows:
            raise ValueError("Observation table exceeds max_observation_rows")
    filters = {
        "cell_line_name": cell_line,
        "digestion_enzyme": digestion_enzyme,
        "n_fractions_in_run": n_fractions_in_run,
        "enrichment": enrichment,
        "fractionation_ph": fractionation_ph,
        "protocol_id": protocol_id,
    }
    selected = _select_scope(peptide_observations, filters)
    scope = {column: _scope_value(selected, column) for column in _SCOPE_COLUMNS}
    acquisition = {column: _scope_value(selected, column) for column in _ACQUISITION_COLUMNS}
    if _scope_value(selected, "search_space_id") != contract.identifier:
        raise ValueError("Observed scope's search_space_id differs from the declared search space")
    if canonical_search_enzyme(_scope_value(selected, "search_enzyme")) != canonical_search_enzyme(
        contract.enzyme
    ):
        raise ValueError("Observed scope's enzyme differs from the declared search space")
    parents = _select_scope(protein_observations, scope)
    if "uniprot_acc" not in parents or parents.uniprot_acc.isna().any():
        raise ValueError("Parent-protein observations require reference protein identifiers")
    if parents.uniprot_acc.duplicated().any():
        raise ValueError("Ambiguous parent-protein rows in one scope")
    if _scope_value(parents, "search_space_id") != contract.identifier:
        raise ValueError("Parent scope's search_space_id differs from the declared search space")
    if (
        "n_peptides" not in parents
        or parents.n_peptides.isna().any()
        or (parents.n_peptides <= 0).any()
        or (parents.n_peptides % 1 != 0).any()
    ):
        raise ValueError(
            "Observed parents require positive n_peptides integers in the selected scope"
        )
    for column in _ACQUISITION_COLUMNS:
        if column in parents and _scope_value(parents, column) != acquisition[column]:
            raise ValueError(f"Parent observation scope has conflicting {column}")
    if "abundance_percentile" in parents:
        abundance = parents.abundance_percentile.dropna()
        if not abundance.between(0, 1).all():
            raise ValueError("Parent abundance_percentile must be within 0-1")
    parent_rows = parents.set_index("uniprot_acc").to_dict("index")
    observed = _sequence_evidence(selected)
    _validate_reference(reference, selected, parent_rows, max_protein_residues, max_reference_bytes)
    first_depth, first_depth_status = _first_seen_depth(
        peptide_observations, selected, contract.identifier
    )
    possible = _scope_value(selected, "n_replicates_possible")
    _integer("n_replicates_possible", possible, 1)
    if any(not pd.isna(value) and value > possible for value in observed.values()):
        raise ValueError("Detected replicate count exceeds possible replicates")
    from .uniprot import uniprot_reference_for_digest

    uniprot_reference = uniprot_reference_for_digest(**reference_digest)
    metadata = {
        "schema_version": 1,
        "hitlist_version": __version__,
        "pandas_version": pd.__version__,
        "label": "observed_in_selected_search_scope",
        "search_space": asdict(contract),
        "search_reference": {"path": str(reference), **reference_digest},
        "scope": scope,
        "acquisition": acquisition,
        "source_inputs": source_inputs,
        "peptide_observations": _frame_digest(peptide_observations),
        "protein_observations": _frame_digest(protein_observations),
        "first_seen_depth_status": first_depth_status,
        "parameters": {
            "max_missed": max_missed,
            "length": list(length),
            "flank": flank,
            "require_protein_observed": require_protein_observed,
            "max_candidates": max_candidates,
            "batch_size": batch_size,
            "max_protein_residues": max_protein_residues,
            "max_reference_bytes": max_reference_bytes,
            "max_observation_rows": max_observation_rows,
        },
    }
    metadata["search_space_id"] = contract.identifier
    if uniprot_reference is not None:
        metadata["search_reference"]["uniprot"] = uniprot_reference
    if first_depth_status == "comparable_protocol_group":
        group = _scope_value(selected, "comparison_group")
        related = peptide_observations.loc[peptide_observations.comparison_group.eq(group)]
        metadata["depth_comparison"] = {
            "group": group,
            "provenance": _scope_value(related, "comparison_provenance"),
            "controls": json.loads(_scope_value(related, "comparison_controls")),
            "protocols": related[["protocol_id", "n_fractions_in_run", "experiment_ids"]]
            .drop_duplicates()
            .to_dict("records"),
        }
    n_candidates = 0
    batch = []
    for accession, fasta_gene, sequence in _fasta_records(
        reference,
        max_protein_residues=max_protein_residues,
        max_reference_bytes=max_reference_bytes,
    ):
        parent = parent_rows.get(accession)
        if require_protein_observed and parent is None:
            continue
        gene = (parent or {}).get("gene_symbol", fasta_gene)
        if pd.isna(gene) or not gene:
            gene = fasta_gene
        for candidate in digest_occurrences(sequence, contract.enzyme, lower, upper, max_missed):
            peptide = candidate.peptide
            if set(peptide) - _RESIDUE_MASS.keys():
                continue  # ambiguous/nonstandard residues cannot establish searchable absence
            mass = 18.010565 + sum(
                _RESIDUE_MASS[r] + contract.fixed_residue_modifications.get(r, 0) for r in peptide
            )
            if contract.max_peptide_mass_da is not None and mass > contract.max_peptide_mass_da:
                continue
            n_candidates += 1
            if n_candidates > max_candidates:
                raise ValueError(
                    "Candidate limit exceeded; no complete dataset was produced. Increase max_candidates explicitly or narrow the input reference/scope"
                )
            present = peptide in observed
            start, end = candidate.start_position, candidate.end_position
            batch.append(
                {
                    "peptide": peptide,
                    "uniprot_acc": accession,
                    "gene_symbol": gene,
                    "start_position": start,
                    "end_position": end,
                    "n_flank": sequence[max(0, start - 1 - flank) : start - 1],
                    "c_flank": sequence[end : end + flank],
                    "n_missed_cleavages": candidate.n_missed_cleavages,
                    "peptide_mass_da": mass,
                    "observed": present,
                    "n_replicates_detected": observed[peptide] if present else 0,
                    "n_replicates_possible": possible,
                    "replicate_count_status": "unresolved_aggregate"
                    if present and pd.isna(observed[peptide])
                    else "available",
                    "first_seen_at_n_fractions": first_depth.get(peptide, pd.NA),
                    "first_seen_depth_status": first_depth_status,
                    "protein_observed": parent is not None,
                    "protein_abundance_percentile": (parent or {}).get(
                        "abundance_percentile", pd.NA
                    ),
                    "search_space_id": metadata["search_space_id"],
                    **scope,
                    **acquisition,
                }
            )
            if len(batch) >= batch_size:
                yield _candidate_frame(batch, metadata)
                batch = []
    if file_digest(reference) != reference_digest:
        raise ValueError("Search FASTA changed during candidate generation")
    if batch or n_candidates == 0:
        yield _candidate_frame(batch, metadata)


def _candidate_frame(rows, metadata):
    frame = pd.DataFrame(rows, columns=_CANDIDATE_COLUMNS)
    integers = {
        "start_position",
        "end_position",
        "n_missed_cleavages",
        "n_replicates_detected",
        "n_replicates_possible",
        "first_seen_at_n_fractions",
        "n_fractions_in_run",
    }
    floats = {"peptide_mass_da", "protein_abundance_percentile", "fractionation_ph"}
    for column in frame:
        dtype = (
            "Int64"
            if column in integers
            else "Float64"
            if column in floats
            else "bool"
            if column in ("observed", "protein_observed")
            else "string"
        )
        frame[column] = frame[column].astype(dtype)
    frame.attrs["detectability"] = copy.deepcopy(metadata)
    return frame


def build_detectability_training_set(**kwargs):
    """Materialize the bounded iterator; prefer iteration for large references."""
    batches = list(iter_detectability_training_set(**kwargs))
    result = pd.concat(batches, ignore_index=True)
    result.attrs = batches[0].attrs.copy()
    return result


def export_detectability_training_set(output_dir, *, max_output_bytes=1024**3, **kwargs):
    """Atomically export candidates.parquet + manifest.json with a storage cap.

    The destination must not exist. Failed/over-limit streams leave no partial
    dataset. The cap covers both exported files (including Parquet metadata).
    A direct iterator consumer must likewise discard all earlier batches if
    iteration fails; only exhausting the iterator establishes completion.
    """
    import io
    import tempfile

    import pyarrow as pa
    import pyarrow.parquet as pq

    _integer("max_output_bytes", max_output_bytes, 1)
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    class LimitedWriter(io.BufferedWriter):
        def write(self, data):
            if self.tell() + len(data) > max_output_bytes:
                raise OSError("Detectability output exceeds max_output_bytes")
            return super().write(data)

    with tempfile.TemporaryDirectory(prefix=".detectability-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        parquet = staging / "candidates.parquet"
        n_candidates = n_observed_candidates = 0
        manifest = None
        with LimitedWriter(io.FileIO(parquet, "wb")) as sink:
            writer = None
            try:
                for frame in iter_detectability_training_set(**kwargs):
                    if manifest is None:
                        manifest = copy.deepcopy(frame.attrs["detectability"])
                    table = pa.Table.from_pandas(
                        frame, preserve_index=False
                    ).replace_schema_metadata(None)
                    if writer is None:
                        writer = pq.ParquetWriter(sink, table.schema, compression="zstd")
                    writer.write_table(table)
                    n_candidates += len(frame)
                    n_observed_candidates += int(frame.observed.sum())
            finally:
                if writer is not None:
                    writer.close()
        manifest.update(
            n_candidates=n_candidates,
            n_observed_candidates=n_observed_candidates,
            artifact={"name": parquet.name, **file_digest(parquet)},
            max_output_bytes=max_output_bytes,
        )
        payload = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
        if parquet.stat().st_size + len(payload) > max_output_bytes:
            raise OSError("Detectability manifest and output exceed max_output_bytes")
        (staging / "manifest.json").write_bytes(payload)
        if destination.exists():
            raise FileExistsError(destination)
        staging.rename(destination)
    return destination


def _packaged_observations(cell_line, enzyme, depth, enrichment, ph, contract, max_rows):
    from importlib.resources import files

    from .bulk_proteomics import _bulk_data_path
    from .curation_yaml import load_curation_yaml

    policy_path = files("hitlist.data.bulk_proteomics") / "detectability.yaml"
    policy = load_curation_yaml(policy_path)
    matches = [
        arm
        for arm in policy["scopes"]
        if arm["cell_line_name"].casefold() == cell_line.casefold()
        and canonical_search_enzyme(arm["digestion_enzyme"]) == canonical_search_enzyme(enzyme)
        and arm["n_fractions_in_run"] == depth
        and arm["enrichment"] == enrichment
        and arm["fractionation_ph"] == ph
    ]
    if len(matches) != 1:
        raise ValueError("No curated packaged detectability scope matches these controls")
    arm = matches[0]
    known = policy["search_constraints"][arm["search_enzyme"]]
    if (
        contract.min_peptide_length != known["min_peptide_length"]
        or contract.max_missed_cleavages != known["max_missed_cleavages"]
    ):
        raise ValueError("Search contract contradicts the deposited search constraints")
    if contract.fixed_residue_modifications != {"C": 57.021464}:
        raise ValueError("Packaged search requires fixed Carbamidomethyl(C), +57.021464 Da")
    tables = []
    inputs = {"curation": file_digest(policy_path), "scope": arm, "source": policy["provenance"]}
    for suffix in ("peptides", "protein_abundance"):
        path = Path(_bulk_data_path(f"bekker_jensen_2017_{suffix}.csv.gz"))
        inputs[suffix] = {"path": str(path), **file_digest(path)}
        chunks = []
        n_rows = 0
        for chunk in pd.read_csv(path, chunksize=25000):
            mask = (
                chunk.cell_line.str.casefold().eq(cell_line.casefold())
                & chunk.digestion_enzyme.eq(arm["reported_digestion_enzyme"])
                & chunk.n_fractions_in_run.eq(depth)
                & chunk.enrichment.eq(enrichment)
                & chunk.fractionation_ph.eq(ph)
            )
            chunk = chunk.loc[mask].copy().rename(columns={"cell_line": "cell_line_name"})
            n_rows += len(chunk)
            if n_rows > max_rows:
                raise ValueError("Packaged scope exceeds max_observation_rows")
            chunks.append(chunk)
        frame = pd.concat(chunks, ignore_index=True)
        if file_digest(path) != {k: inputs[suffix][k] for k in ("sha256", "size_bytes")}:
            raise ValueError("Packaged source changed during reading")
        frame["reported_reference"] = frame["reference"]
        frame["reference"] = policy["reference"]
        for key in (*_SCOPE_COLUMNS, "search_enzyme", "n_replicates_possible"):
            frame[key] = arm.get(key, policy.get(key))
        for key in _ACQUISITION_COLUMNS:
            frame[key] = arm.get(key, policy[key])
        frame["search_space_id"] = contract.identifier
        if suffix == "peptides" and arm.get("replicate_count_status") == "unresolved_aggregate":
            frame["reported_n_replicates_detected"] = frame["n_replicates_detected"]
            frame["n_replicates_detected"] = pd.NA
        tables.append(frame)
    return *tables, inputs
