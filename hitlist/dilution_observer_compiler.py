"""Streaming source preservation and explicit cross-namespace relationships."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path

from .dilution_observer_schema import MEASURE_FIELDS, SCHEMAS
from .mapp_evidence import _Sink

MISSING = frozenset({"", "NA", "NaN", "nan", "null"})


def identity(kind, *values):
    payload = json.dumps(values, ensure_ascii=True, separators=(",", ":"))
    return "pxd024809:" + kind + ":" + hashlib.sha256(payload.encode()).hexdigest()


def number(literal):
    """Preserve raw states; a zero is never interpreted as an absence or bound."""
    if literal in MISSING:
        return None, "missing"
    try:
        value = Decimal(literal)
    except InvalidOperation:
        return None, "invalid"
    if not value.is_finite():
        return None, "nonfinite"
    parsed = float(value)
    if not math.isfinite(parsed) or (parsed == 0 and value != 0):
        return None, "outside_float_range"
    return parsed, "positive" if value > 0 else "negative" if value < 0 else "zero"


def comparison(left, right, contract):
    if left in MISSING or right in MISSING:
        return (
            "both_missing"
            if left in MISSING and right in MISSING
            else "aligned_missing"
            if left in MISSING
            else "scored_missing"
        ), None
    try:
        a, b = Decimal(left), Decimal(right)
    except InvalidOperation:
        return "invalid", None
    if not a.is_finite() or not b.is_finite():
        return "nonfinite", None
    # Enough significant places to preserve exact subtraction across exponents.
    with localcontext() as context:
        context.prec = (
            max(a.adjusted(), b.adjusted()) - min(a.as_tuple().exponent, b.as_tuple().exponent) + 4
        )
        delta = a - b
        status = (
            "same_literal"
            if left == right
            else "decimal_equal"
            if a == b
            else "within_numeric_tolerance"
            if abs(delta)
            <= Decimal(contract["comparison_atol"])
            + Decimal(contract["comparison_rtol"]) * max(abs(a), abs(b))
            else "different"
        )
        return status, str(delta)


class Compiler:
    def __init__(self, source, destination, contract):
        self.source, self.destination, self.contract = Path(source), Path(destination), contract
        self.destination.mkdir(parents=True, exist_ok=False)
        self.sinks = {
            name: _Sink(self.destination, name, schema=schema) for name, schema in SCHEMAS.items()
        }
        self.assets = {a["filename"]: a for a in contract["assets"]}
        self.library = {}
        self.by_form = defaultdict(list)
        self.features = defaultdict(list)
        self.acquisitions = {}
        self.standards = []
        self.levels = {}

    def add(self, table, row):
        self.sinks[table].add(
            {
                **row,
                **{
                    k: self.contract[k]
                    for k in ("source_dataset", "source_dataset_precision", "source_panel")
                },
            }
        )

    def record(self, filename, locator, headers, values, *, index=None):
        key = identity("record", filename, locator)
        self.add(
            "source_records",
            {
                "source_record_id": key,
                "source_asset_id": identity("asset", filename),
                "source_file": filename,
                "source_locator": locator,
                "source_row_index": index,
                "source_row_json": json.dumps(
                    {"headers": headers, "values": values}, ensure_ascii=True
                ),
            },
        )
        return key

    def rows(self, filename):
        role = self.assets[filename]["role"]
        with (self.source / filename).open(newline="") as stream:
            reader = csv.reader(stream, delimiter="\t")
            headers = next(reader)
            if headers != self.contract["headers"][role] or len(set(headers)) != len(headers):
                raise ValueError(f"Dilution source headers differ: {filename}")
            for line, values in enumerate(reader, 2):
                if len(values) != len(headers):
                    raise ValueError(f"Malformed dilution source row: {filename}:{line}")
                record = self.record(
                    filename, f"{filename}:line:{line}", headers, values, index=line - 1
                )
                yield record, dict(zip(headers, values))

    def compile_library(self):
        fields = (
            "TransitionGroupId",
            "ModifiedPeptideSequence",
            "PeptideSequence",
            "PrecursorCharge",
            "Decoy",
            "PeptideGroupLabel",
        )
        for record, row in self.rows(self.contract["library"]):
            key = tuple(row[k] for k in fields)
            if key not in self.library:
                self.library[key] = {
                    "library_precursor_id": identity("library", *key),
                    "source_record_id": record,
                    "native_library_group_id": key[0],
                    "modified_sequence_raw": key[1],
                    "peptide_sequence": key[2],
                    "charge_raw": key[3],
                    "decoy_raw": key[4],
                    "group_label_raw": key[5],
                    "n_transitions": 0,
                }
            precursor = self.library[key]
            precursor["n_transitions"] += 1
            self.add(
                "library_transitions",
                {
                    "transition_id": identity("transition", record),
                    "source_record_id": record,
                    "library_precursor_id": precursor["library_precursor_id"],
                    "native_transition_id": row["TransitionId"],
                    "product_mz_raw": row["ProductMz"],
                    "product_mz": number(row["ProductMz"])[0],
                    "product_charge_raw": row["ProductCharge"],
                    "library_intensity_raw": row["LibraryIntensity"],
                    "library_intensity": number(row["LibraryIntensity"])[0],
                },
            )
        for key, row in sorted(self.library.items()):
            self.add("library_precursors", row)
            self.by_form[(key[1], key[3], key[4])].append(row)

    def membership(self, form, charge, decoy, *, label=None, sequence=None):
        matches = self.by_form.get((form, charge, decoy), [])
        result = {
            "n_library_candidates": len(matches),
            "library_precursor_id": None,
            "library_link_status": "absent_form" if not matches else "ambiguous_form",
        }
        if len(matches) == 1:
            lib = matches[0]
            result.update(
                library_precursor_id=lib["library_precursor_id"], library_link_status="matched"
            )
            if sequence is not None and sequence != lib["peptide_sequence"]:
                result["library_link_status"] = "sequence_disagreement"
            if label is not None:
                other = lib["group_label_raw"]
                result["library_label_status"] = (
                    "same_literal"
                    if label == other
                    else "both_missing_different_literal"
                    if label in MISSING and other in MISSING
                    else "different"
                )
        return result

    def measurements(self, row, *, scored):
        result = {"measurement_unit": "source_signal_arbitrary_units"}
        for field, (raw, parsed) in MEASURE_FIELDS.items():
            source_field = "Intensity" if scored and field == "intensity" else field
            value, status = number(row[source_field])
            result[raw], result[parsed] = row[source_field], value
            if field == "intensity":
                result["value_status"] = status
        return result

    def compile_scores(self):
        from .dilution_observer_metadata import add_acquisition

        assets = sorted(
            a["filename"] for a in self.assets.values() if a["role"] == "per_run_scored_candidates"
        )
        if len(assets) != self.contract["n_scored_assets"]:
            raise ValueError("Scored acquisition inventory differs")
        for filename in assets:
            for record, row in self.rows(filename):
                if (
                    not row["filename"].endswith(".mzML")
                    or row["filename"].replace("#", "_") + ".tsv" != filename
                ):
                    raise ValueError("Scored acquisition filename differs")
                run = row["filename"][:-5]
                acquisition = add_acquisition(self, run, row["run_id"], record)
                typed = {
                    "identification_id": identity("feature", record),
                    "source_record_id": record,
                    "acquisition_id": acquisition,
                    "source_run_id": row["run_id"],
                    "native_feature_id": row["id"],
                    "native_precursor_id": row["transition_group_id"],
                    "peptide_sequence": row["Sequence"],
                    "modified_sequence_raw": row["FullPeptideName"],
                    "charge_raw": row["Charge"],
                    "decoy_raw": row["decoy"],
                    "observation_kind": "exported_scored_feature_candidate",
                    "dilution_d_score_raw": row["d_score"],
                    "dilution_d_score": number(row["d_score"])[0],
                    **self.measurements(row, scored=True),
                    **self.membership(
                        row["FullPeptideName"],
                        row["Charge"],
                        row["decoy"],
                        sequence=row["Sequence"],
                    ),
                }
                self.add("scored_features", typed)
                # Compact index only: large transition arrays remain in the streamed source record.
                self.features[(run, row["id"])].append(typed)

    def compile_aligned(self):
        for record, row in self.rows(self.contract["aligned"]):
            run = row["run"]
            if run not in self.acquisitions:
                raise ValueError("Aligned run lacks an admitted scored-acquisition record")
            measurement = identity("aligned", record)
            members = self.by_form.get((row["sequence"], row["charge"], "0"), [])
            sequence = members[0]["peptide_sequence"] if len(members) == 1 else None
            typed = {
                "measurement_id": measurement,
                "source_record_id": record,
                "acquisition_id": self.acquisitions[run]["acquisition_id"],
                "native_feature_id": row["feature_id"],
                "native_precursor_id": row["precursor"],
                "native_peptide_id": row["peptide_id"],
                "peptide_sequence": sequence,
                "modified_sequence_raw": row["sequence"],
                "charge_raw": row["charge"],
                "decoy_raw": "0",
                "group_label_raw": row["group_label"],
                "observation_kind": "aligned_precursor_run_slot",
                "dilution_alignment_rank_raw": row["alignment_rank"],
                "dilution_alignment_rank": number(row["alignment_rank"])[0],
                **self.measurements(row, scored=False),
                **self.membership(row["sequence"], row["charge"], "0", label=row["group_label"]),
            }
            self.add("aligned_quantities", typed)
            candidates = (
                []
                if row["feature_id"] in MISSING
                else self.features.get((run, row["feature_id"]), [])
            )
            for index, candidate in enumerate(candidates or [None]):
                link = {
                    "claim_id": identity("feature_link", record, index),
                    "source_record_id": record,
                    "measurement_id": measurement,
                    "n_candidates": len(candidates),
                }
                if candidate is None:
                    link["feature_link_status"] = (
                        "missing_feature_id"
                        if row["feature_id"] in MISSING
                        else "unmatched_feature_id"
                    )
                else:
                    checks = {
                        "form_agrees": row["sequence"] == candidate["modified_sequence_raw"],
                        "charge_agrees": row["charge"] == candidate["charge_raw"],
                        "precursor_agrees": row["precursor"] == candidate["native_precursor_id"],
                        "target_agrees": candidate["decoy_raw"] == "0",
                    }
                    link.update(identification_id=candidate["identification_id"], **checks)
                    link["feature_link_status"] = (
                        "ambiguous_feature_id"
                        if len(candidates) > 1
                        else "matched_identity"
                        if all(checks.values())
                        else "id_crosscheck_disagreement"
                    )
                    if len(candidates) == 1:
                        for field, (raw, _) in MEASURE_FIELDS.items():
                            status, delta = comparison(row[field], candidate[raw], self.contract)
                            self.add(
                                "comparisons",
                                {
                                    "claim_id": identity("comparison", record, field),
                                    "source_record_id": record,
                                    "measurement_id": measurement,
                                    "identification_id": candidate["identification_id"],
                                    "compared_source_field": field,
                                    "aligned_literal": row[field],
                                    "scored_literal": candidate[raw],
                                    "comparison_status": status,
                                    "exact_decimal_delta": delta,
                                },
                            )
                self.add("feature_links", link)

    def compile_standard_memberships(self):
        for standard in self.standards:
            members = [
                r
                for r in self.library.values()
                if r["modified_sequence_raw"] == standard["expected_modified_sequence"]
            ]
            for index, row in enumerate(
                sorted(members, key=lambda r: r["library_precursor_id"]) or [None]
            ):
                self.add(
                    "standard_memberships",
                    {
                        "claim_id": identity("standard_member", standard["standard_id"], index),
                        "source_record_id": standard["source_record_id"],
                        "standard_id": standard["standard_id"],
                        "library_precursor_id": row["library_precursor_id"] if row else None,
                        "decoy_raw": row["decoy_raw"] if row else None,
                        "membership_status": ("target" if row["decoy_raw"] == "0" else "decoy")
                        if row
                        else "absent_from_library",
                    },
                )


def compile_dilution_source(source_directory, destination, contract):
    from .dilution_observer_metadata import compile_metadata, finish_contexts

    compiler = Compiler(source_directory, destination, contract)
    try:
        for asset in contract["assets"]:
            compiler.add(
                "assets",
                {
                    "source_asset_id": identity("asset", asset["filename"]),
                    "source_file": asset["filename"],
                    "source_asset_url": asset["source_url"],
                    "source_asset_role": asset["role"],
                    "asset_sha256": asset["sha256"],
                    "size_bytes": asset["size_bytes"],
                    "license_status": asset["license_status"],
                },
            )
        compile_metadata(compiler)
        compiler.compile_library()
        compiler.compile_scores()
        compiler.compile_aligned()
        compiler.compile_standard_memberships()
        finish_contexts(compiler)
    finally:
        for sink in compiler.sinks.values():
            sink.close()
    return {name: sink.n_rows for name, sink in compiler.sinks.items()}
