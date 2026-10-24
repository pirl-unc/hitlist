"""Versioned, source-preserving training decisions over one EvidenceStore."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import sqlite3
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .curation_yaml import load_curation_yaml
from .evidence_bindings import LINKS
from .evidence_catalog import SCHEMA_VERSION
from .evidence_store import EvidenceFieldError, EvidenceQueryError
from .provenance import file_digest

RECIPE_SCHEMA_VERSION = 1
OUTPUT_SCHEMA_VERSION = 1
DECISIONS = ("accepted", "excluded", "ambiguous")
TARGET_TYPES = {"boolean": pa.bool_(), "float64": pa.float64(), "string": pa.string()}
RESERVED = {
    "candidate_id",
    "evidence_type",
    "source_population",
    "target_value",
    "target_unit",
    "decision",
    "reason_codes",
    "split_group_id",
    "split_status",
}
CONTRIBUTOR_SCHEMA = pa.schema(
    [
        ("candidate_id", pa.string()),
        ("source_record_id", pa.string()),
        ("source_assay_id", pa.string()),
        ("source_dataset", pa.string()),
        ("source_reference_id", pa.string()),
        ("link_status", pa.string()),
        ("source_row_index", pa.int64()),
        ("relationships_raw", pa.string()),
    ]
)
LINEAGE_SCHEMA = pa.schema(
    [
        ("candidate_id", pa.string()),
        ("split_group_id", pa.string()),
        ("split_status", pa.string()),
        ("group_fields_json", pa.string()),
    ]
)


class RecipeError(EvidenceQueryError):
    """A recipe is invalid or cannot preserve its declared source semantics."""


class _Writer:
    def __init__(self, path, schema):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.schema = schema
        self.writer = pq.ParquetWriter(path, schema, compression="zstd")
        self.rows = []
        self.n_rows = 0

    def add(self, row):
        self.rows.append(row)
        if len(self.rows) >= 4096:
            self.flush()

    def flush(self):
        if self.rows:
            self.writer.write_table(pa.Table.from_pylist(self.rows, schema=self.schema))
            self.n_rows += len(self.rows)
            self.rows.clear()

    def close(self):
        self.flush()
        self.writer.close()


def _read_recipe(recipe):
    if isinstance(recipe, dict):
        return recipe
    path = Path(recipe)
    if path.suffix.lower() in {".yaml", ".yml"}:
        return load_curation_yaml(path)
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text())
    raise RecipeError("Recipe path must end in .json, .yaml or .yml")


def _keys(value, allowed, label):
    if not isinstance(value, dict):
        raise RecipeError(f"{label} must be a mapping")
    unexpected = set(value) - allowed
    if unexpected:
        raise RecipeError(f"Unknown {label} keys: {sorted(unexpected)}")


def _validate(recipe):
    _keys(recipe, {"schema_version", "name", "sources", "mapping_policy"}, "recipe")
    if recipe.get("schema_version") != RECIPE_SCHEMA_VERSION:
        raise RecipeError("Unsupported recipe schema version")
    if not isinstance(recipe.get("name"), str) or not recipe["name"].strip():
        raise RecipeError("Recipe needs a nonempty name")
    if recipe.get("mapping_policy") != "no_expansion":
        raise RecipeError("Recipe must explicitly preserve mapping alternatives without expansion")
    sources = recipe.get("sources")
    if not isinstance(sources, list) or not sources:
        raise RecipeError("Recipe needs at least one source request")
    seen = set()
    for source in sources:
        _keys(
            source,
            {
                "evidence_type",
                "row_id",
                "select",
                "where",
                "joins",
                "target",
                "required_fields",
                "context_precision",
                "grouping",
                "limit",
            },
            "source",
        )
        evidence_type = source.get("evidence_type")
        if not isinstance(evidence_type, str) or evidence_type in seen:
            raise RecipeError("Source evidence types must be distinct")
        seen.add(evidence_type)
        if not isinstance(source.get("row_id"), str):
            raise RecipeError(f"{evidence_type} needs a row_id field")
        if not isinstance(source.get("select", []), list):
            raise RecipeError(f"{evidence_type}.select must be a list")
        if not isinstance(source.get("required_fields", []), list):
            raise RecipeError(f"{evidence_type}.required_fields must be a list")
        if not isinstance(source.get("where", {}), dict):
            raise RecipeError(f"{evidence_type}.where must be a mapping")
        if not isinstance(source.get("joins", []), list):
            raise RecipeError(f"{evidence_type}.joins must be a list")
        target = source.get("target")
        _keys(
            target,
            {
                "field",
                "mode",
                "dtype",
                "map",
                "unit_field",
                "allowed_units",
                "scale_field",
                "allowed_scales",
                "inequality_field",
                "allowed_inequalities",
                "status_field",
                "allowed_statuses",
                "unit_conversion",
            },
            "target",
        )
        if target.get("mode") not in {"mapped", "numeric", "category"}:
            raise RecipeError("Target mode must be mapped, numeric or category")
        if not isinstance(target.get("field"), str):
            raise RecipeError("Target needs a field")
        if target["mode"] == "mapped":
            if target.get("dtype") not in TARGET_TYPES or not isinstance(target.get("map"), dict):
                raise RecipeError("Mapped target needs a dtype and source-to-target map")
            dtype = target["dtype"]
            for raw, mapped in target["map"].items():
                if dtype == "boolean":
                    valid = isinstance(mapped, bool)
                elif dtype == "string":
                    valid = isinstance(mapped, str)
                else:
                    valid = (
                        isinstance(mapped, (int, float))
                        and not isinstance(mapped, bool)
                        and math.isfinite(mapped)
                    )
                if not isinstance(raw, str) or not valid:
                    raise RecipeError(f"Invalid mapped target value for {raw!r}")
        elif "map" in target or "dtype" in target:
            raise RecipeError("Only mapped targets accept dtype and map")
        if target["mode"] == "numeric" and not ("unit_field" in target or "scale_field" in target):
            raise RecipeError("Numeric targets need an explicit unit or scale policy")
        if "unit_conversion" in target:
            conversion = target["unit_conversion"]
            _keys(conversion, {"output_unit", "factor_by_unit"}, "unit_conversion")
            if (
                target["mode"] != "numeric"
                or "unit_field" not in target
                or "allowed_units" not in target
                or not isinstance(conversion.get("output_unit"), str)
                or not conversion["output_unit"]
                or not isinstance(conversion.get("factor_by_unit"), dict)
                or not conversion["factor_by_unit"]
            ):
                raise RecipeError("Unit conversion needs a numeric target and source unit policy")
            for unit, factor in conversion["factor_by_unit"].items():
                if (
                    not isinstance(unit, str)
                    or not isinstance(factor, (int, float))
                    or isinstance(factor, bool)
                    or not math.isfinite(factor)
                    or factor <= 0
                ):
                    raise RecipeError("Unit conversion factors must be finite and positive")
            if set(conversion["factor_by_unit"]) != set(target["allowed_units"]):
                raise RecipeError("Unit conversion must cover exactly the allowed source units")
        for field_key, allowed_key in (
            ("unit_field", "allowed_units"),
            ("scale_field", "allowed_scales"),
            ("inequality_field", "allowed_inequalities"),
            ("status_field", "allowed_statuses"),
        ):
            if (field_key in target) != (allowed_key in target):
                raise RecipeError(f"{field_key} and {allowed_key} must be declared together")
            if field_key in target and (
                not isinstance(target[field_key], str) or not isinstance(target[allowed_key], list)
            ):
                raise RecipeError(f"Invalid {field_key} policy")
        if target["mode"] == "numeric" and any(
            inequality != "=" for inequality in target.get("allowed_inequalities", [])
        ):
            raise RecipeError("Censored inequalities cannot be exact numeric point targets")
        context = source.get("context_precision")
        if context is not None:
            _keys(context, {"field", "allowed", "on_mismatch"}, "context_precision")
            if (
                not isinstance(context.get("field"), str)
                or not isinstance(context.get("allowed"), list)
                or context.get("on_mismatch") not in {"ambiguous", "exclude"}
            ):
                raise RecipeError("Invalid context precision policy")
        grouping = source.get("grouping")
        _keys(grouping, {"fields", "on_missing"}, "grouping")
        if (
            not isinstance(grouping.get("fields"), list)
            or not grouping["fields"]
            or grouping.get("on_missing") not in {"mark_uncertain", "ambiguous", "exclude"}
        ):
            raise RecipeError("Invalid grouping policy")


def _requested_fields(source):
    target = source["target"]
    fields = [source["row_id"], *source.get("select", []), *source.get("required_fields", [])]
    fields.append(target["field"])
    for name in ("unit_field", "scale_field", "inequality_field", "status_field"):
        if name in target:
            fields.append(target[name])
    if source.get("context_precision"):
        fields.append(source["context_precision"]["field"])
    if source.get("grouping"):
        fields.extend(source["grouping"]["fields"])
    distinct = tuple(dict.fromkeys(fields))
    if RESERVED.intersection(distinct):
        raise RecipeError(f"Reserved output field: {sorted(RESERVED.intersection(distinct))}")
    return distinct


def _target_type(target):
    if target["mode"] == "numeric":
        return pa.float64()
    if target["mode"] == "category":
        return pa.string()
    return TARGET_TYPES[target["dtype"]]


def _missing(value):
    return value is None or value == "" or value == []


def _lineage_value(row, field):
    direct = row.get(field)
    if not _missing(direct):
        return direct
    related = [
        value for name, value in row.items() if name.endswith(f".{field}") and not _missing(value)
    ]
    if len(related) == 1:
        return related[0]
    return None


def _target_value(row, target):
    value = row[target["field"]]
    if _missing(value):
        return None, "missing_target"
    mode = target["mode"]
    if mode == "mapped":
        key = str(value)
        if key not in target["map"]:
            return None, "unmapped_target"
        result = target["map"][key]
    elif mode == "numeric":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None, "nonnumeric_target"
        if not math.isfinite(value):
            return None, "nonfinite_target"
        result = float(value)
    else:
        if not isinstance(value, str):
            return None, "noncategorical_target"
        result = value
    for field_key, allowed_key, reason in (
        ("unit_field", "allowed_units", "unit_not_allowed"),
        ("scale_field", "allowed_scales", "scale_not_allowed"),
        ("inequality_field", "allowed_inequalities", "inequality_not_allowed"),
        ("status_field", "allowed_statuses", "value_status_not_allowed"),
    ):
        if field_key in target and row[target[field_key]] not in target[allowed_key]:
            return None, reason
    if mode == "numeric" and "unit_conversion" in target:
        result *= target["unit_conversion"]["factor_by_unit"][row[target["unit_field"]]]
        if not math.isfinite(result):
            return None, "nonfinite_converted_target"
    return result, None


def _target_unit(row, target):
    if target["mode"] != "numeric" or "unit_field" not in target:
        return None
    conversion = target.get("unit_conversion")
    return conversion["output_unit"] if conversion else row[target["unit_field"]]


def _group(row, grouping, evidence_type):
    if grouping is None:
        return None, "not_requested", None
    values = {field: row[field] for field in grouping["fields"]}
    if any(_missing(value) for value in values.values()):
        return None, "unresolved", "split_group_unresolved"
    identity = json.dumps([evidence_type, values], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(identity.encode()).hexdigest(), "source_group", None


def _decision(row, source):
    excluded = []
    ambiguous = []
    for field in source.get("required_fields", []):
        if _missing(row[field]):
            excluded.append(f"missing_required:{field}")
    target, target_reason = _target_value(row, source["target"])
    if target_reason:
        ambiguous.append(target_reason)
    context = source.get("context_precision")
    if context and row[context["field"]] not in context["allowed"]:
        bucket = excluded if context["on_mismatch"] == "exclude" else ambiguous
        bucket.append("context_precision_unmet")
    group_id, split_status, group_reason = _group(
        row, source.get("grouping"), source["evidence_type"]
    )
    if group_reason:
        policy = source["grouping"]["on_missing"]
        if policy == "exclude":
            excluded.append(group_reason)
        elif policy == "ambiguous":
            ambiguous.append(group_reason)
    if excluded:
        return "excluded", None, excluded + ambiguous, group_id, split_status
    if ambiguous:
        return "ambiguous", None, ambiguous, group_id, split_status
    return "accepted", target, [], group_id, split_status


def _preflight(store, recipe):
    requests = []
    for source in recipe["sources"]:
        target = source["target"]
        target_evidence = source["evidence_type"]
        if "." in target["field"] and not target["field"].startswith("native."):
            relation = target["field"].split(".", 1)[0]
            target_evidence = store._link(source["evidence_type"], relation).target
        if target["mode"] == "numeric":
            for field, policy in (
                ("measurement_inequality", "inequality_field"),
                ("quantitative_status", "status_field"),
                ("value_status", "status_field"),
            ):
                try:
                    store._field(target_evidence, field)
                except EvidenceFieldError:
                    continue
                if policy not in target:
                    raise RecipeError(
                        f"Numeric target {target_evidence} needs an explicit {policy} policy"
                    )
        links = [
            link
            for link in LINKS
            if link.source == source["evidence_type"]
            and link.name == "source_contributors"
            and link.target in store.bindings
        ]
        contributor_link = links[0] if links else None
        fields = _requested_fields(source)
        source_fields = []
        for field in (
            "source_record_id",
            "source_assay_id",
            "source_dataset",
            "source_reference_id",
        ):
            try:
                store._field(source["evidence_type"], field)
            except EvidenceFieldError:
                continue
            source_fields.append(field)
        fields = tuple(dict.fromkeys((*fields, *source_fields)))
        if contributor_link is not None:
            fields = tuple(dict.fromkeys((*fields, contributor_link.source_key)))
        store._prepare_query(
            source["evidence_type"],
            fields,
            source.get("where", {}),
            source.get("joins", []),
            source.get("limit"),
        )
        requests.append((source, fields, contributor_link))
    return requests


def _append_source_contributors(store, database, link, source_name, sink):
    target = store.bindings[link.target][0]
    mhc = link.target == "source_contributor"
    columns = [link.target_key, "source_record_id", "source_dataset"]
    columns.extend(
        ["original_pmid", "source_row", "relationships"]
        if mhc
        else ["source_assay_id", "source_reference_id", "csv_data_row_1based"]
    )
    n_candidates = database.execute(
        "SELECT COUNT(*) FROM candidate WHERE source_type = ?", (source_name,)
    ).fetchone()[0]
    lookup = None
    expression = None
    if n_candidates <= 100_000:
        lookup = defaultdict(list)
        for key, candidate_id in database.execute(
            "SELECT contributor_key, id FROM candidate WHERE source_type = ? "
            "AND contributor_key IS NOT NULL",
            (source_name,),
        ):
            lookup[key].append(candidate_id)
        expression = ds.field(link.target_key).isin(list(lookup))
    scanner = ds.dataset(store.directory / target.materialized_artifact, format="parquet").scanner(
        columns=columns, filter=expression, batch_size=65_536
    )
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            key = str(row[link.target_key])
            candidates = (
                lookup.get(key, ())
                if lookup is not None
                else (
                    candidate_id
                    for (candidate_id,) in database.execute(
                        "SELECT id FROM candidate WHERE source_type = ? AND contributor_key = ?",
                        (source_name, key),
                    )
                )
            )
            for candidate_id in candidates:
                sink.add(
                    {
                        "candidate_id": candidate_id,
                        "source_record_id": row["source_record_id"],
                        "source_assay_id": None if mhc else row["source_assay_id"],
                        "source_dataset": row["source_dataset"],
                        "source_reference_id": (
                            row["original_pmid"] if mhc else row["source_reference_id"]
                        ),
                        "link_status": "provider_source_record",
                        "source_row_index": (
                            row["source_row"] if mhc else row["csv_data_row_1based"]
                        ),
                        "relationships_raw": row["relationships"] if mhc else None,
                    }
                )
                database.execute("INSERT OR IGNORE INTO linked VALUES (?)", (candidate_id,))


def _append_unresolved_contributors(database, source_name, sink):
    n_unresolved_links = 0
    for row in database.execute(
        "SELECT c.id, c.source_record_id, c.source_assay_id, c.source_dataset, "
        "c.source_reference_id FROM candidate c LEFT JOIN linked l ON c.id = l.id "
        "WHERE c.source_type = ? AND l.id IS NULL",
        (source_name,),
    ):
        candidate_id, source_record_id, source_assay_id, source_dataset, source_reference_id = row
        sink.add(
            {
                "candidate_id": candidate_id,
                "source_record_id": source_record_id,
                "source_assay_id": source_assay_id,
                "source_dataset": source_dataset,
                "source_reference_id": source_reference_id,
                "link_status": "source_contributor_unresolved",
            }
        )
        n_unresolved_links += 1
    return n_unresolved_links


def materialize_recipe(store, recipe, destination):
    """Write all candidate decisions without changing the verified source package."""
    recipe = _read_recipe(recipe)
    if isinstance(recipe, dict) and "registered_recipe" in recipe:
        from .digestion_recipe import RECIPE_ID as DIGESTION_ID
        from .digestion_recipe import materialize_timecourses
        from .distribution_recipe import materialize_distribution
        from .hap1_response_recipe import RECIPE_ID as HAP1_RESPONSE_ID
        from .hap1_response_recipe import materialize_responses as materialize_hap1_responses
        from .mapp_controlled_recipe import CONTRAST_ID as CONTROLLED_CONTRAST_ID
        from .mapp_controlled_recipe import RECIPE_ID as CONTROLLED_ID
        from .mapp_controlled_recipe import materialize_controlled
        from .ms_calibration_response import RECIPE_ID, materialize_response
        from .tmab_response_recipe import RECIPE_ID as NATIVE_RESPONSE_ID
        from .tmab_response_recipe import materialize_responses

        if recipe["registered_recipe"] == HAP1_RESPONSE_ID:
            return materialize_hap1_responses(store, recipe, destination)
        if recipe["registered_recipe"] == NATIVE_RESPONSE_ID:
            return materialize_responses(store, recipe, destination)
        if recipe["registered_recipe"] == DIGESTION_ID:
            return materialize_timecourses(store, recipe, destination)
        if recipe["registered_recipe"] in (CONTROLLED_ID, CONTROLLED_CONTRAST_ID):
            return materialize_controlled(store, recipe, destination)
        if recipe["registered_recipe"] == RECIPE_ID:
            return materialize_response(store, recipe, destination)
        return materialize_distribution(store, recipe, destination)
    _validate(recipe)
    requests = _preflight(store, recipe)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Training output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        database = sqlite3.connect(staging / "candidate_ids.sqlite")
        database.execute(
            "CREATE TABLE candidate ("
            "id TEXT PRIMARY KEY, source_type TEXT, contributor_key TEXT, "
            "source_record_id TEXT, source_assay_id TEXT, source_dataset TEXT, "
            "source_reference_id TEXT)"
        )
        database.execute(
            "CREATE INDEX candidate_source_key ON candidate(source_type, contributor_key)"
        )
        database.execute("CREATE TABLE linked (id TEXT PRIMARY KEY)")
        coverage = {}
        for source, fields, contributor_link in requests:
            name = source["evidence_type"]
            n_source_rows = store.count(name)
            n_query_matches = store.count(
                name, where=source.get("where", {}), joins=source.get("joins", [])
            )
            empty = store.query(
                name,
                select=fields,
                where=source.get("where", {}),
                joins=source.get("joins", []),
                limit=0,
            ).table
            output_schema = pa.schema(
                [
                    ("candidate_id", pa.string()),
                    ("evidence_type", pa.string()),
                    ("source_population", pa.string()),
                    *[(field.name, field.type) for field in empty.schema],
                    ("target_value", _target_type(source["target"])),
                    ("target_unit", pa.string()),
                    ("decision", pa.string()),
                    ("reason_codes", pa.list_(pa.string())),
                    ("split_group_id", pa.string()),
                    ("split_status", pa.string()),
                ]
            )
            writers = {
                decision: _Writer(staging / "sources" / name / f"{decision}.parquet", output_schema)
                for decision in DECISIONS
            }
            contributors = _Writer(
                staging / "sources" / name / "contributors.parquet", CONTRIBUTOR_SCHEMA
            )
            lineage = _Writer(staging / "sources" / name / "lineage.parquet", LINEAGE_SCHEMA)
            counts = Counter()
            reasons = Counter()
            n_unresolved_links = 0
            try:
                for batch in store.iter_query(
                    name,
                    select=fields,
                    where=source.get("where", {}),
                    joins=source.get("joins", []),
                    limit=source.get("limit"),
                ):
                    for row in batch.table.to_pylist():
                        identity = row[source["row_id"]]
                        if _missing(identity):
                            raise RecipeError(f"Missing candidate identity: {name}")
                        candidate_id = f"{name}:{identity}"
                        try:
                            database.execute(
                                "INSERT INTO candidate VALUES (?, ?, ?, ?, ?, ?, ?)",
                                (
                                    candidate_id,
                                    name,
                                    str(row[contributor_link.source_key])
                                    if contributor_link is not None
                                    and row[contributor_link.source_key] is not None
                                    else None,
                                    _lineage_value(row, "source_record_id"),
                                    _lineage_value(row, "source_assay_id"),
                                    _lineage_value(row, "source_dataset"),
                                    _lineage_value(row, "source_reference_id"),
                                ),
                            )
                        except sqlite3.IntegrityError as error:
                            raise RecipeError(
                                f"Duplicate candidate identity: {candidate_id}"
                            ) from error
                        decision, target, codes, group_id, split_status = _decision(row, source)
                        counts[decision] += 1
                        reasons.update(codes)
                        writers[decision].add(
                            {
                                "candidate_id": candidate_id,
                                "evidence_type": name,
                                "source_population": batch.source_population,
                                **row,
                                "target_value": target,
                                "target_unit": (
                                    _target_unit(row, source["target"])
                                    if decision == "accepted"
                                    else None
                                ),
                                "decision": decision,
                                "reason_codes": codes,
                                "split_group_id": group_id,
                                "split_status": split_status,
                            }
                        )
                        source_record_id = _lineage_value(row, "source_record_id")
                        source_assay_id = _lineage_value(row, "source_assay_id")
                        if contributor_link is None:
                            if not (source_record_id or source_assay_id):
                                n_unresolved_links += 1
                            contributors.add(
                                {
                                    "candidate_id": candidate_id,
                                    "source_record_id": source_record_id,
                                    "source_assay_id": source_assay_id,
                                    "source_dataset": _lineage_value(row, "source_dataset"),
                                    "source_reference_id": _lineage_value(
                                        row, "source_reference_id"
                                    ),
                                    "link_status": (
                                        "direct_source_key"
                                        if source_record_id or source_assay_id
                                        else "source_key_unresolved"
                                    ),
                                }
                            )
                        lineage.add(
                            {
                                "candidate_id": candidate_id,
                                "split_group_id": group_id,
                                "split_status": split_status,
                                "group_fields_json": json.dumps(
                                    {
                                        field: row[field]
                                        for field in source.get("grouping", {}).get("fields", [])
                                    },
                                    sort_keys=True,
                                ),
                            }
                        )
            finally:
                for writer in (*writers.values(), lineage):
                    writer.close()
            if contributor_link is not None:
                _append_source_contributors(store, database, contributor_link, name, contributors)
                n_unresolved_links = _append_unresolved_contributors(database, name, contributors)
            contributors.close()
            coverage[name] = {
                "n_source_rows": n_source_rows,
                "n_query_matches": n_query_matches,
                "n_filtered_by_query": n_source_rows - n_query_matches,
                "n_limit_omitted": n_query_matches - sum(counts.values()),
                "n_candidates": sum(counts.values()),
                "n_contributor_links": contributors.n_rows,
                "n_unresolved_contributor_links": n_unresolved_links,
                **{f"n_{decision}": counts[decision] for decision in DECISIONS},
                "reason_counts": dict(sorted(reasons.items())),
                "source_population": store.bindings[name][0].source_population,
                "grain": store.bindings[name][0].grain,
                "row_id": source["row_id"],
            }
        database.close()
        (staging / "candidate_ids.sqlite").unlink()
        (staging / "coverage.json").write_text(
            json.dumps(coverage, indent=2, sort_keys=True) + "\n"
        )
        files = {
            str(path.relative_to(staging)): file_digest(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "catalog_schema_version": SCHEMA_VERSION,
            "hitlist_version": store._software_version(),
            "recipe": recipe,
            "recipe_sha256": hashlib.sha256(
                json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "source_package_manifest_sha256": store._manifest_hash(),
            "source_package_schema_version": store.manifest["schema_version"],
            "coverage": coverage,
            "files": files,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        verify_recipe_package(staging, source_package=store.directory)
        staging.rename(destination)
        return destination / "manifest.json"
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_recipe_package(directory, *, source_package=None):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != OUTPUT_SCHEMA_VERSION:
        raise RecipeError("Unsupported training output schema")
    files = manifest.get("files", {})
    actual = {
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file() and path != directory / "manifest.json"
    }
    if set(files) != actual:
        raise RecipeError("Training output file inventory mismatch")
    for relative, digest in files.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or file_digest(directory / path) != digest:
            raise RecipeError(f"Training output artifact mismatch: {relative}")
    recipe_digest = hashlib.sha256(
        json.dumps(manifest["recipe"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if recipe_digest != manifest["recipe_sha256"]:
        raise RecipeError("Training recipe digest mismatch")
    if (
        source_package is not None
        and file_digest(Path(source_package) / "manifest.json")["sha256"]
        != manifest["source_package_manifest_sha256"]
    ):
        raise RecipeError("Source package manifest changed")
    output_kind = manifest.get("output_kind", "row_decisions")
    if output_kind == "registered_hap1_response":
        from .hap1_response_recipe import verify_response_contents

        return verify_response_contents(directory, manifest, source_package)
    if output_kind == "registered_native_response":
        from .tmab_response_recipe import verify_response_contents

        return verify_response_contents(directory, manifest, source_package)
    if output_kind == "registered_digestion_timecourses":
        from .digestion_recipe import verify_timecourse_contents

        return verify_timecourse_contents(directory, manifest, source_package)
    if output_kind in ("registered_processing_runs", "registered_processing_contrast"):
        from .mapp_controlled_recipe import verify_controlled_contents

        return verify_controlled_contents(directory, manifest, source_package)
    if output_kind == "registered_response":
        from .ms_calibration_response import verify_response_contents

        return verify_response_contents(directory, manifest, source_package)
    if output_kind == "registered_distribution":
        from .distribution_recipe import verify_distribution_contents

        return verify_distribution_contents(directory, manifest)
    if output_kind != "row_decisions":
        raise RecipeError(f"Unknown training output kind: {output_kind}")
    for name, report in manifest["coverage"].items():
        if (
            report["n_filtered_by_query"] + report["n_query_matches"] != report["n_source_rows"]
            or report["n_candidates"] + report["n_limit_omitted"] != report["n_query_matches"]
        ):
            raise RecipeError(f"Training coverage does not reconcile: {name}")
        paths = {
            decision: directory / "sources" / name / f"{decision}.parquet" for decision in DECISIONS
        }
        if (
            sum(pq.ParquetFile(path).metadata.num_rows for path in paths.values())
            != report["n_candidates"]
        ):
            raise RecipeError(f"Training decision count mismatch: {name}")
        for decision, path in paths.items():
            if pq.ParquetFile(path).metadata.num_rows != report[f"n_{decision}"]:
                raise RecipeError(f"Training {decision} count mismatch: {name}")
        expected_relations = {
            "contributors": report["n_contributor_links"],
            "lineage": report["n_candidates"],
        }
        if report["n_contributor_links"] < report["n_candidates"]:
            raise RecipeError(f"Training contributor coverage is incomplete: {name}")
        for relation, expected_rows in expected_relations.items():
            if (
                pq.ParquetFile(
                    directory / "sources" / name / f"{relation}.parquet"
                ).metadata.num_rows
                != expected_rows
            ):
                raise RecipeError(f"Training {relation} count mismatch: {name}")
    return manifest
