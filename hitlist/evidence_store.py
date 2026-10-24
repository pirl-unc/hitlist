"""Verified, typed queries over a frozen multi-modality evidence package."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .assay_adjudication import verify_assay_adjudication_package
from .evidence_bindings import ALIASES, CONSTANTS, LINKS, SELECTORS, materialized_bindings
from .evidence_catalog import EVIDENCE, FIELDS, NULL_STATES, SCHEMA_VERSION, TASKS
from .evidence_package import adjudication_manifest_path, package_family
from .reference_evidence import verify_reference_evidence_package
from .supporting_numeric_evidence import verify_supporting_numeric_evidence_package


class EvidenceQueryError(ValueError):
    """A request cannot be answered at the declared evidence grain."""


class EvidenceFieldError(EvidenceQueryError):
    """A requested logical field lacks a safe physical binding."""


class EvidenceRelationshipError(EvidenceQueryError):
    """A requested relationship lacks the required key or cardinality."""


@dataclass(frozen=True)
class QueryResult:
    evidence_type: str
    grain: str
    source_population: str
    manifest_sha256: str
    catalog_schema_version: int
    table: pa.Table
    joined_relationships: tuple[dict, ...] = ()

    @property
    def n_rows(self) -> int:
        return self.table.num_rows

    def to_batches(self, max_chunksize: int = 16_384) -> list[pa.RecordBatch]:
        return self.table.to_batches(max_chunksize=max_chunksize)


@dataclass(frozen=True)
class _QueryPlan:
    evidence_type: str
    select: tuple[str, ...]
    sources: dict
    joins: dict
    physical: tuple[str, ...]
    where: dict
    related_keys: tuple
    limit: int | None


def _predicate(expression, column, operator, value):
    field = ds.field(column)
    if operator in {"eq", "ne"} and value is None:
        raise EvidenceQueryError("Use an explicit is_null predicate for missing values")
    if operator == "eq":
        part = field == value
    elif operator == "ne":
        part = field != value
    elif operator == "in":
        if not isinstance(value, (tuple, list, set, frozenset)):
            raise EvidenceQueryError("'in' requires a sequence of values")
        part = field.isin(list(value))
    elif operator == "lt":
        part = field < value
    elif operator == "le":
        part = field <= value
    elif operator == "gt":
        part = field > value
    elif operator == "ge":
        part = field >= value
    elif operator == "is_null":
        if not isinstance(value, bool):
            raise EvidenceQueryError("'is_null' requires a boolean")
        part = field.is_null() if value else field.is_valid()
    else:
        raise EvidenceQueryError(f"Unsupported predicate: {operator}")
    return part if expression is None else expression & part


def _condition(value):
    if isinstance(value, dict):
        if set(value) != {"op", "value"}:
            raise EvidenceQueryError("Predicate must have only 'op' and 'value'")
        return value["op"], value["value"]
    return "eq", value


def _constant_matches(constant, operator, value):
    if operator == "eq":
        return constant == value
    if operator == "ne":
        return constant != value
    if operator == "in":
        return constant in value
    if operator == "is_null":
        return (constant is None) == value
    raise EvidenceQueryError(f"Unsupported constant predicate: {operator}")


class EvidenceStore:
    """One verified package with explicit evidence grains and physical bindings.

    ``open`` hashes and verifies the complete package once. Every subsequent
    request is bounded by default; callers can explicitly pass ``limit=None``.
    """

    def __init__(self, directory: Path, manifest: dict):
        self.directory = directory
        self.manifest = manifest
        family = package_family(manifest)
        self.bindings = materialized_bindings(
            manifest["schema_version"],
            appended_source_ids=manifest.get("appended_sources", {}),
            include_adjudications=adjudication_manifest_path(manifest) is not None,
        )
        if family == "proteometools_ms1":
            self.bindings = {
                name: binding
                for name, binding in self.bindings.items()
                if name.startswith("proteometools_")
            }
        elif family == "hap1_quantitative":
            self.bindings = {
                name: binding for name, binding in self.bindings.items() if name.startswith("hap1_")
            }
        elif family == "tmab_peptidomics":
            self.bindings = {
                name: binding for name, binding in self.bindings.items() if name.startswith("tmab_")
            }
        elif family == "tcell_processing":
            from .tcell_processing_package import MATERIALIZED_EVIDENCE_TYPES

            self.bindings = {
                name: binding
                for name, binding in self.bindings.items()
                if name in MATERIALIZED_EVIDENCE_TYPES
                or name in {"tcell_processing_family", "tcell_processing_availability"}
            }
        elif family == "digestion_processing":
            from .digestion_processing_package import MATERIALIZED_EVIDENCE_TYPES

            self.bindings = {
                name: binding
                for name, binding in self.bindings.items()
                if name in MATERIALIZED_EVIDENCE_TYPES
            }
        elif family == "precursor_context":
            from .precursor_context_package import MATERIALIZED_EVIDENCE_TYPES

            self.bindings = {
                name: binding
                for name, binding in self.bindings.items()
                if name in MATERIALIZED_EVIDENCE_TYPES
            }
        self.schemas = {}
        for name, (item, selectors, aliases) in self.bindings.items():
            path = directory / item.materialized_artifact
            schema = pq.ParquetFile(path).schema_arrow
            columns = set(schema.names)
            for selector in selectors:
                if selector.column not in columns:
                    raise EvidenceQueryError(f"Missing selector {name}.{selector.column}")
            for logical, physical in aliases.items():
                if logical not in item.fields or physical not in columns:
                    raise EvidenceQueryError(f"Invalid physical alias {name}.{logical}")
            for constant in CONSTANTS.get(name, {}):
                if constant not in item.fields:
                    raise EvidenceQueryError(f"Invalid constant {name}.{constant}")
            self.schemas[name] = schema
        for link in LINKS:
            if link.source not in self.schemas or link.target not in self.schemas:
                continue
            if (
                link.source_key not in self.schemas[link.source].names
                or link.target_key not in self.schemas[link.target].names
            ):
                raise EvidenceRelationshipError(f"Missing relationship key: {link.name}")

    @classmethod
    def open(cls, directory: str | Path) -> EvidenceStore:
        directory = Path(directory)
        envelope = json.loads((directory / "manifest.json").read_text())
        try:
            family = package_family(envelope)
        except ValueError as exc:
            raise EvidenceQueryError(str(exc)) from exc
        if family == "proteometools_ms1":
            from .proteometools_ms1 import verify_proteometools_ms1_package

            manifest = verify_proteometools_ms1_package(directory)
        elif family == "hap1_quantitative":
            from .hap1_quantitative import verify_hap1_quantitative_package

            manifest = verify_hap1_quantitative_package(directory)
        elif family == "external":
            from .external_successor import verify_external_successor

            manifest = verify_external_successor(directory)
        elif family == "binding_review":
            from .binding_assay_context import verify_binding_assay_context_package

            manifest = verify_binding_assay_context_package(directory)
        elif family == "gomez_source":
            from .gomez_source_lineage import verify_gomez_source_package

            manifest = verify_gomez_source_package(directory)
        elif family == "gomez_arm_projection":
            from .gomez_arm_projection import verify_gomez_arm_projection_package

            manifest = verify_gomez_arm_projection_package(directory)
        elif family == "gomez_prediction_panel":
            from .gomez_prediction_panel import verify_gomez_prediction_panel_package

            manifest = verify_gomez_prediction_panel_package(directory)
        elif family == "binding_context_availability":
            from .binding_context_availability import (
                verify_binding_context_availability_package,
            )

            manifest = verify_binding_context_availability_package(directory)
        elif family == "tmab_peptidomics":
            from .tmab_peptidomics import verify_tmab_peptidomics_package

            manifest = verify_tmab_peptidomics_package(directory)
        elif family == "tcell_processing":
            from .tcell_processing_package import verify_tcell_processing_package

            manifest = verify_tcell_processing_package(directory)
        elif family == "digestion_processing":
            from .digestion_processing_package import verify_digestion_processing_package

            manifest = verify_digestion_processing_package(directory)
        elif family == "precursor_context":
            from .precursor_context_package import verify_precursor_context_package

            manifest = verify_precursor_context_package(directory)
        elif family == "adjudication":
            manifest = verify_assay_adjudication_package(directory)
        elif family == "reference":
            manifest = verify_reference_evidence_package(directory)
        else:
            manifest = verify_supporting_numeric_evidence_package(directory)
        return cls(directory, manifest)

    def _field(self, evidence_type: str, field: str):
        item, _, aliases = self.bindings[evidence_type]
        schema = self.schemas[evidence_type]
        if field.startswith("native."):
            physical = field.removeprefix("native.")
            if physical in schema.names:
                return "physical", physical
            raise EvidenceFieldError(f"Unknown native field: {evidence_type}.{field}")
        if field in CONSTANTS.get(evidence_type, {}):
            return "constant", CONSTANTS[evidence_type][field]
        if field in item.fields:
            physical = aliases.get(field, field)
            if physical in schema.names:
                return "physical", physical
            raise EvidenceFieldError(
                f"{evidence_type}.{field} is declared but unavailable at {item.grain} grain"
            )
        raise EvidenceFieldError(f"Unknown field: {evidence_type}.{field}")

    def _canonical_evidence_type(self, evidence_type):
        aliases = self.manifest.get("query_aliases", {})
        canonical = aliases.get(evidence_type, evidence_type)
        if not isinstance(canonical, str):
            raise EvidenceQueryError(f"Invalid query alias: {evidence_type}")
        return canonical

    def _description(self, item):
        if item.name not in self.bindings:
            return {
                "name": item.name,
                "status": (
                    "not_in_generation" if item.status == "v2_materialized" else "contract_only"
                ),
                "kind": item.kind,
                "family": item.family,
                "grain": item.grain,
                "source_population": item.source_population,
                "rights_scope": item.source_population,
                "endpoints": list(item.endpoints),
                "ingest_required": list(item.index_required),
                "fields": dict.fromkeys(
                    item.fields,
                    "not_in_generation" if item.status == "v2_materialized" else "contract_only",
                ),
                "native_fields": [],
                "relationships": [],
            }
        schema = self.schemas[item.name]
        related = {}
        for link in LINKS:
            if link.source != item.name or link.target not in self.bindings:
                continue
            for field in self.bindings[link.target][0].fields:
                try:
                    self._field(link.target, field)
                except EvidenceFieldError:
                    continue
                related.setdefault(field, []).append(link.name)
        readiness = {}
        for field in item.fields:
            try:
                kind, value = self._field(item.name, field)
                readiness[field] = {"status": kind, "source": value}
            except EvidenceFieldError:
                links = related.get(field, [])
                readiness[field] = {
                    "status": "related" if links else "unavailable",
                    "relationships": links,
                }
        return {
            "name": item.name,
            "status": "materialized",
            "kind": item.kind,
            "family": item.family,
            "grain": item.grain,
            "source_population": item.source_population,
            "rights_scope": item.source_population,
            "endpoints": list(item.endpoints),
            "ingest_required": list(item.index_required),
            "artifact": item.materialized_artifact,
            "n_rows": self.count(item.name),
            "selectors": [vars(selector) for selector in SELECTORS.get(item.name, ())],
            "fields": readiness,
            "native_fields": [f"native.{name}" for name in schema.names],
            "relationships": [
                vars(link)
                for link in LINKS
                if link.source == item.name and link.target in self.bindings
            ],
        }

    def catalog(self) -> dict:
        """Describe readiness from actual verified physical schemas."""
        from .distribution_recipe import describe_registered_recipes

        return {
            "schema_version": SCHEMA_VERSION,
            "hitlist_version": self._software_version(),
            "package_schema_version": self.manifest["schema_version"],
            "package_hitlist_version": self.manifest["hitlist_version"],
            "package_manifest_sha256": self._manifest_hash(),
            "null_states": dict(NULL_STATES),
            "evidence": [self._description(item) for item in EVIDENCE],
            "tasks": [vars(task) for task in TASKS],
            "fields": [vars(field) for field in FIELDS],
            "source_registry": self._source_registry(),
            "registered_recipes": describe_registered_recipes(self),
            "query_aliases": dict(self.manifest.get("query_aliases", {})),
        }

    def _source_registry(self):
        """Expose verified source revisions and rights evidence at source scope."""
        family = package_family(self.manifest)
        if family in ("hap1_quantitative", "proteometools_ms1"):
            return self.manifest.get("source_registry", {})
        if family == "tmab_peptidomics":
            return self.manifest.get("source_registry", {})
        if family == "tcell_processing":
            return self.manifest.get("source_registry", {})
        if family == "digestion_processing":
            return self.manifest.get("source_registry", {})
        if family == "precursor_context":
            return self.manifest.get("source_registry", {})
        mhc = json.loads((self.directory / "manifests" / "mhc.json").read_text())
        context = json.loads((self.directory / "manifests" / "context.json").read_text())
        supporting = json.loads((self.directory / "manifests" / "supporting.json").read_text())
        tcell = json.loads((self.directory / "manifests" / "tcell.json").read_text())
        external = json.loads((self.directory / "manifests" / "external.json").read_text())
        registry = {
            "mhc_v4": {
                name: {
                    "license_status": details.get("license_status"),
                    "access": details.get("access"),
                    "snapshot_sha256": details.get("snapshot", {}).get("sha256"),
                }
                for name, details in sorted(mhc["source_registry"].items())
            },
            "human_cohort_corrected_v1": context["source_registry"]["corrected_human_cohort"],
            "cohort_and_current_curation": context["source_registry"],
            "bulk_v4": supporting["source_registry"]["bulk_v4"],
            "line_expression_v4": supporting["source_registry"]["line_expression_v4"],
            "tcell_2026_10_20": tcell["source_registry"],
            "pxd057588": {
                "asset_receipt_sha256": external["sources"]["pxd057588"]["asset_receipt_sha256"],
                "rights_relation": "external_source_asset",
            },
        }
        if self.manifest["schema_version"] >= 7:
            reference_manifest = (
                self.manifest
                if self.manifest["schema_version"] == 7
                else json.loads((self.directory / "manifests/reference.json").read_text())
            )
            registry["pxd057588_mouse_reference_2026_03"] = reference_manifest["reference"]
        review_path = adjudication_manifest_path(self.manifest)
        if review_path is not None:
            review = (
                self.manifest
                if review_path == "manifest.json"
                else json.loads((self.directory / review_path).read_text())
            )
            registry["tcell_multimer_review_2026_10_23"] = {
                "revision": review["review_revision"],
                "source_review_sha256": review["source_review_sha256"],
                "n_assay_reviews": review["reconciliation"]["n_assay_reviews"],
                "rights_scope": "publisher-linked review assertions; source PDFs not redistributed",
            }
        for source_id, source in self.manifest.get("appended_sources", {}).items():
            registry[source_id] = {
                **source["registry"],
                "asset_receipt_sha256": source["asset_receipt_sha256"],
                "adapter_schema_version": source["adapter_schema_version"],
            }
        if family in {
            "gomez_source",
            "gomez_arm_projection",
            "gomez_prediction_panel",
        }:
            registry["gomez_zepeda_2024_original"] = json.loads(
                (self.directory / "gomez_source/registry.json").read_text()
            )
        if family in {"gomez_arm_projection", "gomez_prediction_panel"}:
            registry["gomez_zepeda_2024_arm_projection"] = json.loads(
                (self.directory / "gomez_projection/review.json").read_text()
            )
        if family == "gomez_prediction_panel":
            registry["gomez_zepeda_2024_prediction_panels"] = json.loads(
                (self.directory / "gomez_prediction/review.json").read_text()
            )
        if family == "binding_context_availability":
            registry["binding_assay_context_availability"] = json.loads(
                (self.directory / "binding_context/coverage.json").read_text()
            )
        return registry

    def _manifest_hash(self):
        from .provenance import file_digest

        return file_digest(self.directory / "manifest.json")["sha256"]

    @staticmethod
    def _software_version():
        from .version import __version__

        return __version__

    def _expression(self, name, where, related_keys=()):
        _, selectors, _ = self.bindings[name]
        expression = None
        for selector in selectors:
            expression = _predicate(expression, selector.column, selector.operator, selector.value)
        for field, requested in where.items():
            operator, value = _condition(requested)
            kind, source = self._field(name, field)
            if kind == "constant":
                if not _constant_matches(source, operator, value):
                    return False
            else:
                expression = _predicate(expression, source, operator, value)
        for source_key, keys in related_keys:
            expression = _predicate(expression, source_key, "in", keys)
        return expression

    def _scan_batches(self, name, physical, where, limit, related_keys=()):
        expression = self._expression(name, where, related_keys)
        if expression is False:
            return
        item = self.bindings[name][0]
        path = self.directory / item.materialized_artifact
        scanner = ds.dataset(path, format="parquet").scanner(
            columns=list(physical), filter=expression, batch_size=16_384
        )
        n_rows = 0
        for batch in scanner.to_batches():
            if limit is not None and n_rows + batch.num_rows > limit:
                batch = batch.slice(0, limit - n_rows)
            if batch.num_rows:
                yield batch
            n_rows += batch.num_rows
            if limit is not None and n_rows >= limit:
                break

    def _scan(self, name, physical, where, limit, related_keys=()):
        batches = list(self._scan_batches(name, physical, where, limit, related_keys))
        schema = pa.schema([self.schemas[name].field(column) for column in physical])
        return pa.Table.from_batches(batches, schema=schema)

    def count(self, evidence_type, *, where=None, joins=()):
        """Count source rows after selectors and predicates, before any limit."""
        evidence_type = self._canonical_evidence_type(evidence_type)
        if where is None:
            where = {}
        schema = self.schemas.get(evidence_type)
        if schema is None:
            raise EvidenceQueryError(f"Unknown or unmaterialized evidence type: {evidence_type}")
        plan = self._prepare_query(
            evidence_type, (f"native.{schema.names[0]}",), where, joins, None
        )
        expression = self._expression(evidence_type, plan.where, plan.related_keys)
        if expression is False:
            return 0
        item = self.bindings[evidence_type][0]
        return ds.dataset(self.directory / item.materialized_artifact, format="parquet").count_rows(
            filter=expression
        )

    def _link(self, evidence_type, name):
        links = [
            link
            for link in LINKS
            if link.source == evidence_type and link.name == name and link.target in self.bindings
        ]
        if len(links) != 1:
            raise EvidenceRelationshipError(f"Unknown relationship: {evidence_type}.{name}")
        link = links[0]
        if link.cardinality not in {"one", "optional_one"}:
            raise EvidenceRelationshipError(
                f"{evidence_type}.{name} has {link.cardinality} cardinality; "
                "query the relation as evidence instead"
            )
        return link

    def _target_scan(self, link, physical, where=None, keys=None):
        expression = None
        for selector in SELECTORS.get(link.target, ()):
            expression = _predicate(expression, selector.column, selector.operator, selector.value)
        if where is not None:
            field, requested = where
            operator, value = _condition(requested)
            kind, source = self._field(link.target, field)
            if kind == "constant":
                if not _constant_matches(source, operator, value):
                    return pa.Table.from_arrays(
                        [
                            pa.array([], type=self.schemas[link.target].field(column).type)
                            for column in physical
                        ],
                        names=physical,
                    )
            else:
                expression = _predicate(expression, source, operator, value)
        if keys is not None:
            expression = _predicate(expression, link.target_key, "in", keys)
        item = self.bindings[link.target][0]
        return ds.dataset(self.directory / item.materialized_artifact, format="parquet").to_table(
            columns=physical, filter=expression
        )

    def _related_filter_keys(self, link, field, requested):
        right = self._target_scan(link, [link.target_key], where=(field, requested))
        if right.num_rows > 1_000_000:
            raise EvidenceRelationshipError(
                f"Related predicate on {link.name} matches over 1,000,000 keys; "
                "narrow it before joining"
            )
        return tuple({key for key in right[link.target_key].to_pylist() if key is not None})

    def _joined_values(self, link, field, base):
        kind, source = self._field(link.target, field)
        source_keys = base[link.source_key].to_pylist()
        dtype = (
            self.schemas[link.target].field(source).type
            if kind == "physical"
            else pa.scalar(source).type
        )
        if link.cardinality == "one" and any(key is None for key in source_keys):
            raise EvidenceRelationshipError(f"{link.source}.{link.name} has missing source keys")
        keys = tuple({key for key in source_keys if key is not None})
        if not keys:
            return pa.nulls(len(source_keys), type=dtype)
        physical = [link.target_key] + ([source] if kind == "physical" else [])
        right = self._target_scan(link, physical, keys=keys)
        values = {}
        for row in right.to_pylist():
            key = row[link.target_key]
            if key in values:
                raise EvidenceRelationshipError(
                    f"{link.source}.{link.name} has multiple target rows for {key}"
                )
            values[key] = row[source] if kind == "physical" else source
        if link.cardinality == "one" and any(key not in values for key in keys):
            raise EvidenceRelationshipError(f"{link.source}.{link.name} has missing target rows")
        return pa.array([values.get(key) for key in source_keys], type=dtype)

    def _prepare_query(self, evidence_type, select, where, joins, limit):
        if evidence_type not in self.bindings:
            names = {item.name for item in EVIDENCE}
            detail = "unmaterialized" if evidence_type in names else "unknown"
            raise EvidenceQueryError(f"{detail} evidence type: {evidence_type}")
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 0
        ):
            raise EvidenceQueryError("limit must be a nonnegative integer or None")
        if isinstance(joins, str) or len(joins) != len(set(joins)):
            raise EvidenceRelationshipError("joins must contain distinct relationship names")
        item = self.bindings[evidence_type][0]
        join_map = {name: self._link(evidence_type, name) for name in joins}
        if select is None:
            select = tuple(
                field
                for field in item.fields
                if field in CONSTANTS.get(evidence_type, {})
                or ALIASES.get(evidence_type, {}).get(field, field)
                in self.schemas[evidence_type].names
            )
        elif isinstance(select, str):
            raise EvidenceQueryError("select must be a sequence of field names")
        else:
            select = tuple(select)
        if not select or len(select) != len(set(select)):
            raise EvidenceQueryError("Select at least one distinct field")
        if where is None:
            where = {}
        if not isinstance(where, dict):
            raise EvidenceQueryError("where must map fields to predicates")
        sources = {}
        for field in select:
            if "." in field and not field.startswith("native."):
                relation, related_field = field.split(".", 1)
                if relation not in join_map:
                    raise EvidenceRelationshipError(f"Name relationship in joins: {relation}")
                self._field(join_map[relation].target, related_field)
                sources[field] = ("related", (join_map[relation], related_field))
            else:
                sources[field] = self._field(evidence_type, field)
        direct_where = {}
        related_keys = []
        for field, requested in where.items():
            if "." in field and not field.startswith("native."):
                relation, related_field = field.split(".", 1)
                if relation not in join_map:
                    raise EvidenceRelationshipError(f"Name relationship in joins: {relation}")
                link = join_map[relation]
                self._field(link.target, related_field)
                related_keys.append(
                    (link.source_key, self._related_filter_keys(link, related_field, requested))
                )
            else:
                direct_where[field] = requested
        physical = list(
            dict.fromkeys(source for kind, source in sources.values() if kind == "physical")
        )
        for link in join_map.values():
            if link.source_key not in physical:
                physical.append(link.source_key)
        if not physical:
            physical.append(self.schemas[evidence_type].names[0])
        return _QueryPlan(
            evidence_type,
            select,
            sources,
            join_map,
            tuple(physical),
            direct_where,
            tuple(related_keys),
            limit,
        )

    def _project(self, plan, table):
        arrays = []
        for field in plan.select:
            kind, source = plan.sources[field]
            arrays.append(
                table[source]
                if kind == "physical"
                else self._joined_values(*source, table)
                if kind == "related"
                else pa.chunked_array(
                    [pa.array([source] * table.num_rows, type=pa.scalar(source).type)]
                )
            )
        return pa.Table.from_arrays(arrays, names=plan.select)

    def _result(self, plan, table):
        item = self.bindings[plan.evidence_type][0]
        return QueryResult(
            plan.evidence_type,
            item.grain,
            item.source_population,
            self._manifest_hash(),
            SCHEMA_VERSION,
            self._project(plan, table),
            tuple(vars(link) for link in plan.joins.values()),
        )

    def query(
        self,
        evidence_type: str,
        *,
        select: tuple[str, ...] | list[str] | None = None,
        where: dict | None = None,
        joins: tuple[str, ...] | list[str] = (),
        limit: int | None = 1000,
    ) -> QueryResult:
        """Select direct fields and explicitly named one-to-one relationships."""
        evidence_type = self._canonical_evidence_type(evidence_type)
        plan = self._prepare_query(evidence_type, select, where, joins, limit)
        table = self._scan(
            plan.evidence_type, plan.physical, plan.where, plan.limit, plan.related_keys
        )
        return self._result(plan, table)

    def iter_query(
        self,
        evidence_type: str,
        *,
        select: tuple[str, ...] | list[str] | None = None,
        where: dict | None = None,
        joins: tuple[str, ...] | list[str] = (),
        limit: int | None = None,
    ):
        """Stream bounded Arrow batches without materializing the source relation."""
        evidence_type = self._canonical_evidence_type(evidence_type)
        plan = self._prepare_query(evidence_type, select, where, joins, limit)
        for batch in self._scan_batches(
            plan.evidence_type, plan.physical, plan.where, plan.limit, plan.related_keys
        ):
            yield self._result(plan, pa.Table.from_batches([batch]))

    def materialize(self, recipe, destination):
        """Write a versioned training decision bundle from this verified store."""
        from .evidence_recipe import materialize_recipe

        return materialize_recipe(self, recipe, destination)

    def audit(self):
        """Reconcile declared baselines and field readiness with this package."""
        from .evidence_audit import audit_evidence_store

        return audit_evidence_store(self)
