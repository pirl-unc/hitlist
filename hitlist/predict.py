# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Retro-predictor: reassign class-only peptides to their best-scoring
HLA allele using MHCflurry (primary) or NetMHCpan (optional binary).

Class-only peptides are those annotated in IEDB with
``mhc_restriction == "HLA class I"`` (or "HLA class II") — the paper
authors knew the class but did not commit to a specific allele.  For
peptides from multi-allelic samples, we can often recover the likely
allele by running a binding predictor against the sample's experimental
MHC candidates. Independently reported cellular typing is preserved as
metadata; excluded background alleles do not enter the prediction.

Recovery needs a candidate space we actually have.  Two conditions gate it:
the row must belong to an identified sample rather than a study-wide pool
(#520, #563), and every class-I candidate that sample reported must name a
protein a backend can score (#574).  Where either fails the reassignment
abstains — no prediction and no best allele — because the best of a
knowingly incomplete candidate set is not the best.

The TLAKFSPYL example from our audit: IEDB class-only, sample contains
A*02:01 and A*24:02 (among others).  MHCflurry gives A*02:01 a rank
of 0.03 (strong binder) vs A*24:02 at 2.65 — the peptide is almost
certainly an A*02:01 ligand.

Usage::

    hitlist reassign-alleles -o reassigned.csv
    hitlist reassign-alleles --method netmhcpan --mhc-class I
"""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import NamedTuple

import numpy as np
import pandas as pd
from mhcgnomes import parse

from .curation import MHC_TYPING_COLUMNS

#: The context a prediction belongs to.
#:
#: ``mhc_basis`` stays in the key. #564's review proposed removing it, on the
#: grounds that the exporter rewrites it per row and could split one context in
#: two; under the ``sample_mhc_origin`` guard below it cannot, because the
#: exporter blanks ``mhc_basis`` exactly where ``mhc`` was empty before the pool
#: fill -- rows this function already excludes as ``class_pool`` or drops for
#: having no candidates. Dropping it from the key is not neutral either: two
#: rows of one context that genuinely disagree would be deduplicated to
#: whichever landed first, making the reported basis depend on row order.
_CONTEXT_COLUMNS = ["peptide", "pmid", "sample_label", "sample_mhc", *MHC_TYPING_COLUMNS]
_RESULT_COLUMNS = [
    *_CONTEXT_COLUMNS,
    "n_alleles_tested",
    "best_allele",
    "best_affinity_nM",
    "best_presentation_percentile",
    "is_strong_binder",
    "is_weak_binder",
]


class ClassIPredictionScope(NamedTuple):
    """How a reported class-I candidate list divides at the predictor boundary.

    ``scorable`` and ``unscorable`` partition the class-I candidates of one
    ``sample_mhc`` field.  Keeping the second half rather than dropping it is
    the whole point (#574): a best allele chosen from ``scorable`` alone is a
    claim about a candidate space that a non-empty ``unscorable`` shows we do
    not have, so :attr:`is_eligible` requires it to be empty.
    """

    scorable: tuple[str, ...] = ()
    unscorable: tuple[str, ...] = ()

    @property
    def is_eligible(self) -> bool:
        """True when there is something to score and nothing unaccounted for."""
        return bool(self.scorable) and not self.unscorable


def _names_one_class_i_protein(allele: str) -> bool:
    """True when ``allele`` identifies one HLA class-I protein a backend models.

    Asked of mhcgnomes rather than of the string: the repo's rule is to prefer
    the dependency's ontology over string shape.  Each clause rules out a
    designation that :attr:`~hitlist.curation.SampleMhcCandidates.exact`
    legitimately contains -- ``exact`` means *the source named this outright*,
    not *this is one protein sequence*:

    ``Allele``
        excludes a ``Gene``: ``HLA-A`` is a locus, and no backend scores a
        locus.
    ``num_allele_fields >= 2``
        excludes a one-field allele group.  ``HLA-B*27`` spans more than a
        hundred proteins whose motifs differ, so it names no sequence to score.
    ``is_human``
        this module wires the human class-I models only.  MHCflurry is handed
        the string for its human predictor, and :func:`_netmhcpan_allele_arg`
        merely strips ``*``, which spells a mouse allele ``H2-Kb`` where
        netMHCpan wants ``H-2-Kb``.  Supporting non-human backends needs a
        per-backend name mapping, not a looser filter here.
    ``not is_mutant``
        an engineered molecule is not the wild-type protein its name contains.
    ``not annotation_null`` / ``not is_pseudogene``
        a null or pseudogene product does not reach the surface, so it presents
        no peptide and no backend has a model for it.

    Finer than two fields stays scorable: the third and fourth fields refine
    the DNA sequence, not the protein, so a three-field name is better data
    rather than worse.
    """
    parsed = parse(allele, raise_on_error=False)
    return (
        type(parsed).__name__ == "Allele"
        and parsed.num_allele_fields >= 2
        and parsed.is_human
        and not parsed.is_mutant
        and not parsed.annotation_null
        and not parsed.is_pseudogene
    )


@lru_cache(maxsize=4096)
def class_i_prediction_scope(sample_mhc: str | None) -> ClassIPredictionScope:
    """Split a sample's reported class-I candidates by what a backend can score.

    The single boundary where reported candidates become predictor input, so the
    eligibility rule is stated once (#574).  Before this, the class and species
    filters were the only gate and ``HLA-A`` or ``HLA-A*02`` reached MHCflurry
    and NetMHCpan as if it named a protein.

    The rule is all-or-nothing by design.  Dropping the unresolved candidates
    and scoring the rest would answer a different question -- "which of the
    alleles we happen to know binds this best?" -- and return it with the
    confidence of the question actually asked.  Callers therefore abstain unless
    :attr:`~ClassIPredictionScope.is_eligible`, emitting no prediction and no
    best allele.  Source precision is preserved either way: nothing here
    rewrites a candidate to a coarser or finer designation.

    Candidates of another class are not in the partition at all.  A class-II
    allele, or a class-Ib molecule such as ``HLA-E``, is excluded upstream by
    :func:`hitlist.export.reported_class_alleles`; it is not an unscorable
    class-I candidate and does not hold back a class-I prediction.

    Parameters
    ----------
    sample_mhc
        An exported ``sample_mhc`` value, or ``None``.

    Returns
    -------
    ClassIPredictionScope
        Both halves of the partition, in the parser's sorted order.

    Raises
    ------
    ValueError
        The field carries a mutation label that cannot be assigned to a
        complete molecule.  Propagated from
        :func:`~hitlist.curation.sample_mhc_candidates` rather than caught: a
        mutant designation silently read as its wild type would fabricate a
        restriction, so this boundary refuses it loudly (#574).

    Examples
    --------
    >>> class_i_prediction_scope("HLA-A*02:01 HLA-B*07:02").is_eligible
    True
    >>> scope = class_i_prediction_scope("HLA-A*02:01 HLA-B*27")
    >>> scope.scorable, scope.unscorable, scope.is_eligible
    (('HLA-A*02:01',), ('HLA-B*27',), False)
    """
    from .export import reported_class_alleles

    scorable: list[str] = []
    unscorable: list[str] = []
    for allele in reported_class_alleles(sample_mhc or "", "I"):
        (scorable if _names_one_class_i_protein(allele) else unscorable).append(allele)
    return ClassIPredictionScope(tuple(scorable), tuple(unscorable))


def _predictable_class_i_alleles(sample_mhc: str | None) -> list[str]:
    """Class-I candidates to score for one sample, or none if it must abstain."""
    scope = class_i_prediction_scope(sample_mhc)
    return list(scope.scorable) if scope.is_eligible else []


def _predict_mhcflurry(pairs: pd.DataFrame) -> pd.DataFrame:
    """Run MHCflurry on a (peptide, allele) cross-product DataFrame.

    Returns per-row score: affinity_nM, presentation_percentile.  Keeps
    the caller's index.
    """
    try:
        from mhcflurry import Class1PresentationPredictor
    except ImportError as e:
        raise RuntimeError(
            "mhcflurry not installed.  Install with: pip install mhcflurry && "
            "mhcflurry-downloads fetch"
        ) from e

    predictor = Class1PresentationPredictor.load()
    pairs = pairs.reset_index(drop=True).copy()
    pairs["affinity_nM"] = np.nan
    pairs["presentation_percentile"] = np.nan

    # MHCflurry's affinity model only handles a fixed peptide-length range
    # (currently 5-15); anything outside it raises rather than returning a
    # score. One atypical-length MS hit shouldn't abort scoring the rest of
    # the batch -- leave it NaN, same as a predictor failure, rather than
    # crashing the whole query (#488).
    min_len, max_len = predictor.affinity_predictor.supported_peptide_lengths
    scorable = pairs["peptide"].str.len().between(min_len, max_len)
    if not scorable.any():
        return pairs

    # Class1PresentationPredictor.predict()'s `alleles` only accepts a flat
    # list of <=6 allele strings (one shared genotype tried against every
    # peptide) or a dict of sample_name -> alleles paired with
    # `sample_names` saying which peptide goes with which sample (#488).
    # Each row here wants its own specific allele, not a shared genotype, so
    # key the dict by allele (a one-allele "sample") and use each row's own
    # allele as its sample_names entry -- a flat list of one-element lists
    # looks like neither shape and silently gets treated as the first,
    # capping at 6 *rows* rather than 6 alleles per genotype.
    scored_rows = pairs[scorable]
    unique_alleles = scored_rows["allele"].unique()
    out = predictor.predict(
        peptides=scored_rows["peptide"].tolist(),
        alleles={a: [a] for a in unique_alleles},
        sample_names=scored_rows["allele"].tolist(),
        verbose=0,
    )
    # MHCflurry returns one row per (peptide, allele) pair in input order.
    out = out.reset_index(drop=True)
    pairs.loc[scorable, "affinity_nM"] = out["affinity"].values
    pairs.loc[scorable, "presentation_percentile"] = out["presentation_percentile"].values
    return pairs


def _netmhcpan_allele_arg(a: str) -> str:
    """NetMHCpan class-I allele string (strip the asterisk)."""
    return a.replace("*", "")


def _predict_netmhcpan(pairs: pd.DataFrame) -> pd.DataFrame:
    """Run NetMHCpan 4.x on a (peptide, allele) cross-product.

    Requires the ``netMHCpan`` binary on PATH (install separately from
    DTU; licensed for academic use).  Returns the same two score
    columns as MHCflurry so the best-allele logic is uniform.
    """
    from .curation import resolve_allele_identity

    # One netMHCpan invocation per allele is simplest; batch peptides.
    rows: list[dict] = []
    for allele, grp in pairs.groupby("allele"):
        peps = grp["peptide"].drop_duplicates().tolist()
        with TemporaryDirectory(prefix="hitlist_netmhcpan_") as directory:
            pep_file = Path(directory) / "peptides.txt"
            pep_file.write_text("\n".join(peps) + "\n")
            result = subprocess.run(
                ["netMHCpan", "-p", str(pep_file), "-a", _netmhcpan_allele_arg(allele), "-BA"],
                capture_output=True,
                text=True,
                timeout=600,
                check=True,
            )
        for line in result.stdout.splitlines():
            if "PEPLIST" not in line:
                continue
            parts = line.split()
            if len(parts) < 16:
                continue
            if resolve_allele_identity(parts[1]) != resolve_allele_identity(allele):
                raise RuntimeError(
                    f"NetMHCpan returned allele {parts[1]!r} for requested allele {allele!r}"
                )
            try:
                rows.append(
                    {
                        "peptide": parts[2],
                        # This invocation scores one requested allele. Keep
                        # its input spelling so scores join back to candidates.
                        "allele": allele,
                        "rank_EL": float(parts[12]),
                        "affinity_nM": float(parts[15]),
                    }
                )
            except (ValueError, IndexError):
                continue
    out = pd.DataFrame(rows, columns=["peptide", "allele", "rank_EL", "affinity_nM"])
    # Normalise to the same column names MHCflurry produces.  Use rank_EL
    # as the presentation proxy (lower = better, <= 0.5 = strong, <= 2 = weak).
    out["presentation_percentile"] = out["rank_EL"]
    return out[["peptide", "allele", "affinity_nM", "presentation_percentile"]]


def reassign_class_only_alleles(
    method: str = "mhcflurry",
    mhc_class: str = "I",
    max_alleles_per_sample: int = 6,
) -> pd.DataFrame:
    """Reassign class-only peptides to their best-scoring allele.

    Loads the observations table, restricts to rows where
    ``mhc_restriction`` is class-only ("HLA class I" / "HLA class II")
    and a named sample has reported experimental candidates that
    :func:`class_i_prediction_scope` finds fully resolved, runs the
    requested predictor against those candidates, and returns the
    best allele per peptide/sample context. A context whose candidates are
    not all scorable is omitted entirely rather than scored on the
    resolved subset (#574). Shared peptides retain a separate
    result for each PMID, sample label and typing context; predictions may be reused
    across samples, but the selected allele must belong to that sample.

    Parameters
    ----------
    method
        ``"mhcflurry"`` (pip-installable, default) or ``"netmhcpan"``
        (requires DTU binary on PATH).
    mhc_class
        ``"I"`` or ``"II"``.  MHCflurry currently supports class I only;
        class II support via NetMHCpan requires netMHCIIpan (not yet
        wired).
    max_alleles_per_sample
        Skip samples whose candidate set has more distinct alleles than this (likely
        a pooled-donor curation artifact; see
        tasks/per_sample_allele_curation_audit.md).  Such samples can
        produce misleading "best allele" calls because the pool does
        not represent any one donor. The conservative default is six for the
        classical human class-I genotype; the explicit limit remains configurable
        for separately justified experimental systems. Genotypes are never truncated.

    Returns
    -------
    pd.DataFrame, one row per distinct peptide/sample context, with columns:
        peptide, pmid, sample_label, sample_mhc, n_alleles_tested,
        best_allele, best_affinity_nM, best_presentation_percentile,
        is_strong_binder, is_weak_binder, plus the independent cellular-typing
        columns from the observation export. When no candidate has a finite rank,
        prediction fields are null and both binder flags are false.
    """
    if mhc_class != "I":
        raise NotImplementedError("Only class I reassignment is supported in v1.8.0.")

    from .export import generate_observations_table

    df = generate_observations_table(mhc_class=mhc_class)
    # A study-wide candidate union is not a biological sample, even when it
    # happens to fit under the allele-count limit (#520).
    class_only_mask = df["mhc_restriction"].fillna("").str.startswith("HLA class")
    multi_mask = df["is_monoallelic"].fillna(False).eq(False)
    # A named sample is not enough: the class-pool fallback fills ``sample_mhc``
    # with the study's class-wide union on rows it could not resolve to one
    # arm's own candidates, and such a row keeps its curated label, so the union
    # would be scored as that sample's genotype and the winning allele
    # attributed to a cell that may never have carried it.
    #
    # ``sample_match_type`` is the wrong way to ask. It reports whether the
    # study *has* a class pool, not whether this row took it, so it excluded
    # 14,532 class-only rows that carry their own arm's candidate list and zero
    # rows carrying a union -- all of the cost, none of the protection (#564).
    # ``sample_mhc_origin`` is the fact itself.
    identified = df["sample_label"].fillna("").ne("") & df["sample_mhc_origin"].fillna("").ne(
        "class_pool"
    )
    target = df[class_only_mask & multi_mask & identified].copy()
    # The experiment's candidates remain the prediction scope. Independently
    # reported cellular background alleles do not become peptide restrictions,
    # and cannot resolve a scope the experiment left unresolved (#563).
    #
    # An identified sample is necessary but not sufficient: the guards above
    # settle *whose* candidates these are, and this settles whether they name
    # proteins to score.  Six one-field allele groups pass every check up to
    # here and still describe no genotype a predictor can rank (#574).
    target["_alleles"] = target["sample_mhc"].map(_predictable_class_i_alleles)
    target = target[target["_alleles"].map(len).between(1, max_alleles_per_sample)]
    # Filter before the empty-input return; no predictor needs an empty batch.
    target = target[target["peptide"].str.len().between(8, 12)]
    if target.empty:
        return pd.DataFrame(columns=_RESULT_COLUMNS)

    target = target[[*_CONTEXT_COLUMNS, "_alleles"]].drop_duplicates(_CONTEXT_COLUMNS)
    target = target.reset_index(drop=True)
    target["_context_id"] = target.index
    candidates = target[["_context_id", "peptide", "_alleles"]].explode("_alleles")
    candidates = candidates.rename(columns={"_alleles": "allele"})
    # Score shared pairs once, retaining each context's own candidate membership.
    pair_df = candidates[["peptide", "allele"]].drop_duplicates().reset_index(drop=True)

    if method == "mhcflurry":
        scored = _predict_mhcflurry(pair_df)
    elif method == "netmhcpan":
        scored = _predict_netmhcpan(pair_df)
    else:
        raise ValueError(f"Unknown method: {method!r}")

    scored = scored[["peptide", "allele", "affinity_nM", "presentation_percentile"]].copy()
    scored["presentation_percentile"] = pd.to_numeric(
        scored["presentation_percentile"], errors="coerce"
    )
    scored = scored[np.isfinite(scored["presentation_percentile"])]
    ranked = candidates.merge(scored, on=["peptide", "allele"], validate="many_to_one")
    best = ranked.sort_values(["_context_id", "presentation_percentile", "allele"]).drop_duplicates(
        "_context_id"
    )
    best = best[["_context_id", "allele", "affinity_nM", "presentation_percentile"]].rename(
        columns={
            "allele": "best_allele",
            "affinity_nM": "best_affinity_nM",
            "presentation_percentile": "best_presentation_percentile",
        }
    )
    result = target.merge(best, on="_context_id", how="left", validate="one_to_one", sort=False)

    # Thresholds are the community conventions MHCflurry and NetMHCpan
    # both use: strong binder = rank/percentile <= 0.5, weak <= 2.0.
    result["is_strong_binder"] = result["best_presentation_percentile"] <= 0.5
    result["is_weak_binder"] = result["best_presentation_percentile"] <= 2.0
    result["n_alleles_tested"] = result["_alleles"].map(len)
    result = result.drop(columns=["_alleles"])

    return result[_RESULT_COLUMNS]
