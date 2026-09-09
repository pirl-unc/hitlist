#!/usr/bin/env python3
"""One-shot migration: annotate every ``ms_samples`` arm with flat condition columns.

Reads the 167 distinct curated ``condition`` strings and writes the
categorical block each one supports onto every record that carries it, plus
a persistent ``condition_id`` per record.

Two properties make the output reviewable data rather than a derivation:

* It is written into ``pmid_overrides.yaml`` as ordinary keys and read back
  by the ordinary loader. Nothing re-runs this mapping at export time, so a
  later hand-correction to one record is not silently overwritten.
* Every record it writes is marked ``condition_evidence: curated_text``.
  That is the honest provenance: this normalizes wording somebody already
  curated, and inherits whatever that curation got right. Reading the paper
  is a different claim and gets ``primary_source`` plus a locator.

The table below annotates *only what its key states*. Where a string names a
fact no column captures at its stated precision the record is ``partial``;
where it describes alternatives the source does not separate it is ``mixed``
and the agent columns stay blank, because a union of alternatives in one row
reads as a combination treatment nobody performed.

Text insertion, not a YAML round-trip: ruamel is not a dependency and
PyYAML's dump would reflow all 8k lines and drop every comment. The keys go
in directly after each record's ``condition:`` line, located by node marks
from ``yaml.compose``.

Run once from the repo root; re-running is safe (records that already carry
``condition_id`` are left alone).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hitlist.conditions import CONDITION_COLUMNS

YAML_PATH = Path(__file__).resolve().parent.parent / "hitlist" / "data" / "pmid_overrides.yaml"


def A(**kwargs: str) -> dict[str, str]:
    """Build an annotation, defaulting to a fully annotated curated-text record."""
    ann = {"condition_status": "annotated", "condition_evidence": "curated_text"}
    ann.update({f"condition_{k}": v for k, v in kwargs.items()})
    return ann


def M(**kwargs: str) -> dict[str, str]:
    """A record whose source string describes alternatives it does not separate."""
    ann = A(**kwargs)
    ann["condition_status"] = "mixed"
    return ann


def P(**kwargs: str) -> dict[str, str]:
    """A record stating a fact no column captures at its stated precision."""
    ann = A(**kwargs)
    ann["condition_status"] = "partial"
    return ann


#: Nothing about an experimental condition has been annotated. Used for the
#: four arms whose ``condition`` records why the arm was never run at all —
#: those strings describe the deposit, not an experiment.
UNREPORTED = {"condition_status": "unreported"}

#: Curated condition string -> the categorical block it supports.
#:
#: Ordered as the corpus lists them so a reviewer can diff this against
#: ``sorted(set(conditions))``. The completeness check below fails if the two
#: ever disagree, which is what stops a new curated string from silently
#: landing with no annotation.
ANNOTATIONS: dict[str, dict[str, str]] = {
    "1 uM decitabine 72h": A(drugs="decitabine", combination="single"),
    "1 uM imatinib in culture": A(drugs="imatinib", combination="single"),
    "5-azacytidine treatment": A(drugs="azacitidine", combination="single"),
    # AAV delivers an allogeneic class-I allele alongside the endogenous
    # ones. Recorded as MHC context, not as an infection: a vector name is
    # not a pathogen.
    "AAV-transduced allogeneic MHC-I expression": A(
        transduction="MHC-I", mhc_context="mhc_coexpression", combination="single"
    ),
    "Alg8 neoantigen peptide pulse": A(antigen_exposure="peptide_pulse", combination="single"),
    "B2M CRISPR/Cas9 knockout": A(knockout_genes="B2M", combination="single"),
    # Restimulation with the organism, not infection of the presenting cell.
    # `partial`: the antigen's identity is not representable as a stimulus.
    "Borrelia burgdorferi short-term restimulation": P(
        stimulation="antigen_restimulation", combination="single"
    ),
    "CALR CRISPR/Cas9 knockout (calreticulin)": A(knockout_genes="CALR", combination="single"),
    "CANX CRISPR/Cas9 knockout (calnexin)": A(knockout_genes="CANX", combination="single"),
    "CDK4/6 inhibitor (palbociclib)": A(drugs="palbociclib", combination="single"),
    "CIITA transduction": A(transduction="CIITA", combination="single"),
    "CIITA transduction + T6BP/TAX1BP1 siRNA knockdown": A(
        transduction="CIITA", knockdown_genes="TAX1BP1", combination="simultaneous"
    ),
    "CIITA transduction only": A(transduction="CIITA", combination="single"),
    "CIITA transduction — HLA-II not naturally expressed in GBM": A(
        transduction="CIITA", combination="single"
    ),
    "CMV pp65 retroviral transduction": A(transduction="CMV pp65", combination="single"),
    "CyHV-2 viral infection": A(infection="Cyprinid herpesvirus 2", combination="single"),
    # ERAAP is the murine name for ERAP1; the original wording stays in
    # `condition`, and apm.py already treats the two as one gene.
    "ERAAP knockout": A(knockout_genes="ERAP1", combination="single"),
    "ERAP1 CRISPR/Cas9 knockout": A(knockout_genes="ERAP1", combination="single"),
    "ERAP1 CRISPR/Cas9 knockout (clone 3.36, 3 replicates)": A(
        knockout_genes="ERAP1", combination="single"
    ),
    "ERAP1 deletion": A(knockout_genes="ERAP1", combination="single"),
    "ERAP1 pharmacological inhibition": A(drugs="ERAP1_inhibitor", combination="single"),
    # A knockdown is not a knockout and must not be relabelled as one.
    "ERAP1 shRNA knockdown": A(knockdown_genes="ERAP1", combination="single"),
    "ERAP1 variant comparison": P(genetic_variants="ERAP1 unspecified", combination="single"),
    "ERAP1+ERAP2 double CRISPR/Cas9 knockout (3 replicates)": A(
        knockout_genes="ERAP1;ERAP2", combination="simultaneous"
    ),
    # "combinations" is a set of arms the string does not separate.
    # A union here would say both variants apply to every contributing arm,
    # which is what "combinations" declines to state.
    "ERAP1/ERAP2 variant combinations": M(),
    "ERAP2 CRISPR KO": A(knockout_genes="ERAP2", combination="single"),
    "ERAP2 CRISPR/Cas9 knockout (3 replicates)": A(knockout_genes="ERAP2", combination="single"),
    "ERAP2 KO comparison": A(knockout_genes="ERAP2", combination="single"),
    "ERAP2 overexpression": A(overexpression_genes="ERAP2", combination="single"),
    # "various combinations": no agent is established for every contributing
    # condition, so no agent column is filled. Listing all three would assert
    # a triple treatment the source explicitly does not claim.
    "EZH2i + decitabine + IFNg (various combinations)": M(),
    "Entinostat (MS-275, HDAC inhibitor)": A(drugs="entinostat", combination="single"),
    "GANAB CRISPR/Cas9 knockout (glucosidase II alpha)": A(
        knockout_genes="GANAB", combination="single"
    ),
    "GM-CSF (500 IU/mL) + IL-4 (250 IU/mL) 6d": A(cytokines="CSF2;IL4", combination="simultaneous"),
    "GM-CSF + IL-4 6d → LPS (60 EU/mL) + IFN-gamma (2000 IU/mL) 24h": A(
        cytokines="CSF2;IFNG;IL4", stimulation="LPS", combination="sequential"
    ),
    "Gemcitabine (alters proteasome composition)": A(drugs="gemcitabine", combination="single"),
    "HHV-6B infection": A(infection="HHV-6B", combination="single"),
    "HIV Env transduction": A(transduction="HIV Env", combination="single"),
    "HIV-1 infection": A(infection="HIV-1", combination="single"),
    "HLA-DM co-transfection — mono-allelic peptide editing": A(
        transfection="HLA-DM", mhc_context="monoallelic", combination="single"
    ),
    "HLA-DO knockout": A(knockout_genes="HLA-DO", combination="single"),
    # The source says only "IFN". Resolving that to IFNG would manufacture a
    # fact; `IFN` is the canonical coarse token and the record is `partial`.
    "IFN treatment": P(cytokines="IFN", combination="single"),
    "IFN-gamma": A(cytokines="IFNG", combination="single"),
    "IFN-gamma (20 ng/ml) + doxycycline (1 ug/ml), 51 h": A(
        cytokines="IFNG", drugs="doxycycline", combination="simultaneous"
    ),
    "IFN-gamma (20 ng/ml, 48 h) then dTAG-13 (1 uM, 3 h) degrader": A(
        cytokines="IFNG", drugs="dTAG-13", combination="sequential"
    ),
    "IFN-gamma (mouse, 20 ng/ml, 51 h)": A(cytokines="IFNG", combination="single"),
    "IFN-gamma (required to induce MHC-I expression)": A(cytokines="IFNG", combination="single"),
    "IFN-gamma + doxycycline (48 h) then dTAG-13 (1 uM, 3 h) degrader": A(
        cytokines="IFNG", drugs="dTAG-13;doxycycline", combination="sequential"
    ),
    "IFN-gamma 100 IU/mL 24h": A(cytokines="IFNG", combination="single"),
    "IFN-gamma 100 ng/ml 72h": A(cytokines="IFNG", combination="single"),
    "IFN-gamma 50 IU/mL 48h": A(cytokines="IFNG", combination="single"),
    "IFN-gamma 50 IU/mL 48h — HLA-II only expressed after IFNg": A(
        cytokines="IFNG", combination="single"
    ),
    "IFN-gamma treatment": A(cytokines="IFNG", combination="single"),
    "IL-2 expansion + OKT3 rapid expansion": A(
        cytokines="IL2", stimulation="OKT3", combination="simultaneous"
    ),
    "IL-4 (2 ng/mL) + CD40L-Tri (1 ug/mL) 48h": A(
        cytokines="CD40LG;IL4", combination="simultaneous"
    ),
    "IL-4 + GM-CSF 5d differentiation → IFN-gamma + LPS overnight maturation → "
    "synthetic peptide pulse (10 ug/mL, 2h)": A(
        cytokines="CSF2;IFNG;IL4",
        stimulation="LPS;dc_maturation",
        antigen_exposure="peptide_pulse",
        combination="sequential",
    ),
    "IRF2 CRISPR/Cas9 knockout": A(knockout_genes="IRF2", combination="single"),
    "Listeria monocytogenes infection": A(infection="Listeria monocytogenes", combination="single"),
    "MEK inhibitor (binimetinib, 100 nM, 72 h)": A(drugs="binimetinib", combination="single"),
    # `partial`: the HIVconsv immunogen the vector carries has no column.
    "MVA.HIVconsv (modified vaccinia Ankara) infection": P(
        infection="Vaccinia virus", combination="single"
    ),
    "Marek's disease virus infection": A(infection="Mareks disease virus", combination="single"),
    "Mycobacterium tuberculosis H37Rv infection": A(
        infection="Mycobacterium tuberculosis", combination="single"
    ),
    "Mycobacterium tuberculosis infection": A(
        infection="Mycobacterium tuberculosis", combination="single"
    ),
    "NOT in this study — TSAs from prior Laumont 2018 study": dict(UNREPORTED),
    "NOT profiled (no PRIDE files despite text implying exposure)": dict(UNREPORTED),
    "NOT profiled by MS": dict(UNREPORTED),
    "NOT profiled by MS — used for T cell isolation only": dict(UNREPORTED),
    "PDIA3 CRISPR/Cas9 knockout (ERp57)": A(knockout_genes="PDIA3", combination="single"),
    "PMA (10 ng/mL) + Ionomycin (1 ug/mL) 48h": A(
        stimulation="PMA;ionomycin", combination="simultaneous"
    ),
    "PMA differentiation → influenza A/H3N2/Wisconsin": A(
        stimulation="PMA", infection="Influenza A virus", combination="sequential"
    ),
    "PMA differentiation → influenza A/H3N2/X31": A(
        stimulation="PMA", infection="Influenza A virus", combination="sequential"
    ),
    "PROTAC BET degrader treatment": A(drugs="BET_degrader", combination="single"),
    "PRRSV infection": A(infection="PRRSV", combination="single"),
    "PRRSV infection (in vivo)": A(infection="PRRSV", material="in_vivo", combination="single"),
    "PromoCell DC medium 6d → fed UV-irradiated apoptotic Wisconsin-infected A549 → "
    "DC activation 4h": A(
        culture="PromoCell_DC",
        antigen_exposure="apoptotic_cell_feeding;cross_presentation",
        stimulation="dc_maturation",
        combination="sequential",
    ),
    "SARS-CoV-2 infection": A(infection="SARS-CoV-2", combination="single"),
    # A transfected spike segment is not an infection.
    "SARS-CoV-2 spike S1+S2 segment transfection": A(
        transfection="SARS-CoV-2 spike S1+S2", combination="single"
    ),
    "SILAC heavy DCs + light tumor cells → cross-presentation": A(
        labeling="SILAC", antigen_exposure="cross_presentation", combination="sequential"
    ),
    # Labeling is not an intervention, so no combination is claimed.
    "SILAC metabolic labeling": A(labeling="SILAC"),
    "SPPL3 CRISPR/Cas9 knockout": A(knockout_genes="SPPL3", combination="single"),
    "T. parva infection": A(infection="Theileria parva", combination="single"),
    # Intrinsic to the line (T2), not introduced by this study.
    "TAP deficiency": A(background="TAP_deficient"),
    "TAP1 CRISPR/Cas9 knockout": A(knockout_genes="TAP1", combination="single"),
    "TAP1 knockout": A(knockout_genes="TAP1", combination="single"),
    "TAP1 knockout + Mycobacterium tuberculosis H37Rv infection": A(
        knockout_genes="TAP1",
        infection="Mycobacterium tuberculosis",
        combination="simultaneous",
    ),
    "TAP2 CRISPR/Cas9 knockout": A(knockout_genes="TAP2", combination="single"),
    "TAPBP CRISPR/Cas9 knockout (tapasin)": A(knockout_genes="TAPBP", combination="single"),
    # "overexpression/mutation" is two arms the string does not separate, so
    # neither mechanism is established for every contributing condition.
    "TAPBPR overexpression/mutation": M(),
    "TIL expansion protocol": A(stimulation="til_expansion", combination="single"),
    "TNF-alpha + IFN-gamma": A(cytokines="IFNG;TNF", combination="simultaneous"),
    "TP53 R175H mutant transfection": A(
        transfection="TP53", genetic_variants="TP53 R175H", combination="single"
    ),
    "TP53 R273H mutant transfection": A(
        transfection="TP53", genetic_variants="TP53 R273H", combination="single"
    ),
    "Tap1 hepatocyte-specific KO + AAV H-2Kd stabilized": A(
        knockout_genes="TAP1", mhc_context="mhc_coexpression", combination="simultaneous"
    ),
    "Toxoplasma gondii infection": A(infection="Toxoplasma gondii", combination="single"),
    # A control for an infection arm; `partial` because this string does not
    # name the virus that was inactivated.
    "UV-inactivated virus control": P(control="mock"),
    "Vaccinia virus (MVA) infection": A(infection="Vaccinia virus", combination="single"),
    "biomaterial contact + LPS": A(
        stimulation="LPS;biomaterial_contact", combination="simultaneous"
    ),
    "canine distemper virus infection": A(infection="Canine distemper virus", combination="single"),
    "carbamazepine exposure": A(drugs="carbamazepine", combination="single"),
    "doxorubicin treatment": A(drugs="doxorubicin", combination="single"),
    "flucloxacillin treatment": A(drugs="flucloxacillin", combination="single"),
    "in vitro activation": A(stimulation="in_vitro_activation", combination="single"),
    "influenza A infection": A(infection="Influenza A virus", combination="single"),
    "influenza A/H3N2/Wisconsin 22h": A(infection="Influenza A virus", combination="single"),
    "influenza A/H3N2/X31 22h": A(infection="Influenza A virus", combination="single"),
    "influenza infection (PR8)": A(infection="Influenza A virus", combination="single"),
    "lenalidomide treatment in vitro": A(drugs="lenalidomide", combination="single"),
    # Provenance, not an experimental condition: nothing categorical to fill.
    "transplant context": P(),
    "tumor": P(),
    "unperturbed": A(control="untreated"),
    "unperturbed (3 biological replicates)": A(control="untreated"),
    # `unperturbed + X` is the curator convention for a treatment applied on
    # top of the baseline, so this is an IFN-gamma arm, not a control.
    "unperturbed + IFN-gamma (2000 U/mL, 3d)": A(cytokines="IFNG", combination="single"),
    "unperturbed — 48h transfection": A(transfection="unspecified", combination="single"),
    "unperturbed — CD19 sorted": A(control="untreated"),
    "unperturbed — DMEM": A(control="untreated", culture="DMEM"),
    "unperturbed — DMEM + 10% FBS": A(control="untreated", culture="DMEM"),
    "unperturbed — DMSO vehicle control": A(control="vehicle", drugs="DMSO"),
    # A co-transfection is an intervention. `simplify_condition` blanks
    # everything after `unperturbed — `, so these 42 arms and the 4 that
    # explicitly lack HLA-DM are one `condition_category` today.
    "unperturbed — HLA-DM co-transfected": A(transfection="HLA-DM", combination="single"),
    "unperturbed — Hap10 (low-activity) ERAP1 background": A(
        control="untreated", background="ERAP1_hap10"
    ),
    "unperturbed — IMDM + 10% FBS": A(control="untreated", culture="IMDM"),
    "unperturbed — L243 (anti-HLA-DR) IP": A(control="untreated"),
    "unperturbed — LCL endogenous": A(control="untreated"),
    "unperturbed — PDX tumor": A(control="untreated", material="in_vivo"),
    "unperturbed — RPMI": A(control="untreated", culture="RPMI-1640"),
    "unperturbed — RPMI + 10% FBS": A(control="untreated", culture="RPMI-1640"),
    "unperturbed — RPMI + GlutaMAX + 10% FBS": A(control="untreated", culture="RPMI-1640"),
    "unperturbed — RPMI-1640": A(control="untreated", culture="RPMI-1640"),
    "unperturbed — baseline": A(control="untreated"),
    "unperturbed — bronchoalveolar lavage": A(control="untreated", material="biofluid"),
    "unperturbed — comparison only": A(control="untreated"),
    "unperturbed — congenic mice": A(control="untreated", background="congenic"),
    "unperturbed — constitutively MHC-I positive": A(
        control="untreated", background="mhc_i_constitutive"
    ),
    "unperturbed — direct ex vivo": A(control="untreated", material="direct_ex_vivo"),
    "unperturbed — empty vector": A(control="empty_vector"),
    # "endogenous low expression" does not say low expression of what.
    "unperturbed — endogenous low expression": P(
        control="untreated", background="endogenous_low_expression"
    ),
    "unperturbed — ex vivo": A(control="untreated", material="direct_ex_vivo"),
    "unperturbed — fresh tumor biopsies": A(control="untreated", material="fresh"),
    # Corrected by the #450 primary-source pilot: the paper says snap frozen at
    # -80 C, and says nothing about these patients' prior therapy.
    "unperturbed — snap-frozen tumor punch biopsies": A(material="frozen"),
    "unperturbed — in vivo": A(control="untreated", material="in_vivo"),
    "unperturbed — matched non-malignant tissue from cancer patients": A(control="untreated"),
    "unperturbed — mock infection control": A(control="mock"),
    "unperturbed — mono-allelic": A(control="untreated", mhc_context="monoallelic"),
    "unperturbed — mono-allelic (DLA in human host)": A(
        control="untreated", mhc_context="mhc_transfectant;monoallelic"
    ),
    "unperturbed — mono-allelic expression system": A(
        control="untreated", mhc_context="monoallelic"
    ),
    "unperturbed — mono-allelic phosphopeptidome (IMAC + TiO2 enrichment)": A(
        control="untreated", mhc_context="monoallelic"
    ),
    "unperturbed — mono-allelic soluble HLA": A(
        control="untreated", mhc_context="monoallelic;soluble_mhc"
    ),
    "unperturbed — mono-allelic transfectant": A(
        control="untreated", mhc_context="mhc_transfectant;monoallelic"
    ),
    "unperturbed — mono-allelic transfectant (protective)": A(
        control="untreated", mhc_context="mhc_transfectant;monoallelic"
    ),
    # Explicit absence, which is a positive claim and the reason `none`
    # exists. These are the comparator for the 42 co-transfected arms.
    "unperturbed — mono-allelic; no HLA-DM co-transfection": A(
        control="untreated", mhc_context="monoallelic", transfection="none"
    ),
    "unperturbed — multiple sclerosis lesion": A(control="untreated"),
    "unperturbed — natural ERAP2 genotype": A(
        control="untreated", background="ERAP2_natural_genotype"
    ),
    "unperturbed — pooled mono-allelic": A(control="untreated", mhc_context="monoallelic"),
    "unperturbed — sHLA from plasma": A(
        control="untreated", mhc_context="soluble_mhc", material="biofluid"
    ),
    "unperturbed — sHLA from pleural fluid": A(
        control="untreated", mhc_context="soluble_mhc", material="biofluid"
    ),
    "unperturbed — sHLA from serum": A(
        control="untreated", mhc_context="soluble_mhc", material="biofluid"
    ),
    "unperturbed — sHLA immunoaffinity purification": A(
        control="untreated", mhc_context="soluble_mhc"
    ),
    "unperturbed — self-peptidome": A(control="untreated"),
    "unperturbed — snap-frozen": A(control="untreated", material="frozen"),
    "unperturbed — snap-frozen surgical resection": A(control="untreated", material="frozen"),
    "unperturbed — soluble HLA (parental allele comparison)": A(
        control="untreated", mhc_context="soluble_mhc"
    ),
    "unperturbed — soluble HLA refolding": A(control="untreated", mhc_context="refolded_mhc"),
    "unperturbed — soluble HLA secreted into supernatant": A(
        control="untreated", mhc_context="soluble_mhc"
    ),
    # "some patients" is exactly the case where a control claim would be
    # fabricated: the record mixes treated and untreated donors, so neither
    # `untreated` nor the drug is true of every contributing condition.
    "unperturbed — some patients post-ipilimumab": M(),
    "unperturbed — standard culture": A(control="untreated", culture="standard_culture"),
    "unperturbed — standard culture (IMDM + 10% FBS)": A(control="untreated", culture="IMDM"),
    "unperturbed — tal1b5 (anti-HLA-DR) IP": A(control="untreated"),
    "vaccinia (VACV) infection": A(infection="Vaccinia virus", combination="single"),
    "vaccinia virus infection": A(infection="Vaccinia virus", combination="single"),
}


def slugify(label: str, taken: set[str]) -> str:
    """A stable, readable ``condition_id`` from a sample label.

    Labels are unique within every study in the corpus, so slugging one
    gives a unique and *meaningful* key — ``c1r_erap2_ko`` rather than a row
    number. Derived once, here, and written to the YAML; nothing recomputes
    it, so renaming the label later cannot move an observation to a
    different arm.
    """
    # The substitution leaves only [a-z0-9_] and the strip removes leading
    # underscores, so the result already starts with [a-z0-9] or is empty.
    slug = re.sub(r"[^a-z0-9]+", "_", label.casefold()).strip("_")[:48].strip("_") or "arm"
    candidate, n = slug, 1
    while candidate in taken:
        n += 1
        candidate = f"{slug}_{n}"
    taken.add(candidate)
    return candidate


def main() -> int:
    src = YAML_PATH.read_text()
    lines = src.splitlines(keepends=True)
    root = yaml.compose(src)

    conditions_seen: set[str] = set()
    # line index (0-based, insert *before*) -> block of new lines
    insertions: dict[int, list[str]] = {}
    n_records = n_annotated = 0

    for study in root.value:
        keys = {k.value: v for k, v in study.value}
        if "ms_samples" not in keys:
            continue
        pmid = keys.get("pmid").value if "pmid" in keys else keys["submission_id"].value
        taken: set[str] = set()
        for sample in keys["ms_samples"].value:
            n_records += 1
            fields = [(k.value, v) for k, v in sample.value]
            names = [k for k, _ in fields]
            condition = next(v.value for k, v in fields if k == "condition")
            if "condition_id" in names:
                continue  # already migrated; a later hand-correction stands
            conditions_seen.add(condition)

            ann = dict(ANNOTATIONS.get(condition, UNREPORTED))
            label = next((v.value for k, v in fields if k == "sample_label"), f"pmid_{pmid}")
            block = {"condition_id": slugify(label, taken), **ann}

            body = "".join(
                f'        {col}: "{block[col]}"\n' for col in CONDITION_COLUMNS if block.get(col)
            )
            # Insert after the `condition:` line, so the categorical block
            # reads directly under the wording it categorizes.
            idx = names.index("condition")
            if idx + 1 < len(fields):
                # The *key* node's line, not the value's: a block sequence
                # value starts at its first item, so inserting there would
                # land between `reference_proteomes:` and its own list.
                after = sample.value[idx + 1][0].start_mark.line
            else:
                after = max(v.end_mark.line for _, v in fields)
            insertions[after] = [*insertions.get(after, []), body]
            n_annotated += 1

    missing = conditions_seen - set(ANNOTATIONS)
    if missing:
        print(f"ERROR: {len(missing)} curated condition string(s) have no annotation:")
        for c in sorted(missing):
            print(f"  {c!r}")
        return 1
    unused = set(ANNOTATIONS) - conditions_seen
    if conditions_seen and unused:
        print(f"ERROR: {len(unused)} annotation(s) match no curated string (stale/typo):")
        for c in sorted(unused):
            print(f"  {c!r}")
        return 1

    out: list[str] = []
    for i, line in enumerate(lines):
        for block in insertions.get(i, []):
            out.append(block)
        out.append(line)
    for block in insertions.get(len(lines), []):
        out.append(block)
    YAML_PATH.write_text("".join(out))

    print(
        f"{n_records} ms_samples records, {len(conditions_seen)} distinct condition strings, "
        f"{n_annotated} annotated."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
