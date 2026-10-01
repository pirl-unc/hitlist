"""Temporary inventory of cross-species literature-review candidates."""

import json
from pathlib import Path

from hitlist.curation import load_pmid_overrides
from hitlist.export import generate_ms_samples_table
from hitlist.observations import load_observations

output = Path("species-context-inventory")
output.mkdir(exist_ok=True)
columns = [
    "pmid", "source_species", "host_organism", "mhc_species",
    "is_chimeric", "is_engineered_mhc", "xenograft",
]
observations = load_observations(columns=columns)
flagged = observations.loc[
    observations["is_chimeric"] | observations["is_engineered_mhc"] | observations["xenograft"]
]
groups = flagged.groupby(columns, observed=True, dropna=False).size().reset_index(name="n_rows")
groups = groups.sort_values("n_rows", ascending=False)
groups.to_json(output / "flagged_observations.json", orient="records", indent=2)
samples = generate_ms_samples_table()
text_columns = ["sample_label", "condition", "source", "note", "study_label"]
text = samples[text_columns].fillna("").astype(str).agg(" ".join, axis=1)
candidates = samples.loc[
    samples["species_axes_agreement"].eq("false")
    | text.str.contains(r"xenograft|xenogeneic|chimeric|transgenic", case=False, regex=True)
]
candidates.to_json(output / "candidate_samples.json", orient="records", indent=2)
pmids = set(int(value) for value in groups["pmid"].dropna())
pmids.update(int(value) for value in candidates["pmid"].dropna())
overrides = load_pmid_overrides()
entries = {str(pmid): overrides.get(pmid, {}) for pmid in sorted(pmids)}
(output / "candidate_studies.json").write_text(json.dumps(entries, indent=2) + "\n")
summary = {
    "n_observation_rows": len(observations),
    "n_flagged_rows": len(flagged),
    "n_flagged_pmids": int(groups["pmid"].nunique()),
    "n_candidate_samples": len(candidates),
    "n_candidate_pmids": len(pmids),
}
(output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
for pmid in sorted(pmids):
    study = overrides.get(pmid, {})
    n_rows = int(groups.loc[groups["pmid"].eq(pmid), "n_rows"].sum())
    print(pmid, n_rows, study.get("study_label", "UNCURATED"))
