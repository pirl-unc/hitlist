While admitting the original HAP1 quantitative source for #726, an optional context
join fails for rows with unresolved acquisition context. The source rows must stay
queryable with null metadata; dropping those rows or inventing context is incorrect.

On main e8faa7f, `EvidenceStore._joined_values` sends `(None,)` to a string target
column's Arrow `isin` predicate. The result is:

```
pyarrow.lib.ArrowTypeError: Array type doesn't match type of values set: string vs null
```

The HAP1 package retains 40,529 original precursor rows across two unmapped runs.
Querying their optional context therefore exposes this shared API defect.

Acceptance:

- All-null optional foreign keys return the same number of typed null outputs.
- Mixed-null joins resolve known keys while retaining null rows.
- Null keys never match null target identities.
- Required-one joins still reject missing source/target identities.
- Add regression tests and validate against the original HAP1 package; preserve
  its unchanged source quantities and unresolved-context status.
