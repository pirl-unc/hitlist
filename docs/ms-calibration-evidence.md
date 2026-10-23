# Known-load non-MHC MS response evidence

PXD004886 supplies a nominal synthetic-standard panel and reported instrument
responses in a human HEK293 digest background. It does not measure APC processing
or intracellular peptide concentration. The source adapter preserves the original
Supplementary Data 1/2/4/8/13/14/15 and article XML, including known source conflicts.

The installed package contains source receipts and parsing code. Original assets
are caller-supplied and hash checked; they are not bundled with Hitlist. The
article declares CC-BY4.0 with its third-party-credit exceptions. Source attribution
and archive/publisher equivalence receipts are retained.

## Shared generation and typed queries

Build a fresh successor from a verified reference generation, including every
registered source required in the resulting store:

```sh
hitlist evidence append-sources /path/to/reference-generation /path/to/new-generation \
  --source-root pxd009092=/path/to/mapp-assets \
  --source-root pxd004886=/path/to/calibration-assets
```

The reference generation and original MAPP generation remain immutable. This
command starts from the schema-7 reference package; it does not recursively append
to an existing schema-8 successor. Including both sources retains the MAPP
relations alongside calibration. Compare original source bytes when qualifying a
new generation. Older catalog-10 successors remain readable without advertising
calibration tables that they do not contain.

```python
from hitlist import EvidenceStore

store = EvidenceStore.open("/path/to/new-generation")
result = store.query(
    "calibration_s4_measurement",
    select=[
        "measurement_id", "representation", "quantitative_value",
        "measurement_unit", "value_status",
        "nominal_load.quantitative_value", "nominal_load.measurement_unit",
    ],
    joins=["nominal_load"],
    limit=None,
)
```

The two quantity columns have different meanings: reported response is arbitrary
fragment-ion peak area; the joined load is nominal fmol of the panel standard on
column. It is not certified molarity of a specific reporter form. This query
exposes source evidence; it is not a scientifically admitted training recipe.

All thirteen calibration relations are available through the same catalog/query
interface. Every native field remains addressable as `native.<column>`. Logical
aliases are limited to fields with matching meaning at the same grain. Named
relationships connect measurements to reporters, original column slots, nominal
loads and source cells. The reverse standard-to-forms relationship is explicitly
many-valued and cannot silently multiply measurements.

## Source distinctions

- `calibration_source_sheet` and `calibration_source_cell` preserve original
  coordinates, types, formulas, caches, merged headers and publication versus
  workbook labels. Data8 calls itself Data9 internally; both labels survive.
- `calibration_standard` describes 30 nominal sequences, with response-informed
  dilution groups. `calibration_nominal_load` stores 25 group/sample design
  amounts. The source heading says concentration, but its unit is fmol on column.
- `calibration_reporter_form` retains 31 labels. CNNVLYIR has separate CAM and PCm
  reporter forms; isotope and aggregation annotations remain explicit. Panel
  membership does not establish each form's share of nominal molar load.
- `calibration_source_slot` preserves 99 structural columns under 11 site
  headers. Local column order does not supply file, day, replicate or specimen
  identity.
- `calibration_s4_measurement` contains both raw and normalized representations
  of all planned S4 slots. Absent cells, explicit empty strings and positive
  values remain different. Missing values are not zeros or confirmed negatives.
- `calibration_summary_measurement` retains published means, standard deviations
  and CVs, including empty states and source-matrix replay diagnostics.
- `calibration_curve` and `calibration_curve_measurement` preserve the actual RSQ
  argument ranges and plotted amount labels. Nominal sample links remain
  unresolved; source grid values must not silently replace design amounts.
- `calibration_normalization_diagnostic` preserves normalized-only positives and
  nonuniform raw/normalized ratios. Derived diagnostic ratios are not supplied
  normalization coefficients or instructions to correct source measurements.
- `calibration_protocol_parameter` preserves site-specific LC/MS systems, columns,
  stage-specific gradients and the common64-window acquisition settings. Source
  “As above/below” directives remain explicit; pilot-stage gradients are not silently
  applied to main-study rows. The article’s2µL injection volume is a reference-protocol
  assertion, distinct from unavailable individual-run receipts.
- `calibration_source_assertion` records scoped protocol, source selection,
  unresolved and inapplicable context. Background-cell or enzyme species does
  not become APC species. Standards added after digestion do not label digestion
  yield.

The source compiler replays its typed records against the original assets during
verification. That establishes source conservation, not calibrated prediction.
Lot, purity, assigned-amount uncertainty, molecular-form fractions, acquisition
identity and detailed censor/saturation mapping remain unknown or unresolved.
Training admission requires a separate canonical recipe with explicit eligibility
and retained exclusions. Full curve calibration and transfer to MAPP or MHC
elution require additional evidence.
