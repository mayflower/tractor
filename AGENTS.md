# tractor

tractor is the System One Extraction Model and its local HTTP API/Python SDK.
The sole release bundle is models/tractor. Keep publication files focused on the
current product; do not add experiment histories, predecessor models or local paths.
Both release targets must remain private: mayflower/tractor and mayflowergmbh/tractor.

The authoritative scope and implementation requirements are in reqs.md.

## Binding product and evaluation contract

- There is exactly ONE public extraction API: POST /v1/extract and Client.extract.
  Every field has a mandatory discriminated type that determines processing.
  Type parameters (unit, currency, date order, etc.) belong to the field itself.
  Do not introduce parallel document/task endpoints or optional processing maps.
  Keep the public response to data and issues. Do not add include_details,
  source-evidence, score or offset options; evidence is internal diagnostics only.
  Contact types select their normalizers automatically. Names preserve spelling
  and qualifiers; phone/postal context must be explicit, never guessed.
  HTTP, SDK, checkpoint selection and evaluation must use the same typed request
  and production service. Internal corpus/task representations are adapters only.
  Legacy corpus fields without a processing contract are explicit text migration
  debt, not evidence that semantic normalization is complete.
  Compile native training through the public API schema adapter too: identical
  anonymous field IDs, descriptions, cardinalities, record schema and per-task
  request boundaries. Preserved legacy task-named native files are not API-aligned
  training. Match provisional entity/record thresholds across baseline and Dev
  checkpoint selection; never silently compare different inference policies.
  Evaluate both final data and expected field/record issue codes. Correct rejection
  of invalid or ambiguous evidence is a valid outcome; not_found is not a substitute
  for recognizing invalid present evidence. Expected issues are offline answer
  annotations only. Report rejection accuracy separately from positive extraction
  F1, and preserve Decimal/type-aware comparison in diagnostics and saved reports.

tractor is an extraction AND programmatic normalization pipeline. The product is
not the raw GLiNER model. The user cares about the final API/SDK output after
extraction, deterministic parsing, validation, normalization and ambiguity handling.
Do not forget or reverse this contract when expanding domains or diagnosing errors.

- Select checkpoints, compare against LLMs and apply release gates using the FULL
  production pipeline and its final field/record values. Raw span/model scores are
  diagnostics only; never use them alone as the product score or selection gate.
- Evaluation must execute the SAME processing code as the API/SDK. Do not create
  benchmark-only repairs or use gold labels to guide parsing, field recovery,
  grouping, normalization or candidate selection. Define expected final outputs
  independently; keep original spans available as evidence, separately scored.
- Extraction identifies relevant explicitly stated source values and their roles.
  Deterministic code parses and normalizes them where the supported contract allows
  it (already quantities/units; address components and other domains need explicit
  implementations). Missing normalizers are pipeline gaps, not automatically model
  failures or optional additions outside the product objective.
- Do not demand that the model perform a reliably programmable parsing step solely
  to satisfy a raw-span benchmark. Training targets must support the full pipeline.
  Conversely, do not claim an unimplemented parser or normalizer already exists.
- Preserve source evidence, Decimal arithmetic, explicit locale/country constraints,
  record boundaries and ambiguity handling. Never invent missing facts, infer
  postal codes from cities, geocode silently, or normalize onto external taxonomies.
- Repeated mentions of the same value are evidence occurrences, not automatically
  distinct business values. Score final results according to their declared scalar,
  unique-list or record contract; do not require every duplicate mention unless the
  user explicitly requests occurrence extraction. Preserve distinct records.
- Deterministic parsing of explicitly stated values does NOT authorize table/OCR
  layout reconstruction, classification, QA, generic relation extraction or reasoning.
- Require >70% for each declared domain/output form on pipeline results; report
  missing real evidence as missing. No claim of general reliability from raw model
  scores, synthetic-only evidence, or a changed evaluation contract alone.

Extract only explicitly stated values into specified fields, repeated records or
typed entity lists. Exclude ALL classification, tagging, routing, sentiment,
urgency, QA, relation extraction, summaries, code normalization and coding-agent
helper tasks. No inference of unstated facts or reasoning criteria.
Only explicitly described records in readable text are in scope. Table parsing,
column/row reconstruction and layout-dependent invoice positions are excluded
from training and release gates. Generic relation triplets are also excluded.
Synthetic training inputs must be realistic authored document texts, not fixed
field-list templates with substituted values. Deterministic facts may establish
gold internally; independent realism/semantic review must precede scaling.
Current active corpus paths and build status are recorded in reports/extraction-build.
Typed corpus paths and the frozen migration policy are recorded in
reports/extraction-build/typed-corpus.json. Use those views for the next pipeline
evaluation/training; retain explicit contract/annotation gaps. Migration never
licenses guessing countries, currencies, target units or mixed semantic types.
Old generalist runs contain excluded supervision and are not clean-scope evidence.
Require >70% per declared domain and output form; missing real evaluation cannot
be compensated by synthetic scores. Product >=90% remains a secondary regression
floor, never a reason to restore product specialization. Only full-pipeline Dev evaluation selects models;
calibration/test are held out. Report real, teacher and synthetic evidence separately.

Prefer established Python libraries and standard-library methods where they fit.
Reuse Pint for units, Babel for locale data/numbers, Decimal for arithmetic and
stdlib datetime for calendar validation. Keep explicit product constraints around
library calls: no guessed date order, invented components or silent ambiguity.
Keep annotation conversion independent of the production normalizer.
Use libpostal through the postal Python bindings for address component parsing,
and google-i18n-address for country/postal rules. Do not reintroduce hand-written
English street-suffix recognizers. Keep the native library and data pinned in Nix
and Docker; parsing is local, statistical, and never supplies unstated facts.
Use dateparser for complete named DE/EN dates with strict absolute parsing and
source-component checks; its defaults can still discard qualifiers or infer years.
Use Babel/CLDR for supported currency names and symbols, and isoduration for exact
Decimal ISO duration components. Keep explicit date order, currency, precision and
ambiguity rules around library calls. Standard identifiers use python-stdnum only
when Identifier.format explicitly requests isbn/ean/vat; VAT requires country.

Use nix develop, just setup and uv run --no-sync. just check runs lint, fixtures
and unit tests. just check-model models/tractor cpu checks real HTTP and encoder
behavior. Keep SDK imports independent of ML libraries and downloads explicit.
Preserve Decimal arithmetic, original offsets, ambiguity handling and bounded
inference admission. Do not infer model quality from fixture-only tests.

Only dev data may select checkpoints or recipes. Final score/threshold calibration
uses a separate calibration split after model selection; never rank checkpoints
on calibration data. Provisional dev threshold experiments are not final calibration.
Report synthetic and real-world
results separately. Do not run GPU latency measurements without a new user request;
the GPU is shared. Published latency measurements are CPU HTTP round trips only.

For local training commands and current supported scope, see
[the training runbook](docs/manual/how-to/training.md). External Teacher usage has
a 500 USD total ceiling in configs/generalist-budget.json; preserve its durable
reservation ledger across retries/runs. Do not enable cloud compute or GPU latency
benchmarks as part of a training run.

Create distributable snapshots with scripts/package-release.py. Its allowlists
exclude Git history, logs, datasets, training runs and private provenance. Do not
upload files or change repository visibility as part of packaging.
