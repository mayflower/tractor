# Model evaluation

The measurements below apply to the distributed tractor weights, identified by
SHA-256 in [metrics.json](../release/metrics.json). Exact responses require all
fields, records and issue codes to match. Field F1 measures field extraction.

| Dataset | Requests | Exact responses | Field F1 |
|---|---:|---:|---:|
| Synthetic acceptance | 2,048 | 1,898 (92.68%) | 0.9876 |
| Synthetic regression | 20,000 | 19,610 (98.05%) | 0.9974 |
| Authored API regressions | 32 | 32 (100%) | 1.0000 |

The acceptance set covers product names, incomplete records, wide schemas,
competing values, repeated equal values and general quantities. Its source texts
are disjoint from training and dev; some template families are shared with other
evaluation sets. The regression suite has 18,000 unique source texts: some requests
share text but vary schemas or target units. Authored fixtures are known regression
targets, not an independent holdout. Model selection used dev data only.

The datasets differ in difficulty and must not be compared as interchangeable
accuracy estimates. All evaluations are synthetic or authored. Real-world accuracy,
OCR, unrestricted tables and other languages are unmeasured. The acceptance set
still has 150 incorrect complete responses, chiefly record and wide-schema cases.

The serving architecture uses one encoder pass followed by deterministic
normalization. No validated latency measurement is available for this bundle.
Run `tractor bench --help` to measure your own deployment on an otherwise idle device.
