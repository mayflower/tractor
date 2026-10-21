# Understanding accuracy and performance

Tractor is evaluated as a complete extraction and normalization pipeline. Final
typed values, record assignment and issue codes determine correctness. Raw spans
are diagnostic evidence, not the product score.

The current pipeline is evaluated on 4,429 Dev documents, with CORD excluded. Synthetic, real/human
and real/deterministic evidence is reported separately. Field F1 scores individual
fields; complete-record F1 requires the whole record to be correct. On synthetic
Dev, these are 90.30% and 57.61% respectively for repeated records. A high field
score does not imply that a complete record is safe to accept automatically.
See the [current report](../../benchmarks.md) for all measured domains.

Equivalent source-supported answers are accepted through reviewed offline
annotations after normalization. Incomplete references leave unresolved answers
unscored; they are neither successes nor confirmed errors. Changing reference
coverage can change the denominator of exact-request scores. Report coverage
alongside scores, and do not compare changing denominators as if they were fixed.

Dev selection does not establish independent test accuracy. Calibration and test
are separate, held-out splits; the current thresholds remain provisional. The
owner selected this model for broader generalist behavior while accepting known
regressions. The strict quality checks remain failed and are published as such.
Missing real evidence cannot be replaced by high synthetic scores.

No latency or Haiku/Luna comparison has been measured for the current weights.
To assess performance, measure the deployment and schema distribution you intend
to use. Separate startup from requests and report hardware, concurrency, failures,
median and tail latency. A specialized model plus normalization has a different
execution path from prompted LLM generation; numerical advantages require a
matched benchmark of the actual released bundle.

[Manual home](../index.md)
