# tractor architecture

The client sends one request containing text, field descriptions, optional locale,
and optional record anchor. Strict validation rejects malformed bodies, duplicate
JSON keys, invalid field definitions and unsupported target units.

The service accepts one active and at most one waiting request by default. A worker
thread owns the inference lock through actual model completion, including after a
client disconnect. A bounded cache stores only immutable, text-free schema plans.
Each request receives a private native schema object. The processor checks the
joint schema/text window before one encoder pass.

Without an anchor, fields are document-wide entity queries. With an anchor, native
record decoding binds optional fields to each anchor occurrence. Original offsets
identify repeated names and values. Multiple independent candidates produce an
ambiguity issue, not a highest-score guess.

The model selects source spans. The normalizer parses numbers with explicit locale
rules and resolves a fixed unit alias table. Pint computes with Decimal. simplejson
writes finite JSON numbers directly; the SDK reads fractional numbers as Decimal.
Neither the client nor normalizer imports the model stack.

Training uses the same field IDs and descriptions as inference. Synthetic facts,
family-level splits and independent arithmetic oracles are exported as native
GLiNER targets. The native trainer supplies optimization and checkpointing. Dev-only
threshold/checkpoint selection precedes separate synthetic-test and authored
regression evaluation. Bundles load from local files; preparation is the only
model-download action.

Server-Timing reports queue, preflight, native extraction and normalization durations.
Native extraction includes its own preprocessing and decoding. The preflight
currently repeats tokenization; timings make this overhead visible. Standard logs
contain no request source text. One process is used per GPU; restart the service
after an internal inference error marks it unready.
