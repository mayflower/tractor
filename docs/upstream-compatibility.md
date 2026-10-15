# Native GLiNER compatibility

GLiNER is pinned to `b86d35c02f2c1a4d9840ea50197bd1b6d13b031e`;
base model `fastino/gliner2.5-multi-v1` to
`cf5593a5d45e3bbf204b9df621b13b1c0cf25ee3`. Python package versions are in uv.lock.

Verified against installed source and real weights:

- AutoExtractor selects BoundaryExtractor; the checkpoint includes record heads,
  tokenizer and embedded encoder configuration and loads with HF offline flags.
- Document entities support descriptions and list candidates with Unicode offsets.
  Natural records use an explicit internal anchor and zero-or-more optional fields.
- Native formatting may return None for missing fields, and may filter an anchor
  after proposing a record. None becomes an empty candidate list; records with no
  surviving anchor are omitted. Malformed nonempty spans remain visible errors.
- Training records are repeated `json_structures` entries, each with explicit
  span dictionaries. A declared structure with empty string fields produces zero
  record targets, verified by the real collator. No fake gold anchor is inserted.
- Upstream randomly omits descriptions during training even when labels are opaque.
  The small DescribedProcessor adapter always includes supplied descriptions and
  disables removal of tasks/fields. It changes no architecture or optimizer logic.
- `transform_record` determines actual combined schema/text token count. Serving
  conservatively caps this at the minimum of checkpoint max_len (4096) and encoder
  max_position_embeddings (512). It performs no truncation or chunking. This bound
  is a service policy derived from the checkpoint, not a claim that relative
  attention can never process longer inputs.
- One `encode_tokens` call reaches the actual encoder. Native tests instrument it
  for 1/4/8 fields. Offline calibration reuses that sample's encoder states through
  the native hidden_states argument, but reruns native decoding for every threshold.
  Offline calibration also reuses the threshold-independent native core, boundary
  head and dense record tensors for that one example. Exact equivalence across
  all 27 combinations is checked in scripts/check-calibration.py.
  Serving and latency tests never cache model outputs or encoder states.
- Native checkpoints contain model weights, configuration, and tokenizer, but no
  complete optimizer/scheduler/RNG restore API. `load_checkpoint` is weights-only.
  An interrupted training run must restart under a new run directory.

Candidate proposal budgets can still miss true facts; confidence thresholds do not
prove completeness. The wrapper rejects detected record/field output saturation.
Source spans are validated before any quantity conversion. The authored fixtures
are open regression targets, not an independent real-world benchmark.

The native trainer's legacy structure_loss metric is zero for this boundary model;
actual supervised record losses are exposed as record_field_loss. The separate
native loss/gradient check reads these components and backpropagates the combined
output.total_loss (the nested losses['total_loss'] is the span-only subtotal).
