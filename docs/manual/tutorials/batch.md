# Tutorial: build a batch import

You will turn a JSONL feed of short supplier descriptions into JSONL extraction
results. Start the local API first. The complete program is
[batch.py](../../../examples/manual/batch.py).

## 1. Inspect the input

[requests.jsonl](../../../examples/manual/requests.jsonl) contains one object per line:

```jsonl
{"id":"item-1","text":"Breite: 12,5 cm.","locale":"de-DE"}
{"id":"item-2","text":"Modell CM125 hat keine Maßangaben."}
```

An ID belongs to your import system; it is carried around the extraction call and
is not an extra API request property. The batch program always extracts `width_mm`.

## 2. Run the import

```sh
uv run --no-sync python examples/manual/batch.py   examples/manual/requests.jsonl /tmp/tractor-results.jsonl
cat /tmp/tractor-results.jsonl
```

Choose a new output filename when rerunning: the program refuses to overwrite an
existing file. It opens one reusable SDK client and makes requests sequentially,
which fits a single server's bounded admission policy.

The first item has `status="extracted"`, `width_mm=125`, and no issues. The second
has `status="needs_review"`, a null width, and a `not_found` issue. A failed HTTP
call or malformed input line produces `status="error"`. The script preserves the
original ID and writes one output per nonblank input line, so a bad item does not
silently disappear.

## 3. Add a review step

Read each result and branch on `status`. `extracted` means the API reported no
issues; it is not a guarantee that the model chose the correct source span. Apply
your own plausible-range checks and acceptance policy before a database write.
Keep the import ID as a unique key to make retries of the import idempotent.

For CSV input, first map each row to an object with `id`, `text`, and optional
`locale`. Keep the original row ID. Serialize with `tractor.jsonio.dumps` so Decimal
values remain JSON numbers. For CSV output, write a Decimal using `str(value)` and
choose a separate representation for null; never coerce it to `0`.

This program provides per-item error handling, not job scheduling or resumable
checkpoints. A production queue should persist completion by ID, retry only
appropriate transport failures, and keep review items separate from retry items.
See [errors and retries](../how-to/errors-and-retries.md).

[Manual home](../index.md)
