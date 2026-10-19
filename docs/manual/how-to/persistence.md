# How to store extracted data in a database

Run the [batch tutorial](../tutorials/batch.md) to produce JSONL results. Then use
the complete [SQLite example](../../../examples/manual/sqlite_import.py):

```sh
uv run --no-sync python examples/manual/sqlite_import.py \
  /tmp/tractor-results.jsonl /tmp/tractor-products.sqlite
```

The program stores only rows with `status="extracted"`, no issues, and a non-null
width. It preserves decimal values as text in SQLite, uses parameterized SQL, and
upserts by the import ID. Running the import again updates the same row instead of
creating duplicates. For the tutorial feed, one item is stored and one is skipped
for review. The input JSONL remains your review/error record.

The schema is deliberately small:

```sql
CREATE TABLE IF NOT EXISTS products (
    id TEXT PRIMARY KEY,
    width_mm TEXT NOT NULL
);
```

The column name makes the normalized unit explicit. Read the string back as Python
`Decimal`. Do not use SQLite floating-point arithmetic on this text if exact decimal
arithmetic is required; calculate with Decimal in the application. For a database
with exact numeric columns, choose a scale and rounding policy appropriate for your
domain before inserting, and record any intentional quantization.

The example's acceptance rule is only a starting point. An issue-free extraction
can still select the wrong source. Add domain range checks, an audited acceptance
policy, and manual review where required before downstream business actions. Keep
source IDs, field-schema version, model weight hash, and original text or an approved
source reference in a production audit record.

For anchored arrays, use a stable document ID plus an occurrence identifier. A
product name alone is not unique: repeated equal names are separate source records.
Preserve the original input text when required by your storage policy.

[Manual home](../index.md) · [Record semantics](../explanation/records.md)
