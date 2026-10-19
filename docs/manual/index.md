# tractor user manual

**Text in. Structured data out. Ready for your application.**

tractor turns short German and English texts into named fields, product records,
and quantities in the units your software needs. Run its HTTP API on your own CPU
or GPU and call it from any language. The Python SDK preserves decimal values.

This manual follows [Diátaxis](https://diataxis.fr/): learn through tutorials,
complete a task with a how-to guide, look up a contract in reference, or understand
the design through explanation. Examples use the released model and the actual API.

## Learn by doing

1. [Your first extraction](tutorials/first-extraction.md): install, download, serve, and call HTTP.
2. [Build a Python product extractor](tutorials/python.md): fields, records, and normalized values.
3. [Build a batch import](tutorials/batch.md): process JSONL and route uncertain results.

## Complete a task

- [Design fields and extract records](how-to/fields-and-records.md)
- [Convert quantities and choose locales](how-to/quantities.md)
- [Integrate JavaScript and other HTTP clients](how-to/http-clients.md)
- [Use async Python and web applications](how-to/async.md)
- [Store results in a database](how-to/persistence.md)
- [Handle errors, retries, and backpressure](how-to/errors-and-retries.md)
- [Handle normalized results and issues](how-to/results.md)
- [Deploy with Docker and Compose](how-to/deployment.md)
- [Install an offline or pinned model](how-to/offline.md)
- [Validate an integration and diagnose failures](how-to/validation.md)
- [Train and evaluate native extraction tasks](how-to/training.md)

## Look up the contract

- [HTTP request, response, and endpoints](reference/http.md)
- [Python SDK](reference/python.md)
- [Units, numeric rules, and issue codes](reference/quantities-and-issues.md)
- [CLI and runtime settings](reference/cli.md)

## Understand the design

- [How a System One extraction system works](explanation/system-one.md)
- [Records, missing values, and ambiguity](explanation/records.md)
- [Accuracy, performance, and evaluation](explanation/evaluation.md)

## Run the examples

Run commands from the repository root, inside `nix develop`, unless a page says
otherwise. The server is normally at `http://127.0.0.1:8000`; examples also accept
`TRACTOR_URL` for another address. Keep the serving terminal open while using a
second terminal for clients. The [example directory](../../examples/manual) contains
complete programs and request files, not just fragments.

The source and model repositories are currently private. You need access to both
for a fresh installation. Release licensing is pending. The API accepts text;
PDF parsing, OCR, long-document splitting, databases, authentication, and workflow
orchestration belong in the surrounding application.

[Project README](../../README.md) · [Benchmark report](../benchmarks.md) ·
[Architecture](../architecture.md) · [Contributing](../../CONTRIBUTING.md)
