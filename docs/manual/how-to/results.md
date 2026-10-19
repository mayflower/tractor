# Handle normalized results and issues

Every extraction returns `data` and `issues`. Inspect both before storing results.
Missing or ambiguous scalar values are null; list fields use empty arrays.
A null amount is not zero. An issue names the affected field and a stable code;
repeated records also include an index.

Keep the original input in your application when required by your workflow.
The extraction API does not expose model scores, source offsets or evidence options.

[Manual home](../index.md)
