"""Controlled extraction backends for Clairic profile software tests.

They return fixed candidate spans per field description and count calls. They
test protocol and processing behavior only, never model quality.
"""

ROUTE = "/client/v4/accounts/local/ai/run/@cf/cloudflare/clairic"


def spans_of(text, value):
    spans, start = [], 0
    while (index := text.find(value, start)) >= 0:
        spans.append({"text": value, "start": index, "end": index + len(value)})
        start = index + 1
    return spans


class DescriptionBackend:
    """Finds the configured literal candidates for each field description.

    `signals` maps a candidate text to raw scores, returned like BiQwen's
    candidates_with_signals. One call is one simulated backbone pass.
    """

    def __init__(self, answers, signals=None, tokens=None):
        self.answers = answers
        self.signals = signals
        self.tokens = tokens
        self.calls = []
        self.encoder_calls = 0
        self.last_tokens = 0
        self.last_timings = {}

    def candidates(self, request):
        return self.candidates_with_signals(request)[0]

    def candidates_with_signals(self, request):
        if request.anchor:  # Server warmup of the record path.
            return [], {}
        self.calls.append(request)
        self.encoder_calls += 1
        self.last_tokens = self.tokens or 5 + len(request.text) + 3 * len(request.fields)
        record, signals = {}, {}
        for key, field in request.fields.items():
            record[key] = [s for value in self.answers.get(field.description, [])
                           for s in spans_of(request.text, value)]
            for span in record[key]:
                if self.signals and span["text"] in self.signals:
                    signals[(None, key, span["start"], span["end"])] = self.signals[span["text"]]
        if self.signals is None:
            return [record], {}
        return [record], signals

    @property
    def inference_calls(self):
        """Calls other than the server's warmup request."""
        return [r for r in self.calls if r.text != "Atlas: Breite 12 cm."]
