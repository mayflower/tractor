"""Length-aware independent batches; preserves requests and optimizer boundaries."""


def batch_indices(indices, lengths, runtime):
    batch = []
    for index in indices:
        candidate = [*batch, index]
        if batch and (len(candidate) > runtime['max_microbatch'] or
                      max(lengths[i] for i in candidate) * len(candidate) > runtime['padded_token_budget']):
            yield batch
            batch = []
        batch.append(index)
    if batch:
        yield batch


def checkpointing(backbone, tokens, runtime):
    if tokens > runtime['checkpointing_above_tokens']:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    else:
        backbone.gradient_checkpointing_disable()
