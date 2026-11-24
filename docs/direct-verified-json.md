# Direct replies for complete, verified JSON calculations

The presolve router already computes and verifies a bounded inline physical
calculation, then includes its JSON in a model request. When the instruction
asks for exactly that JSON, this final generation step is redundant. The new
path returns the complete result through the ordinary evaluator response before
model configuration, Pi startup or task CLI copying.

## Admission

The instruction must begin with one of the existing complete-result covers:

```
Run the calculation below and return its JSON result.
Run the calculations below and return their JSON results in the same order.
```

The remaining text must be one supported JSON request or an ordered batch of at
most two. Every calculation must pass the existing physical certificate. The
parser rejects extra prose, duplicate fields, unsupported parameters, partial
batches and mismatched singular/plural forms. Bare or fenced specifications
without the cover keep the model path.

Observed context must be empty and unambiguous: no files, incomplete inventory,
warnings, file metadata, MCP work or deliverables. The result must retain its
exact instruction binding. Malformed or nonfinite JSON and incomplete result
batches are refused. The successful response is the original complete JSON
string, without paraphrasing or numeric reserialization. Existing verifier
assumptions still apply; a transport success or legacy presolve flag alone
cannot finish a task.

Independent review found a file in `a/b/c/d/e/f/physical-input.csv` that appeared
as a warning-free empty inventory. The bounded probe now reports an incomplete
directory scan when nonempty directories are pruned by depth or exclusion,
directory symlinks are not traversed, or enumeration fails. That warning vetoes
admission without expanding the scan limit.

## Protocol and cancellation

The ordinary read-only inventory precedes admission. The bounded calculation
runs outside the event loop so the reader can process cancellation. The harness
checks cancellation and the task deadline before and after calculation. A
cancelled or expired task cannot obtain a new direct calculated reply. The
mathematical function may finish its bounded in-flight work; the cancelled path
does not proceed to a provider call.

The harness returns a string to `taskbridge._session`. That session remains the
sole owner of `done` and its existing cancellation summary grace. No duplicate
terminal event or invented usage record is emitted. A successful direct reply
has zero provider requests and zero provider inference. Inventory and local
solver CPU still cost resources; this change does not infer a dollar saving.

## Verification and limits

`tests/test_certified_inline_reply.py` runs actual evaluator WebSockets and the
installed Pi runtime against loopback Anthropic SSE. Paired courses cover a pipe
network, electrochemical cell, multi-branch jacketed CSTR and ordered batch. The
baseline disables only the new completion hook, retaining normal presolve and
model execution. Each candidate must remove the baseline's one provider request
and return equal physical JSON. The fixture's synthetic token counters are not
a price or engineering-accuracy oracle. Requests and input bytes are measured.

Separate protocol controls complete all four cases with no model credentials,
name or URL. They require one inventory operation, one terminal event, no task
writes and no model usage events. Other controls cover extra obligations, stale
source binding, malformed results, partial requests, cancellation and expiry
before and during computation. Actual files exercise depth, hidden and excluded
directory pruning; a fault-injected enumeration error also prevents admission.

`tests/replay_empty_worlds.py` compares the ordinary and candidate paths over all
retained public task instructions using real Pi, SSE and evaluator WebSockets.
The deliberately empty evaluator cannot fulfill their deliverable obligations.
Both arms must retain equal terminal summaries and provider counts, with zero
observed direct admissions and no false completed-delivery claims.

This qualifies a complete-calculation execution path. It does not establish a
new organizer score, real-model accuracy, cost reduction on the 29 public tasks,
or a Pareto ranking. No portal submission is used. Coverage remains limited to
tasks completely expressible through an already-qualified closed calculation.
