# Preserve an acknowledged tool batch at the delivery cutoff

A task read or MCP response can arrive after the soft delivery reserve. Previously the parent killed Pi at that reserve even when Pi had already issued a valid tool batch. The tool result then never entered the private continuation history. A missing acknowledgement correctly prevents continuation, but it also leaves the delivery pass unable to use the calculation.

The first checked work phase now permits that one pending batch to reach Pi's existing completed-turn boundary, bounded by the task's final reserve. Both deadlines are fixed from the same initial monotonic budget. A waiting model, malformed batch, absent extra headroom or task cancellation gets no extension. The existing task-side shell-command deadline remains unchanged. Completion is still established by the complete private-session history, exact assistant and result identities, and the existing boundary marker where applicable.

## Reproduce

Run `npm ci --no-fund --no-audit` and install `requirements.txt`, then:

```bash
python ship/verify_tool_batch_cutoff.py \
  --baseline 053dd5f0ff394063ffb5723b63962394150ad5f8 \
  --out /tmp/ple-boundary-proof
```

The paired check sends one delayed read through the installed Pi, its registered task tools and the WebSocket task relay. A fresh nonce and 256 supplied values must return through the actual read result before the delivery response can write its receipt. The reference must reproduce the missing acknowledgement; the candidate must preserve it, verify its private history, deliver the matching receipt and perform no second read. Separate real-process controls cover a stalled tool, a waiting model, no extra time and immediate cancellation.

These are protocol controls using a deterministic loopback provider. They do not measure engineering quality, sponsored-model performance, competition score or organizer submission readiness. Committed task artifacts are unchanged; a genuine authorized `eval/run.py` run remains required for a new PLE result.
