# Sparse reservoir forest qualification

Sparse tree paths and nodal projection replace dense multi-reservoir inverses while retaining independent continuity, energy, and cancellation checks.

Measured source: 3d447538a0b3a3865fe31ed31c3b9cf7f0b924cc. Baseline: d2738eacdc00491e3f62226a33aec7272bfd7ccd. [Original CI](https://github.com/Jacob-Met/hamon-ple-harness/actions/runs/45296211634).

Fresh ABBA subprocesses, one BLAS thread, two samples per arm. CPU covers the solve; RSS includes runtime. Raw records are adjacent.

| Case | Mean CPU s: baseline → candidate | Speedup | Peak RSS KiB: baseline → candidate |
| --- | --- | --- | --- |
| long ring | 0.381661 → 0.016807 | 22.71× | 144116 → 81196 |
| dense mesh | 0.043178 → 0.007876 | 5.48× | 93156 → 79356 |
| parallel exchanges | 0.020892 → 0.007448 | 2.81× | 90248 → 78024 |

All 17 initial tests passed. All twelve benchmark solves matched the independent physical oracle. A subsequently added rough-turbulent Colebrook/minor-loss test passed locally; final CI covers its cold run. Production source is unchanged. Independent review found no algebra blocker.

Long forest-path fill and exact cancellation components remain costs. No official PLE score, full kit readiness, model run, dev-set artifacts, or portal submission is claimed.
