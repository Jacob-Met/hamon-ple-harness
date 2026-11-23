# HAMON physical-engineering agent harness (Google PLE)

An Apache-2.0 harness derived from Google's Pi / Claude Opus 5.5 sample for the Physical-Engineering's Last Exam. It combines the organizer protocol with task-native IFC4 surface geometry/clearance checks and source-native DXF dimensions, unit-safe structural and electrical calculation packs, read-only file inventory, vector PDF extraction and [source-anchored dimensional measurement](docs/pdf_measurement.md).

**Evaluation status:** Existing `artifacts/` are inherited organizer sample-run material, not a scored HAMON dev run for current source. Public 29-task dev data access and a sanctioned zero-spend model evaluation route remain unverified. Static CI or historical template rewards do **not** establish an eligible competition score or leaderboard placement.

## Use it

1. Click **Use this template**. Public works. Make it private if you add a `.env` with API keys or want to keep your code to yourself, and install the [PLE-submission](https://github.com/apps/ple-submission) app on it.
2. Clone it next to the competition repository and set `name` in `submission.yaml`.
3. From the competition repository, check the entry and run the dev set (a `google-cloud` model needs a local proxy, see `docs/models.md`):
   ```bash
   uv run python -m kit check ../your-repo
   uv run python eval/run.py ../your-repo     # writes artifacts/
   ```
4. Commit everything, `artifacts/` included, push, and submit the commit on the competition site.

`artifacts/` currently holds an inherited template run. **Do not present it as a fresh score for this changed harness.** Regenerate it only from the exact submitted source using authorized dev tasks and an approved model endpoint. The source provenance checker explicitly flags the inherited manifest as stale.

For a source-custody receipt from the public organizer kit, run the following
from the benchmark checkout. This reads the entry and its exports on disk. It
runs the original artifact review, returns the per-task results, and also checks
the complete Git source tree, dev-set revision and per-task export provenance.

```bash
uv run python ../your-repo/ship/check_artifact_provenance.py ../your-repo --kit-root .
```

An accepted artifact layout can still have stale source, changed task files or
an empty task. The checker exits 0 for verified provenance (including the kit's
documentation-only changes), 3 for stale or unverified artifacts, and 2 when
the kit could not provide a review. Scoring still requires the authorized
dev-task files and appropriate model and judge access.

## Engineering task commands

Inside the isolated task container, Pi can run `python3 /tmp/hamon-engineering-toolkit.py` with `pdf-vector`, `pdf-dimension-links`, `pdf-auto-measure`, `pdf-measure`, `ifc-narrowphase`, `calc-pack`, `quantity`, `check` and `xlsx-audit`. The tools produce qualified evidence rather than assuming source labels, collision envelopes or engineering code compliance. See the [anchored PDF guide](docs/pdf_measurement.md) for the new source-verified dimension graph: it requires a real PDF word box and two separate crossing endpoint strokes before using an automatically associated reference, then validates four independent directions against the physical affine tensor. Missing/ambiguous evidence fails closed; neither visual proximity nor a passing metric is building-code approval.

The `dual-verify` command also computes passive transformer voltage regulation, operating efficiency and the conditions under which a finite optimal load exists. See the [transformer efficiency contract](docs/transformer_efficiency.md).

The [pile-group compression contract](docs/pile_group_capacity.md) gives the `pile_group_capacity` inputs, the two competing resistance calculations and the scope of the load check.

For a small diaphragm model with strongly contrasting stiffnesses, set
`displacement_precision` to `compensated` in the existing `diaphragm_membrane`
spec. The solver checks up to 64 displacement degrees of freedom and 1,024
strain rows (five per rectangular panel, one per axial bar or spring).
When `verified_displacement_representation` is `binary64_primary_plus_remainder`,
the equilibrium and work checks apply to the sum of `displacements_m` and
`displacement_remainders_m`, element by element. Both arrays hold binary64
numbers in metres and follow the input node order. To reproduce a displacement
in Python, use a Decimal context with at least 60 digits and evaluate
`Decimal.from_float(primary) + Decimal.from_float(remainder)` after parsing
the JSON. A binary64 addition can discard the remainder. The capacity decision
uses the axial force from the decoded pair before rounding the displayed ratio.

In default mode, a bar passes when the exact product of the magnitude of its
reported binary64 `bar_tension_N` and `demand_factor` is at most the allowable
for its sign. The displayed ratio uses the solver’s working precision and is
rounded to binary64. It can read 1.0 on either side of the limit. Error in the
reported force relative to the exact equilibrium solution is outside this comparison.

## Files

| File | Purpose |
|---|---|
| `submission.yaml` | The model, and the install and start commands |
| `serve.py` | The WebSocket endpoint: hands each task to `harness.solve` |
| `harness.py` | Runs Pi on each task. Edit `NOTE` (prompt), `THINKING`, or `UNKNOWN_MODEL` (limits for a model Pi doesn't know) |
| `task-tools.ts` | Points Pi's `read`, `write`, `edit` and `bash` tools at the task container |
| `taskbridge.py` | The competition protocol |
| `package.json`, `package-lock.json` | Pi 1.1.0 and every package it uses, pinned. To upgrade, edit `package.json`, run `npm install --package-lock-only`, commit both |
| `artifacts/` | The dev-set artifacts the portal grades, one folder per dev task |
| `AGENTS.md` | The full rules, for a coding agent working on your entry |


## Native CAD dimension recovery on PDF-outlined drawing text

Run python3 /tmp/hamon-engineering-toolkit.py dxf-dimensions SOURCE.dxf
when the original ASCII DXF accompanies a drawing PDF whose glyphs are
all outlines rather than extractable words. Three independently published
source CAD drawings have zero PDF native words, yet source-native geometric
evidence recovers 62 of 62 dimension geometries (15/15, 25/25, 22/22).
Diameter and radius entities use separate native center/circumference
geometry, never disguised as linear dimensions. This is neither PDF-DXF equivalence nor an official
scored PLE development run. See docs/dxf_native_dimensions.md.


## Curve-aware PDF vector geometry (outlined-text pages)

The existing pdf-vector inventory now computes exact affine-transformed
centerline bounds for SVG cubic and quadratic Bezier paths, including
smooth/relative commands, and affine ellipse/circle extrema. It does not
bridge curved path endpoints as source-certified straight edges. Unknown SVG
arc syntax remains unlocalized instead of guessed; these bboxes reveal
where outlined glyph and curved engineering geometry sits but do not
invent the lettering or physical drawing scale.
See docs/svg_curve_bounds.md and the pinned real-CAD regression.
