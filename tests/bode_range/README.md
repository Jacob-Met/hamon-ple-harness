# Bode frequency evidence across float range

Exact control certificates can be computed before an ancillary Bode frequency
conversion aborts the response. Finite coefficient arrays can have a gain
crossover at 10^340 rad/s. They can also have one at 10^-340 rad/s, whose
floating rendering becomes zero. The first case previously raised during
frequency conversion; the second could approve a false zero-frequency claim
and lose the phase margin.

Dropping the event would lose useful information. Its phase or gain margin
can remain an ordinary finite number even when its frequency cannot be
rendered. Several distinct events can also lie outside float range, so a
nullable or rounded frequency cannot determine their physical order.

## Preserve the frequency coordinate

The existing Bode calculation constructs exact rational polynomials in
x = omega^2 and isolates their positive roots. This repair retains those
isolating intervals through event ordering, margin evaluation, selection,
and output rendering.

The Bode coefficient convention remains Fraction(str(c)) for each
transported coefficient. Fixture coefficients are formed in Decimal before
float conversion so multiplication by three or five does not introduce a
separate rounding step. The strict transfer-stability certificates retain
their existing, distinct interpretation of accepted binary64 values as
exact rationals. Neither convention is silently changed.

Positive frequency rendering produces a finite positive float or null.
Zero-rounding is explicitly unavailable, not an actual zero-frequency
event. A genuine positive subnormal remains a finite positive rendering.
The extra evidence appears in a separate bode_frequency_evidence object,
preserving the shape of the existing twelve Bode/DC fields.

| Sidecar item | Meaning |
| --- | --- |
| coordinate | omega_squared_rad2_per_s2 |
| rendering_basis | isolating_interval_midpoint |
| selected_gain_index | Selected isolated gain row, or null if none exists |
| selected_phase_index | Selected isolated negative-real row, or null if none exists |
| gain_crossovers / phase_crossovers | One positive-frequency record per corresponding legacy row |
| unity_gain_intervals_rad_s / negative_real_intervals_rad_s | Two endpoint records per legacy interval |
| imaginary_axis_poles_rad_s / removable_holes_rad_s | One positive-frequency record per corresponding event |

Each positive record has two exact rational strings in squared_interval
and one rendering status: finite, below_float_range, or above_float_range.
A true origin endpoint instead has status zero and interval ["0","0"].
A genuinely unbounded endpoint has status unbounded and a null interval.

This distinguishes three legacy null values: an absent isolated crossing,
a present but unrenderable finite frequency, and an unbounded continuum
endpoint. A selected unrenderable crossing keeps its integer index and
finite margin. Absence keeps a null index and null scalar margin.

The range status describes the isolating interval midpoint's rendering.
It does not assert an additional exact comparison of every algebraic root
against a binary64 conversion boundary. The rational bounds remain visible
for a receiver to inspect.

## Order before rendering

Real-axis unit-gain roots and off-axis unit-gain roots are isolated from
different polynomial factors. Their intervals can overlap even when the
underlying roots are distinct. The candidate refines those intervals until
they are disjoint, then merges them in squared-frequency order.

Distinctness follows from removing the real-axis factor from the off-axis
polynomial. Sufficiently narrow disjoint intervals therefore exist. This
does not establish a uniform runtime bound: degree, coefficient growth,
and root separation still affect the work.

The selected row minimizes the absolute margin, with physical event order
retained for ties. The selected frequency is copied from that row even
when its value is null. The margin is not discarded with the frequency.

## Phase without a large or tiny frequency

At a gain crossing, the relevant complex numerator after multiplying by
the denominator's conjugate has the form

    r + j*sqrt(x)*i.

The old evaluation normalized r and i before multiplying the imaginary
part by omega. That intermediate ratio can underflow even when the final
direction is meaningful. A frequency that itself rounds to zero or
overflows makes the problem explicit.

When the existing finite calculation has normal nonzero components and
products, retain it. Otherwise compare r^2 and x*i^2 exactly. If the real
component dominates, use signed components

    (1, sqrt(x*i^2/r^2)).

If the imaginary component dominates, use

    (sqrt(r^2/(x*i^2)), 1).

The square-root ratio is at most one. The existing rational scaled
square-root helper computes it before conversion, so a representable root
is not erased by first rounding its much smaller square. Apply the exact
component signs and then atan2.

The phase-wrap side also needs the exact imaginary sign. For

    G(s) = -s^2 * (B-A*s)/(B+A*s),
    A = 1e-140, B = 1e200,

the unique gain crossover is omega=1, where
G(j)=(B-j*A)/(B+j*A). Its phase is strictly negative but too small to
render as a nonzero angle. The signed margin rounds to +180 degrees.
If the intermediate imaginary component becomes negative zero, the old
modulo expression produces -180 degrees instead. The fallback chooses
the wrap using the exact imaginary sign. The established exact-axis
conventions remain unchanged: positive-real unity has margin -180, and
negative-real unity has margin zero.

These margins remain numerical approximations. Exact rational frequency
intervals do not turn trigonometric or logarithmic displays into symbolic
exact values.

## Independent analytic course

The source base is 17a68fe70859fbdf5b807db9db7d0f286bfb1d34, which combines
the transfer repair with concurrent PR 166 AC changes. The full maintained
candidate CI also qualifies that combined base.

The final protocol SHA256 is
33fd68de94e02b6be7a7d4a697961c5959fefb1a4d51d5839ff0f442efe45789.

| Family | Independent evidence |
| --- | --- |
| 5B/(4As+3B), three scales | Gain frequency B/A = 10^340, 10^-340, or 1; the same phase margin 180-degrees(atan2(4,3)) |
| s(As-B)/(As+B), three scales | Gain frequency 1; negative-real frequency B/A; gain margins -6800, +6800, or 0 dB |
| (B/2)/(As^2+B), two scales | With q=B/A, gain roots x=q/2 and 3q/2, margins -180 and 0, true pole x=q, negative-real interval above that pole |
| Negative unity with an axis hole | A removable hole at omega=10^314; two unity and negative-real intervals; formal feedback identically zero |
| 2(s+1)/(s+1) | No positive-frequency events or isolated scalar crossings |
| Signed phase wrap | Unique gain at omega=1 and a correctly signed +180-degree margin |

The two resonance scales include an unrenderable pole at 10^314 and a
genuinely representable positive subnormal pole at 10^-314. The latter
must not become zero or be incorrectly marked unavailable.

The independent receiver derives every expected x algebraically from
these transfer families. It requires exact interval containment, disjoint
event order, complete aligned evidence, selected-row consistency, and
correct zero/unbounded sentinels. A 100-digit Decimal square root of the
exact rational reference supplies the finite-frequency oracle, with at
most one binary64 ULP of error. This tolerance remains meaningful for
positive subnormals. Representative registered Pi fixtures also check the
existing relative interval-width bound of 2^-56.

There are 11 unclaimed and 34 claimed requests: 45 calls before the
aggregate assertion. Ten unclaimed systems pass; the undefined-feedback
system refuses while retaining its open-loop Bode evidence. All 22
zero-frequency claims use atol=rtol=0 and must refuse. Nine truthful
phase-margin and three truthful gain-margin claims use atol=1e-10 and
rtol=0 and must pass. The required claim total is 12 approvals and
22 refusals, with no solver or strict-JSON errors.

Twenty-four additional registered request specifications cover six
representative fixtures, including the signed-zero wrap. The actual Pi
tool executes them through both maintained unittest and pytest entry
points. No source or test was executed before the protocol freeze.

## Paired evidence and retained behavior

The first execution is one matched stock/candidate pair in the maintained
repository PR workflow. Both branches have identical test and protocol
files; only the solver differs. Original failures, exceptions, and partial
outputs are retained without changing cases or expectations.

The single-arm test records the twelve-field legacy projection for every
system. The paired review requires exact equality for the unit one-pole,
unit phase, subnormal resonance, and absent-crossing controls. It also
requires all nineteen valid inherited transfer-course payloads and the
invalid empty payload to retain their digests. The low gain, low phase,
and signed-wrap projections may change for the explicitly described
repairs. A stock abort is not described as a measured output equality.

The inherited exact transfer 20/80, rank 49/196, Lyapunov 32/128,
characteristic-diagnostic 17/68, and every other maintained gate remain
required. Actual CI and actual landed merge trees are compared separately.
There is no alternate test dispatcher, workflow change, organizer/dev/scored
evaluation, paid-model run, or benchmark performance claim.

## Runtime references

Python 3.12 documents the floating operations used by the renderer and
independent oracle, including ldexp, atan2, signed remainder behavior, and
ULP spacing:

- <https://docs.python.org/3.12/library/math.html>.

The distinction between the smallest positive normal float and a positive
subnormal is documented separately; sys.float_info.min is the former,
while math.ulp(0.0) identifies the smallest positive subnormal:

- <https://docs.python.org/3.12/library/sys.html#sys.float_info>.
