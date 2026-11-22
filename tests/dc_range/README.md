# Exact DC limits and safe scalar rendering

This is a preregistered source-only regression course for the transfer-function route. Its first numerical execution is the matched stock/candidate pair in the maintained repository pull-request workflow.

## Defect and contract

For N(s)=1e-300 and D(s)=1e100(s+1), the exact ratio of the accepted binary64 coefficients is strictly positive. Direct floating-point division rounds it to zero. A finite-value check therefore allows a zero scalar claim with both tolerances zero. Negative and half-minimum-subnormal limits expose the same problem.

The repair computes the origin limit from the exact accepted coefficient values, exposes its canonical rational string in dc_gain_exact, and then attempts the existing diagnostic float conversion. A finite nonzero exact limit that overflows or rounds to signed zero has dc_gain=null and dc_gain_status=outside_float_range. Nonzero representable subnormals remain available. Exact zero remains zero.

If N=s^a N0 and D=s^b D0, with nonzero N0(0) and D0(0), the limit is zero for a>b, N0(0)/D0(0) for a=b, and a genuine pole for a<b. The zero numerator also has limit zero. The original denominator origin order continues to distinguish finite from finite_limit when the scalar is available. A genuine pole returns null for both DC fields and pole_at_zero.

The rational string is evidence. Numeric claims targeting dc_gain_exact explicitly refuse, including integer-looking strings and a null result. Numeric claims targeting dc_gain continue to concern the rounded scalar and the requested tolerances. A renderable 1/3 is therefore the binary64 display of 1/3. The prescribed 3u/2 tie rounds to 2u, where u=2^-1074.

## Frozen sources

- Base commit: 57c8e0180d9d38fe4482de687eb4093a509d1154
- Base tree: 9c4581445f455fac9375333f9d4468a05574dbbd
- Base solver SHA256: 7bb044f6fbcd1c2cac363f2c09524ee7bd5b47b146a4052bf7a4836323e59081
- Candidate solver SHA256: 8f2f41a06c8413c69d6cc05e4b19e7212bd28643d5da985db9c22bc69c60312f
- Independent receiver SHA256: 628ba409e74bf066723d0ee76b1782d654b83659233beeb0f47be5af6b22401b
- Registered Pi receiver SHA256: e450fd9c2b240962f338b66e7802aa7e444e4dfc3ea4bd94ca646ffd627daf96
- Protocol SHA256: c7bb567e166ae421b74d422c767b5ce417bb3da1628bc4c63f07e80f686d6e0c

The stock branch contains the new tests, protocol and this README on the frozen base. The candidate adds only the solver change. The protocol contains the exact ordered inputs and decisions. No case, coefficient, tolerance or expected value changes after first execution.

## Independent course

The fourteen factored systems cover an ordinary third, an identically zero numerator, an extra origin zero, a negative finite origin limit, a genuine origin pole, positive and negative extreme underflow, positive and negative overflow, both signed minimum subnormals, both half-subnormal ties, and the three-halves tie.

Expected values come from these factored families. Extreme constants use Fraction.from_float to retain the accepted binary64 value. The subnormal outputs are prescribed independently: u and -u remain available, +/-u/2 are withheld, and 3u/2 displays as 2u.

Every system is called once unclaimed and twice claimed. All 42 calls occur before the aggregate assertion. All claims explicitly use atol=0 and rtol=0. Expected totals are fourteen unclaimed approvals and 28 full-evidence correct claim checks: seven approvals and 21 refusals. Seven limits are renderable, six finite nonzero limits are outside the scalar rendering range, and one is a genuine DC pole.

The actual registered Pi tool adds thirteen request specifications across positive underflow, negative underflow with a common origin factor, the three-halves tie, and a genuine pole. Four are unclaimed; nine claimed requests require one approval and eight refusals. The dedicated unittest wrapper and the maintained pytest route both exercise the registered tool. The representable finite_limit case is covered by the Python course, not this smaller Pi subset.

DC_RANGE_CASE and DC_RANGE_TOTAL record complete-schema checks, claim decisions, false approvals and refusals, exceptions, and paired projection digests. Stock can correctly refuse an unknown evidence key while lacking new evidence. Those schema failures must be reported separately from incorrect scalar approvals.

## Paired preservation and acceptance

Ten systems must retain both legacy DC fields exactly. Only underflow/positive, underflow/negative-common-origin, tie/half-positive and tie/half-negative change those fields.

All fourteen systems must retain all ten Bode fields exactly. Their full unclaimed result after excluding exactly dc_gain, dc_gain_status and dc_gain_exact must also match. This includes the exact root certificates, numerical pole diagnostics and the Bode frequency-evidence sidecar.

Inspect every paired row and aggregate. Retain the inherited Bode 11-system/34-claim and transfer 20-system/80-claim courses with unchanged legacy projection digests, as well as rank 49/196, Lyapunov 32/128, characteristic diagnostics 17/68, the registered-tool tests, and all other maintained workflow gates.

Integration requires all fourteen complete systems, all 28 correct claim checks, zero solver or JSON exceptions, the prescribed paired equalities, actual Pi success, and every maintained gate. Preserve both original CI logs and all failures. Record the actual workflow checkout and tree, because concurrent changes can enter a pull-request merge. Compare the eventual landed tree directly with the successful CI tree before claiming full integration qualification.

## Scope and references

The exact field certifies the limit of the accepted binary64 coefficients. It does not reconstruct an earlier intended decimal value. The existing Bode decimal-string convention, approximate frequency and margin displays, root diagnostics, and shared scalar claim checker remain unchanged. This course makes no universal runtime, physical feedback well-posedness, model-score or benchmark-score claim.

- Python 3.12 [fractions documentation](https://docs.python.org/3.12/library/fractions.html) specifies exact preservation of a supplied float and explains why that value can differ from an intended decimal rational.
- Python 3.12 [math documentation](https://docs.python.org/3.12/library/math.html) describes float spacing and the smallest positive subnormal through ulp(0.0).
- Python 3.12 [sys.float_info documentation](https://docs.python.org/3.12/library/sys.html#sys.float_info) distinguishes the minimum positive normal value from subnormal values.
