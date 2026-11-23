# STEP length units

The CAD verifier resolves a selected solid definition's length unit through the
shape representations that directly contain it and their referenced
`GLOBAL_UNIT_ASSIGNED_CONTEXT`. It does not infer length from labels, prefixes on
angle or time units, or unrelated unit definitions. Multiple representations of
the selected definition must agree on the exact conversion to metres. Selecting
a solid from a file with several definitions selects that solid's contexts.

The reader preserves simple and complex Part 21 entity components. It recognizes
metres with the twenty SI prefixes from yocto through yotta, plus positive
`CONVERSION_BASED_UNIT` chains using typed length measures. Conversion units must
declare the length dimension, and every referenced conversion unit must have the
same dimension. Decimal conversion factors are multiplied as rational values
before rounding the final scale. Unit names are labels; their spelling never
overrides the referenced conversion factor.

Missing references, cycles, nonpositive conversion factors, conflicting length
assignments, unsupported unit definitions, dimensional mismatches, and scales
outside the finite positive binary64 range withhold geometric properties. An
explicit finite positive `unit_scale_m` remains an intentional override. IFC
continues to use its existing typed project-unit resolver.

## Legacy inputs and provenance

Minimal inputs without represented unit metadata remain supported. A single
standalone length-unit definition supplies the scale; multiple standalone length
definitions require an explicit scale. If no length units are declared, the
legacy metre default is reported as `implicit_metre`. A unitless representation
context can use this default only when the file has no unit definitions.
Representations with a declared unit assignment must contain exactly one length
unit. Detached unit contexts cannot supply a scale to unreferenced geometry.

Successful STEP reports include `unit_scale_source`, and inferred scales include
the applicable representation-context and length-unit entity IDs. The source is
one of `representation_context`, `standalone_length_unit`, `implicit_metre`, or
`explicit_override`.

This change qualifies the existing supported solid definitions. It does not add
assembly placement, curved-surface meshing, or support for complex solid topology.
Mass properties continue to describe the finite binary64 vertices after scaling;
the exact decimal conversion graph does not make the entire coordinate pipeline
an arbitrary-precision CAD kernel.

## Qualification

The preserved baseline certified a two-metre cube as 8e-9 m³ when a milliradian
angle unit or an unused millimetre definition appeared, and as 0.000131096512 m³
when one point was labeled `INCH`. It also ignored declared inch/foot conversion
chains. The regression-only checkpoint has six failing cases in two methods.

The expanded public-route suite checks selected definitions with separate
contexts, shared definitions with agreeing or conflicting contexts, all twenty
prefixes, multi-stage conversions with cancelling extreme factors, malformed
graphs, overrides, record order, comments, quoted labels, and legacy inputs.
It compares volume, area, mass, centre, full centroidal inertia, and principal
moments against independently scaled cube formulas.

References: [ISO 10303-41 measure schema](https://www.steptools.com/stds/smrl/data/resource_docs/fundamentals_of_product_description_and_support/sys/19_schema.htm)
and its [unit-definition examples](https://www.steptools.com/stds/smrl/data/resource_docs/fundamentals_of_product_description_and_support/sys/examples.htm).
