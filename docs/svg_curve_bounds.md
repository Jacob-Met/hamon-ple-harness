# Exact centerline bounds for outlined CAD PDF geometry

The existing Poppler SVG centerline inventory previously localized only
straight M/L/H/V/Z source paths. On the pinned real external corpus at
haiha-ux/cad-sample-tests@150f4dce05a41a85d956189de8ded2cdd0d4a4d3,
the 698 SVG vector primitives in three PDFs included 175 paths that
were unlocalized because they contained curves (62 + 61 + 52).

The new bounds code parses path M/L/H/V/C/S/Q/T/Z commands, including
relative coordinates, and evaluates cubic Bezier extrema at the exact
roots of each x/y derivative **after** the declared SVG affine transform.
Quadratics are converted algebraically to cubics; smooth commands reflect
the immediately preceding compatible control. Ellipse/circle affine
bounds use the support function of the transformed radii.

A bbox is a mathematical centerline bound, not glyph recognition, not a
stroke-width expansion, not a certified scale, and not proof of PDF-DXF
registration. Unknown elliptic arc commands, malformed numbers, invalid
transformations and over-budget paths remain unlocalized rather than
reported as false geometry. Curves never become fake straight-line
measurement segments. Source-level changed-control tests show that the
computed extrema change with the actual path.
