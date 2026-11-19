# PLE multi-sheet drawing and scale annotation reader

Task container CLI (requires Poppler: \`pdfinfo\`, \`pdftotext\`, \`pdftocairo\`):

\`\`\`bash
python3 /tmp/hamon-engineering-toolkit.py pdf-vector drawings.pdf \
  --from-page 1 --to-page 4 --limit-paths 250 --limit-words 500
\`\`\`

Reads each actual PDF page through native Poppler's layout and SVG vector
converters. Returns page count, PDF-page word boxes, SVG viewBox, vector
paths/lines/rectangles and native straight-line centerline boxes, embedded
image count (for scan-vs-vector diagnosis), glyph instances, and printed
scale/title-block candidates. It excludes glyph outline definitions from
engineering path counts and never executes PDF scripts. Paths with curves or
unsupported transformations are marked unlocalized rather than assigned
invented boxes.

\`--region xmin ymin xmax ymax\` filters only geometry with bounded
straight-centerline envelopes in SVG viewBox coordinates; unlocalized
curves remain in the bounded sample. The crop is exploratory because
stroke widths, glyphs, rotations and XObject semantics may require
additional shape processing.

Printed strings like \`SCALE 1:50\` or \`NTS\` are reported as unverified
annotations in original PDF page coordinates. **This tool does not
apply them as measurements:** title-block scales may differ from detail
views, native PDF points may not equal plotting length, and a drawing
may have been resized after export. Calibrate known dimensions and
independently validate page rotation and scale before computing any
real-world length or design compliance.

Sources: Poppler \`pdftocairo -svg\` and \`pdftotext -bbox-layout\`;
native malformed/encrypted files and unsupported tools fail loudly.
