# Inspect a drawing window with source coordinates

`pdf-window` creates a bounded PNG of one PDF sheet, together with a source hash,
page number, image hash, and reversible pixel-to-SVG transform. It works with
outlined lettering and scanned pages because it renders the actual marks. It
does not recognize text or promote a visually plausible number to a verified
engineering dimension.

## Task workflow

The existing engineering toolkit is provisioned into the task container. Start
with an overview, then use the agent's ordinary `read` tool on the returned PNG:

```sh
python3 /tmp/hamon-engineering-toolkit.py pdf-window drawing.pdf \
  --page 1 --output /tmp/sheet-1.png
```

The default is 120 dpi. For a detail, choose a region from the `pdf-vector`
inventory and render at greater resolution:

```sh
python3 /tmp/hamon-engineering-toolkit.py pdf-vector drawing.pdf \
  --from-page 1 --to-page 1
python3 /tmp/hamon-engineering-toolkit.py pdf-window drawing.pdf \
  --page 1 --region 70 440 110 510 --dpi 300 \
  --output /tmp/dimension-detail.png
```

Those region values are illustrative. Use the coordinates of the actual sheet.
Keep the emitted JSON alongside any cited visual evidence. `image.path` is a
normal PNG path; the existing task-native image-read operation can open it.
Choose a new output name for another view: existing files are never replaced.

## One explicit frame for vectors and windows

Both commands use `poppler-cropbox-rotated-no-fit-v1`: the displayed CropBox,
with the PDF page rotation applied, origin at the top left, and y increasing
downward. The vector conversion explicitly disables paper shrinking and
centering. This matters for fractional page sizes. The PNG explicitly uses the
CropBox; Poppler's default raster MediaBox can have a different extent and
origin.

The window region uses that SVG frame, **not** native PDF word-box coordinates
or unrotated PDF user space. `pdf-words` and the word fields in `pdf-vector`
retain their separate frame. The coordinate transform does not certify a
physical scale, plotted dimension, or PDF `UserUnit` interpretation.

The requested region is rounded outward to integer pixel edges. The result
reports both `requested_region_svg` and the actual `covered_region_svg`.
At a fractional page edge, the final pixel can extend less than one pixel
beyond the continuous sheet boundary. This is disclosed in the latter box.

An affine array `[a,b,c,d,e,f]` maps `(x,y)` to
`(a*x+c*y+e, b*x+d*y+f)`. `svg_to_pixel` maps source coordinates into the image;
`pixel_to_svg` maps image coordinates back into the displayed sheet. Integers
denote pixel edges; pixel centers are `(x+0.5,y+0.5)`.

The source is copied into a private bounded snapshot before native conversion.
Both vector inventory and PNG use that same snapshot. Source hash, page,
SVG hash, PNG hash, raster dimensions, and coordinate frame are returned as
one evidence record. The input document is not edited.

Windows copy exact pixels from a bounded full-page native render using the
system libpng codec. Re-rendering only a clipped PDF region can change Cairo's
antialiasing at source marks; the extraction path preserves the full rendering
exactly, including its pixel grid and color channels.

## Bounds and failures

- One page per invocation; valid page numbers are 1 through the document count.
- PDF input is at most 50 MB, SVG intermediate at most 15 MB.
- Resolution is 36 through 600 dpi; default 120.
- The emitted image is at most 4 million pixels, 8192 pixels on either side,
  and 8 MB encoded. The complete intermediate raster is limited to 32 million
  pixels, 32767 pixels per axis, and 128 MB encoded. Its decoded RGBA buffer is
  bounded to 128 MB; a smaller output window does not bypass that source bound.
- Invalid or out-of-page regions are refused. Resolution is never silently
  lowered and coordinates are never silently clipped.
- PNG output is published only after its dimensions and size are checked.
  Atomic exclusive creation also refuses an output that appears concurrently.
- The native conversion timeout is 30 seconds per rendering command.

If a full overview exceeds the image bound, choose a lower dpi or a smaller
region. A highly compressed scan can still exceed the encoded PNG limit;
request a smaller window. The tool preserves all marks and leaves their
interpretation to the reader.

## Evidence and limits

The tests use literal PDF fiducials with nonzero MediaBoxes, smaller CropBoxes,
all four quarter-turn rotations, fractional sizes, `UserUnit` metadata, and
multiple pages. Eighteen native raster controls require at most one pixel of
edge error. A separately rendered full-page oracle must match the cropped
window exactly as decoded RGB. Tests also exercise the actual provisioned CLI,
source immutability, existing-file refusal, and resource bounds.

`ship/verify_pdf_window_corpus.py` repeats full-image and crop equality on three
hash-pinned public CAD PDFs. They are exposed development controls, not unseen
holdouts. These checks measure faithful rendering and coordinate consistency;
they do not measure text recognition, agent task success, or a hosted PLE score.

Existing `pdf-measure` and `pdf-auto-measure` source-word, reference-dimension,
and calibration gates are unchanged. A visual window does not bypass them.
