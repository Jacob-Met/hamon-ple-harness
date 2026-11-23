# EAGLE schematic wire connectivity

The default task-container command reads one page of a two-color EAGLE vector schematic.

```sh
python3 /tmp/hamon-engineering-toolkit.py pdf-eagle-connectivity SCHEMATIC.pdf --page 1
```

Each entry in `wires` has two endpoints in the rotated CropBox SVG frame. Equal values at corresponding indices in `wire_groups` identify a proposed sheet-local connection. `supply_glyphs` records the printed name, stem and glyph geometry. Each integer under `printed_equivalence_anchors` indexes a wire in a geometrically connected component. A wire crossing in the interiors of two strokes does not contribute a connection. Coincident endpoints, signal labels with glyph baselines aligned to their wires and visibly terminated supply symbols can connect components. A supply name observed on a grounded bar or supply arrow requires that glyph to join separated wires.

The supported export uses green linework for wires, red linework for symbols and native PDF text. Inspect the returned evidence before using a proposed connection. The page may contain ambiguous labels, invisible or unrecognized wires, split paths, different symbol conventions, or signals that continue on other sheets. The command returns `unresolved_profile` when required ink, recognized PDF creator or complete supported geometry is unavailable. Its group identifiers are local to a page. It does not identify device pins or certify a complete electrical netlist.

## External comparison

On the Arduino Uno through-hole Rev3e development drawing, endpoint equality recovered 52 of 61 native nets exactly. A printed-name rule alone joined seven unrelated DTR/GND wire pairs. Requiring the actual supply glyph removed these false joins. The complete candidate recovered 59 nets exactly and left 353 connected pairs separated.

The Mega 2560 Rev3 was selected before opening its vendor drawing or native CAD. The algorithm and scoring course were frozen before that evaluation. All predictions used only the two-page PDF. Its complete 30,520-byte forecast was recovered by SHA256 before the independent EAGLE archive was requested. Both the printed title and native filename identify the Rev3e schematic. An independent receiver matched all 483 native wire sections using the physical paper units and an independently recovered page translation.

| Mega 2560 Rev3e | Visible wire sections | Endpoint-only exact nets | Candidate exact nets | Candidate false joins | False separations, endpoints → candidate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sheet 1 | 326 | 38/86 | 82/86 | 0 | 2,736 → 74 |
| Sheet 2 | 157 | 28/31 | 31/31 | 0 | 208 → 0 |
| Complete design with page-local groups | 483 | 60/111 | 102/111 | 0 | 4,910 → 2,040 |

A false join or separation counts an unordered pair of visible wire sections. An exact net has exactly the native membership, including every wire and excluding every foreign wire. The independent whole-document result includes 1,966 unjoined cross-sheet pairs. Sheet 1 retains 53 GND, 11 +3V3, eight VIN and two AREF separated pairs. Drawings from other exporters, physical pins and the PLE competition score have no measured outcome in this course.

The replay needs Poppler and the vendor's paired files. Download the [schematic PDF](https://content.arduino.cc/assets/MEGA2560_Rev3e_sch.pdf) and [native EAGLE ZIP](https://content.arduino.cc/assets/MEGA2560_Rev3e-reference.zip) from Arduino.

```sh
python3 ship/verify_eagle_schematic.py MEGA2560_Rev3e_sch.pdf MEGA2560_Rev3e-reference.zip
```

The replay verifies the vendor hashes, invokes the public command for both sheets before opening the CAD, checks a one-to-one geometric correspondence, and recounts sheet and design membership.


## Flat multi-sheet designs

For a PDF containing **one flat EAGLE schematic**, use the document mode.

```sh
python3 /tmp/hamon-engineering-toolkit.py pdf-eagle-connectivity SCHEMATIC.pdf --flat-document
```

The outer index of `document_wire_groups` follows the sheets in `pages`. Each inner index follows that sheet’s `wires`. Equal document IDs identify a proposed connection. `inter_sheet_anchors` records which named sheet-local groups contributed to each join. Use page mode for hierarchical designs and collections of unrelated drawings because signal names can repeat in different scopes.

A join requires the same exact name on two or more sheets and one component per participating sheet. Each participating component normally needs a single printed name. Recognized power and ground names also require a corresponding glyph on each participating sheet. Unsupported sheets leave whole-document membership unresolved.

In the initial flat-document reader, the four qualifying names on the two-sheet Mega comparison were +5V, PE0, PE1 and USBVCC. Joining their components recovered **106 of 111** complete native nets and reduced false separations from **2,040 to 854**. False joins stayed at zero. GND accounts for 755 of the remaining separated pairs and RESET for 78. The ground-symbol rule also recognizes a ground net when every participating component has physical ground bars on at least two distinct wire islands. The bars must carry the same printed name. A competing text anchor is accepted as incidental only when it occurs on one wire island in the entire document and that island bears a matching ground bar. Competing supply names or glyphs still veto the join. The output records the witness island indices and the incidental names.

Before baseline orientation was used, sheet 1 of the Mega PDF had 16 such GND islands, one also bearing the text RN5D. Sheet 2 has five. The added connection recovered 689 native GND pairs, taking pair recall from 85.39% to 97.18% (5,681 of 5,846). Every predicted pair belongs to the same native net. Exact nets remained at 106 of 111. The 165 remaining separated pairs are RESET (78), GND (66), +3V3 (11), VIN (8) and AREF (2). An independent three-sheet Arduino Due PDF returned `unresolved_profile` with no document groups. Its export lacks the EAGLE creator marker and the third sheet has no recognized green wire segments.

### Text orientation

The reader matches PDF word boxes to contiguous origins of the Cairo glyphs in drawing order. It accepts a baseline only when the glyphs share one font and advance along one of the four page axes. It measures the word and nearby wires in that baseline’s coordinate frame. Perpendicular text, incomplete runs and overlapping or ambiguous glyph evidence cannot supply a wire label. The result uses the existing Poppler conversion and needs no new runtime library.

On the same Mega drawing this recovered all **78** remaining RESET pairs. The PDF prints RESET along its wire, while an adjacent component label runs perpendicular to it. The six cross-sheet links now recover **107 of 111** complete nets and **5,759 of 5,846** true wire pairs (98.51% recall). All predicted pairs are correct. The remaining 87 separations are GND (66), +3V3 (11), VIN (8) and AREF (2).

The native comparison on the single-sheet UNO TH Rev3e remained at 59 of 61 complete nets, zero false joins and 353 false separations over 447 wires. The three-sheet Due continued to return the same unresolved evidence. To check the Mega’s physical endpoint coordinates, local memberships, cross-sheet memberships and witness groups under 90°, 180° and 270° rotations, run the replay with the existing pypdf test dependency.

```sh
python3 ship/verify_eagle_schematic.py MEGA2560_Rev3e_sch.pdf MEGA2560_Rev3e-reference.zip --quarter-turns
```

### Labels over a free wire end

EAGLE can print a net name across the open end of a wire. The page reader uses the first native glyph origin and the direction of the complete glyph run to locate its baseline. At least half of the printed name must overlap one collinear wire island. A free end within that label must be clear of other green wires and red symbol strokes, including the interior of each stroke.

A label qualified this way can extend a separately established same-page net. A supply name requires a matching physical supply symbol. Conflicting names on either component prevent the join, subject to the existing repeated-ground-bar rule for a uniquely identified incidental annotation. The decision uses the groups present before any free-end joins. Each accepted entry in `wire_end_label_witnesses` gives the name, wire index, physical endpoint and glyph baseline origin in page points.

| Manufacturer drawing | Wire sections | Complete native nets, before → after | False joins | Missed native wire pairs, before → after |
| --- | ---: | ---: | ---: | ---: |
| Arduino Mega 2560 Rev3e, both sheets in flat-document mode | 483 | 107 → 109 of 111 | 0 | 87 → 17 |
| Arduino Uno TH Rev3e, single sheet | 447 | 59 → 60 of 61 | 0 | 353 → 302 |

The Mega result recovers AREF and GND in full, plus two pairs on +3V3. Its 17 remaining separations comprise eight VIN pairs and nine +3V3 pairs. The Uno result recovers its remaining GND wire. All 302 outstanding Uno pairs belong to +5V. The comparisons use manufacturer CAD membership as the reference for PDF-only predictions. The Mega endpoint coordinates and group memberships also pass the three quarter-turn replays.
