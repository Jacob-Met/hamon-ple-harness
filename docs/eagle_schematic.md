# EAGLE schematic wire connectivity

The task-container command reads one page of a two-color EAGLE vector schematic.

```sh
python3 /tmp/hamon-engineering-toolkit.py pdf-eagle-connectivity SCHEMATIC.pdf --page 1
```

Each entry in `wires` has two endpoints in the rotated CropBox SVG frame. Equal values at corresponding indices in `wire_groups` identify a proposed sheet-local connection. `supply_glyphs` records the printed name, stem and glyph geometry. Each integer under `printed_equivalence_anchors` indexes a wire in a geometrically connected component. A wire crossing in the interiors of two strokes does not contribute a connection. Coincident endpoints, matched horizontal signal labels and visibly terminated supply symbols can connect components. A supply name observed on a grounded bar or supply arrow requires that glyph to join separated wires.

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
