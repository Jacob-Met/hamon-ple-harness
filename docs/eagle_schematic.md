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

A printed occurrence can supply evidence for only one wire island across ordinary wire labels, supply symbols and free-end proposals. Separate occurrences of the same name can connect the islands. For unfilled straight symbol paths, the clearance check follows the segments that actually draw ink, including closure edges and disjoint subpaths. For curves and filled shapes it conservatively withholds a free-end join inside the transformed bounds expanded by 0.13 page points. Unlocalized red geometry withholds free-end joins on that page.

| Manufacturer drawing | Wire sections | Complete native nets, before → after | False joins | Missed native wire pairs, before → after |
| --- | ---: | ---: | ---: | ---: |
| Arduino Mega 2560 Rev3e, both sheets in flat-document mode | 483 | 107 → 109 of 111 | 0 | 87 → 17 |
| Arduino Uno TH Rev3e, single sheet | 447 | 59 → 60 of 61 | 0 | 353 → 302 |

The Mega result recovers AREF and GND in full, plus two pairs on +3V3. Its 17 remaining separations comprise eight VIN pairs and nine +3V3 pairs. The Uno result recovers its remaining GND wire. All 302 outstanding Uno pairs belong to +5V. The comparisons use manufacturer CAD membership as the reference for PDF-only predictions. The Mega endpoint coordinates and group memberships also pass the three quarter-turn replays.


### Power names printed beside supply arrows

A supply value can be rotated independently of its arrow and printed beside the stem. The reader now recognizes this placement from the native glyph run. It still requires the complete red stem and both arrow wings at a green wire endpoint. The word must run parallel to the stem, lie entirely on one side, and cover at least half of the stem in projection. In units of stem length, its nearest box edge must be more than 0.1 and less than 1 away, and its farthest edge may be at most 2 away.

This evidence can extend only a power name already established by an independent ordinary supply glyph on the same page. Ambiguous ordinary tip labels, competing side words, other names on either wire component, and reuse of one printed occurrence all prevent the proposed join. A plausible nearby power word whose native direction is unknown can veto a match but cannot establish one. Negative and decimal voltage spellings receive a separate power-only baseline field; they do not broaden the older wire-label spelling rule.

An occurrence already used by an ordinary tip glyph or an ordinary wire-label anchor cannot also supply a side name. A word proposed for both a loose wire end and a different arrow is rejected by both extension paths. These checks use the original component and occurrence evidence before applying proposed joins. Accepted entries in `supply_glyphs` retain the stem endpoints, the printed box and baseline, and `label_placement: "side"`.

On the same Uno TH Rev3e drawing, the two remaining +5V islands have complete arrows with values printed vertically beside their stems. Recognizing their two distinct values recovers all 302 outstanding native wire pairs:

| Manufacturer drawing | Visible wire sections | Complete native nets, before → after | False joins | Missed native wire pairs, before → after |
| --- | ---: | ---: | ---: | ---: |
| Arduino Uno TH Rev3e | 447 | 60 → 61 of 61 | 0 → 0 | 302 → 0 |
| Arduino Mega 2560 Rev3e, flat document | 483 | 109 → 109 of 111 | 0 → 0 | 17 → 17 |

An independent receiver froze the PDF-only predictions for each original drawing and all three quarter-turns before opening the corresponding manufacturer CAD. All 930 visible wire sections had unique geometric matches. Page and document membership, wire endpoints, glyph stems and the side-label boxes were covariant under rotation. The three-sheet Due export remained unresolved. A separate receiver checked distinct labels and shared-occurrence counterexamples under four rotations and both source orders.

The Uno replay uses the manufacturer's [schematic PDF](https://content.arduino.cc/assets/UNO-TH_Rev3e_sch.pdf) and [native EAGLE ZIP](https://content.arduino.cc/assets/UNO-TH_Rev3e-reference.zip). It checks all three source hashes and produces every PDF prediction before opening the native schematic.

```sh
python3 ship/verify_eagle_uno.py UNO-TH_Rev3e_sch.pdf UNO-TH_Rev3e-reference.zip --quarter-turns
```

These are measured visible-wire connectivity results on the named drawings. They do not establish device-pin connectivity, support for another export profile, or a PLE competition score.

## Native EAGLE schematic and PCB

When a flat EAGLE XML schematic is available, the toolkit can read its logical pins and package pads directly.

```sh
python3 /tmp/hamon-engineering-toolkit.py eagle-netlist SCHEMATIC.sch --board LAYOUT.brd
```

The optional board comparison reads signal contact references after generating the schematic prediction. Each logical row identifies its source, part, gate, pin, net, package and pads. A pin may map to several physical pads. Package-free supply symbols contribute logical rows. An explicit pin connection takes priority over an automatic power connection. Unwired power pins use the rail name defined by the symbol, including power pins on an uninvoked gate. The output preserves the full symbol pin name and removes its distinguishing `@` suffix only when deriving an automatic rail name. See section 8.4 of the [EAGLE 9 manual](https://forums.autodesk.com/autodesk/attachments/autodesk/fusion-electronics-forum-en/8948/1/manual_en.pdf).

Inspect `board_comparison.agreement`, `exact_contacts` and `mismatches`. Each mismatch has the printed element and pad identifiers and the net name from each source. A missing side is null. The comparison checks logical membership recorded in the board file. Routed copper continuity, component values and hierarchical module expansion require separate evidence.

On the Arduino Due V02g, explicitly referenced schematic pins recovered 604 of the PCB's 605 signal contacts. The extra contact was RESET pad 5, assigned to 5-GND through an unwired power pin. Including that EAGLE rule recovered all 605 contacts and all 163 physical nets. On the Adafruit Feather RP2040 Original, a schematic-only prediction produced before opening the board signal assignments matched all 207 PCB contacts and all 52 physical nets. Both predictions used the specific paired revisions below.

- [Arduino Due CAD archive](https://docs.arduino.cc/static/f051671bbed80ddf7352991d62cb0d5c/A000056-cad-files.zip)
- [Adafruit Feather RP2040 Original, revision ea88166](https://github.com/adafruit/Adafruit-Feather-RP2040-PCB/tree/ea88166891ee0a1697a3899a5d55ab3722a2f125)

### Routed copper witnesses

Run `eagle-netlist SCHEMATIC.sch --board LAYOUT.brd --routed-net NET` to check a named signal for an explicit route between its schematic-predicted terminals. The existing `board_comparison` reports declared membership. The separate `routed_continuity` result reports a source-addressed walk to each destination when all terminals connect. Each step identifies the trace, via or terminal, its copper layer, and a position in nanometres. Source trace and via ordinals are zero-based within each kind on the requested signal. Deleting contact declarations leaves the physical geometry intact.

The initial search follows straight, continuous, positive-width outer-layer trace centerlines, cardinally rotated pad centers, full-span vias and plated through pads. It splits a trace at existing collinear route vertices. The centerline must clear board holes, transformed footprint holes and neighboring pad or via drills. An approach to selected plating must reach outside its bore on the used face. Its wire steps carry a `plated_approach` reference to the supporting pad or via. If another drill touches the bore, this search omits that plated connection. A clipped surface pad can still feed a straight wire when it crosses the bore rim inside the inscribed disk of the pad. The search uses only verified approaches, so an interior-only shortcut cannot hide a valid detour. Dashed or dotted artwork cannot establish a continuous centerline; a separate continuous route remains eligible.

When that search is unresolved, a finite-width pass subtracts every known drill void from supported copper regions. It uses inscribed copper polygons, circumscribed drill polygons and a numerical guard. Distinct connected pieces join only through positive-area overlap; tangency alone supplies no proof. Each successful `material` component gives an explicit `path_nm` polyline within one surviving source piece and at least 1 nm nominal in-plane clearance. The path follows material around a bore instead of claiming a straight chord through the hole. `exit_overlap` records an interior disk in the intersection. A cross-layer step requires an independently approved full-span plated pad or via and an intact circular collar; its evidence records the drill center and barrel-wall position. Pieces of a pad separated by drilling do not acquire a virtual connection through their terminal name.

The finite pass supports straight round- or flat-ended traces, round-ended circular EAGLE traces, cardinal rectangular SMDs when their library and board rules establish zero roundness, and conservative inscribed disks for rounded or locally rotated SMDs. It requires an explicit round annulus for plated pads. A full-span round via can use explicit dimensions or its board’s supported outer annulus rules. Undetermined annuli, non-round plated pads, flat-ended curved traces, pours, unsupported placements and other omitted artwork can leave real routes unresolved. A full-width butt joint may also remain unresolved because the sufficient overlap test erodes each primitive independently. Unknown drill geometry prevents either proof. A missing geometry dependency is reported explicitly. The existing Shapely 2.1.2 dependency supplies polygon operations. Autodesk documents the [SMD roundness and rotation](https://help.autodesk.com/cloudhelp/ENU/Fusion-ECAD/files/ECD-CLI-S.htm) and the [distinction between library and placed-board pad data](https://help.autodesk.com/cloudhelp/ENU/Fusion-ECAD/files/ECD-ULP-SMD.htm).

These are sufficient witnesses for nominal layout topology. `unresolved` does not establish an open circuit. Whole-board safety, shorts, fabrication tolerances and design-rule correctness require additional evidence; the 1 nm guard is a numerical proof margin, not a fabrication specification.

The registered Arduino Due `N$3` and Adafruit Feather RP2040 Original `MISO` controls can be replayed with `python ship/verify_eagle_routed_copper.py --raw RAW --repo . --out OUT`. Supply the Due files as `due-0.sch` and `due-1.brd`, and the Feather Original files under `feather/` using their manufacturer filenames. The receiver checks each intact board digest, verifies every primitive in each ordered walk against the original XML, certifies the finite copper gaps after the registered edits, and tests that deleting contactrefs alone changes membership independently of the copper path.

On the Arduino Due, a 0.60 mm hole at (12.4049, 28.4999) cuts the full width of the 0.254 mm diagonal trace on `N$3` for at least 0.54358 mm. The routed result becomes unresolved while all 605 contact and 163 net declarations still match. Moving the hole 2 mm in X restores the original witness.

### Finite-width manufacturer controls

The independent course in `ship/eagle_finite_course_v1.json` fixes the source hashes, all 19 edited-board hashes and expected physical outcomes before candidate execution. It contains nine connected layouts and ten layouts for which the supported artwork must not yield a route. On the pinned Due trace, a centered 0.127 mm bore leaves 0.0635 mm of copper on each side of the 0.254 mm trace; a 0.204 mm bore leaves 0.025 mm. Both now have explicit material paths. A centered 0.254 mm bore closes those passages and remains unresolved, as do the larger cut controls.

The course also checks a rotated footprint drill, wire-end overlap versus tangency and separation, a transformed SMD landing, a valid parallel detour and a second hole severing it, and an off-center plated landing on the Feather. The independent receiver checks the emitted path against the true circular holes and the source wire capsules without importing the geometry backend. All 19 controls retain their original declared contact and net counts.

```sh
python ship/verify_eagle_finite_copper.py --raw RAW --repo . \
  --frozen ship/eagle_finite_course_v1.json --out finite-copper-result.json
```

The provisioner transfers `hamon_eagle_material.py` with the ordinary toolkit modules. `ship/verify_eagle_provisioned.py` exercises that provisioner and the actual copied CLI through a local WebSocket receiver with Shapely available and with Python site packages disabled. Copying modules does not install third-party packages into the task container: if its interpreter lacks the finite-width backend, intact centerline witnesses remain available and an otherwise unsupported material route reports `finite_width_geometry_dependency_unavailable`. These local checks do not establish the official task image's installed packages or replace the organizer's Docker, cancellation and evaluation gates.

### Checking a curved route between two pads

`eagle-netlist SCHEMATIC.sch --board LAYOUT.brd --routed-net NET --routed-pads PART1 PAD1 PART2 PAD2` checks two distinct pads from the schematic’s selected net. All its artwork and all board and footprint drills still participate. The output identifies the pair and reports the full net’s terminal count.

A signed EAGLE curve specifies an angular sweep between endpoints. The finite pass builds an inscribed-width tube around a sequence of chords. It subtracts the maximum arc-to-chord deviation from the trace radius before buffering, then applies the existing clearance guard. In-plane coordinates can retain sub-nanometre source detail. Flat-ended arcs, sweeps of 360 degrees or more, and geometries outside the bounded approximation remain unsupported.

On the unmodified Arduino Due, U1 pad 37 to USB1 pad 3 on D+ traverses 21 curved wire records. U1 pad 38 to USB1 pad 2 on D− traverses 12. An independent polar-distance check certifies each entire returned curved-wire path segment against the original width. Drilling through the longest traversed arc breaks each witness; shifting the same drill away restores it. The board also contains an automatic full-span via on each five-terminal net. The complete-net check below exercises those two crossings.

Run `python ship/verify_eagle_arc_copper.py --repo . --raw RAW --out OUT` using the same pinned Due files as the finite-width course.

### Automatic through-via annuli

For a round via spanning layers 1 through 16, the finite-width pass can derive nominal outer copper from a single board rule set. It multiplies the drill diameter by `rvViaOuter`, clamps the radial ring width between `rlMinViaOuter` and `rlMaxViaOuter`, and adds twice that width to the drill diameter. The rule fraction must be explicit and the bounds must use mm or mil. The computed outer diameter rounds inward at sub-nanometre precision. See [Autodesk's EAGLE annular-ring explanation](https://www.autodesk.com/support/technical/article/caas/tsarticles/ts/mUNCp0xBxwO1rJ9L5DOhL.html).

On the original Arduino Due both 0.4064 mm drills use a 0.25 ratio with an 8–20 mil ring. The 8 mil minimum gives a 0.2032 mm ring and a 0.8128 mm outer diameter. Each full-net query returns four pad-to-pad paths covering five terminals. The plated path enters an intact annular collar on one face, traverses the physical barrel wall, and exits on the other. The output records the source rules, drill and conservative diameter.

Run `python ship/verify_eagle_via_rules.py --repo . --raw RAW --out OUT` on the same pinned files. Its controls remove or interrupt the collar, shift a drill away, substitute a blind or nonplated hole, and remove, duplicate or corrupt a required rule. A damaged collar can make a path unresolved even when other copper survives.
