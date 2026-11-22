# Native DXF dimension evidence

Run: python3 /tmp/hamon-engineering-toolkit.py dxf-dimensions SOURCE.dxf

This operates on ASCII DXF DIMENSION entities without packages, OCR or network.
For aligned linear entities it computes the Euclidean length between extension
origin points 13/23 and 14/24. For rotated linear entities it projects that
vector onto the angle held in group 50. For radial and diameter entities, codes 10/20/30 and 15/25/35
supply the center/opposite or circle endpoint; Euclidean distance
provides a dimension-qualified radius/diameter (not a linear width).
A stored CAD measurement in group 42
must agree with geometry to within 1e-4 relative (1e-6 absolute). A mismatch
fails closed, missing CAD measurement is marked geometry-only, and all other
dimension types are reported but not converted to a linear source value.

Evidence records source SHA-256, byte length, entity group offset, handle,
layer, original text override, dimensions in drawing-model units, and INSUNITS
declarations. The text override is not validated displayed PDF glyphs.
No drawing scale, PDF-DXF sheet registration, construction safety or code
compliance can be inferred. It restores access to native dimension semantics
when outlined-font PDF drawings are accompanied by their DXF source; it does
not solve PDF-only outlined text.


Primary DXF references: Autodesk Common Dimension Group Codes
https://help.autodesk.com/cloudhelp/2023/ENU/AutoCAD-DXF/files/GUID-EDD54EAC-A339-4EBA-AEA6-EC8066505E2B.htm
and Radial and Diameter Dimension Group Codes
https://help.autodesk.com/cloudhelp/2024/ENU/AutoCAD-DXF/files/GUID-165A992D-9017-4C1E-B8CC-E70A17191BFE.htm

## Angular dimensions and the selected sweep

For a two-line angle, the same supporting lines can describe an acute angle, its supplement or a reflex sweep. On LibreCAD’s published dimension sample, handle `F5` has a shared vertex at (37.6, 43.36) and end rays at 45° and 180°. The marker sits near 357°. Dividing both lines into four consecutive sectors returns 135°. The linked `*D14` picture draws its arc counterclockwise from 180° through 360° to 45°, which yields **225°**. The two arrow tips lie on the same circle at those end rays. Handle `F4` on that drawing yields 45°. [LibreCAD dimension sample](https://github.com/LibreCAD/LibreCAD/blob/master/librecad/res/dxf/dim_sample.dxf)

The `angular` record keeps the marker-sector diagnostic, the sweep recovered from the native picture, the raw group 42 cache and its positive value in radians when valid, and the qualified `usable_degrees`. The existing fields for linear model units stay empty for angular dimensions. A two-line dimension uses WCS points 13, 14, 15 and 10, and an OCS arc marker in group 16. A three-point dimension uses vertex 15, the directed rays through points 13 and 14, and the WCS arc marker in group 10. The reader checks coplanarity and constructs the OCS basis from the extrusion normal. [Autodesk angular dimension groups](https://help.autodesk.com/cloudhelp/2024/ENU/AutoCAD-DXF/files/GUID-09821B78-9F8E-43BA-82F2-8C931485EDC9.htm)

For untransformed +Z geometry, the reader follows the dimension’s reference to an anonymous block. The circle must agree with the center, model-space elevation and marker radius. Circular-arc endpoints or tips of nondegenerate triangular arrowheads must anchor two distinct end rays. An arrowhead tip lies on the matching circle. Both base vertices must lie on the same side of its radial line and opposite sides of its tangent line. Every positive arc span and the marker must lie inside the same oriented sweep. Arrows and text often trim the drawn arc, so its length alone gives the wrong measurement. The reader never uses annotation text as a geometric witness. Transformed or ambiguous native pictures do not supply a verified sweep.

`angular_picture_verified` identifies an independently constructed picture sweep. `angular_picture_and_cad_agree` adds agreement with the stored measurement. `angular_geometry_and_cad_agree` identifies a marker-sector calculation corroborated by its native cache when a qualifying picture is absent. `angular_unverified_sector`, `angular_picture_unresolved`, `cad_measurement_mismatch` and `geometry_unavailable` have no usable angular value. These statuses describe the authored DXF in model space. They do not establish correspondence with an external PDF or validity of a physical design.

### Reproducing the two CAD source checks

Download `librecad/res/dxf/dim_sample.dxf` from LibreCAD and `samples/sample_AC1009_ascii.dxf` from ACadSharp, then run the command below from the repository root. The verifier checks both original Git blobs and invokes the same CLI that the task harness calls. All four checked entities lack a numeric group 42 cache. [LibreCAD source drawing](https://github.com/LibreCAD/LibreCAD/blob/78014ecde2f53fdd8c06453f3854b1cae1a3fd0d/librecad/res/dxf/dim_sample.dxf) · [ACadSharp source drawing](https://github.com/DomCR/ACadSharp/blob/6ef32981983904695060e055c66acbe5a6220b89/samples/sample_AC1009_ascii.dxf)

```bash
python ship/verify_dxf_angular.py --librecad dim_sample.dxf --acadsharp sample_AC1009_ascii.dxf
```

For ACadSharp’s handles `515` and `516`, the publisher’s unmodified `DxfReader.Read` and `Dimension.Measurement` returned 94.39870535499560° (three-point) and 94.39870535499566° (two-line), respectively, from the original AC1009 file. Both returned `94°` from `GetMeasurementText()`. These are two entity forms in one drawing. The native SDK 8 build rejected the author’s newer C# syntax. Building the same sources with supported SDK 10 succeeded. [ACadSharp angular measurement source](https://github.com/DomCR/ACadSharp/tree/6ef32981983904695060e055c66acbe5a6220b89/src/ACadSharp/Entities)

The first cross-project test of a marker-only calculation scored six correct, one wrong and one without a decoded reference across eight entities. The wrong result was LibreCAD `F5`. The oriented-picture candidate then ran on 16 entries from eight additional file exports. After excluding repeated geometries, the two first representatives had missing caches and degraded `94?` text, so the preregistered direct numeric-reference gate failed. Fourteen later entries repeated the two geometries. The independent author-reader check above supplied the missing comparison on the original bytes. It does not turn the repeated exports into additional examples. The regression set covers authored CAD samples and tool tests. It supplies neither a held engineering-part error rate nor a new PLE benchmark score.

### Model-space ordinate dimensions

For an axis-aligned ordinate in the default +Z plane, the reader subtracts the UCS datum (group 10) from the feature (group 13). Bit 64 of group 70 selects X when set and Y when clear. The ordinate record reports the signed coordinate in model units. The qualified dimension is its absolute magnitude. Moving the leader endpoint (group 14) in the same plane leaves it unchanged. A conflicting or invalid measurement cache withholds the qualified value. Rotated or tilted coordinate frames, ambiguous group values, and nonplanar points also remain unresolved. Drawing-unit declarations and text overrides keep their separate evidence status. [Autodesk ordinate group codes](https://help.autodesk.com/cloudhelp/2023/ENU/AutoCAD-DXF/files/GUID-72F01288-0D63-43E8-8179-8CE3BA544C40.htm)

A zero along the measured axis is supported if the datum and feature are distinct points. When they coincide, the reader withholds the value. The output is in drawing model units. Dimension styles can scale the displayed annotation. In ezdxf’s public ordinate example, the drawing contains projected distances of 4 and 2.5 model units and stores the labels `400` and `250` under a dimension style with `DIMLFAC=100`. Use the dimension block or the applicable style when transcribing a printed value. [ezdxf ordinate example](https://github.com/mozman/ezdxf/blob/b34993dfe96dfc6c0af994867e777fc2dcdf4bd6/examples/render/dimension_ordinate_global_features.py)
