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
