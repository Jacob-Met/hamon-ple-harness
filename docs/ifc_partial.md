# Partial IFC occurrence and clash coverage

Pass `allow_partial: true` to the registered `hamon_ifc_spatial_clash` tool to inventory physical occurrences when some geometry is unresolved. The default call keeps its complete-file behavior.

The reader records every source product, including openings, features, virtual elements and spatial products. The positive quantity and pairing scope covers physical elements. Each product row carries its STEP ID, GUID, representation roles and the IDs of its declared void relationships. Openings contribute by subtraction from their host. Fillings remain independent physical occurrences. A broken shared geometry reference affects each dependent product. A failed cutter withholds its host's net geometry and quantity. Unassignable links and unresolved project units prevent a dimensional model report.

Physical rows distinguish known meshes, certified empty products and unresolved geometry. The separate volume status permits surface pairing when the boundary is usable but a compound volume remains unresolved. The known volume ID list and sum describe per-occurrence net quantities. Overlaps between separate positive products can contribute to both quantities.

Pair endpoints use STEP IDs. The requested count comprises evaluated pairs, certified-empty pairs, missing-geometry pairs and exceptional numeric pairs. Evaluated pairs identify which distances were measured and which clearances follow from an axis-aligned bounding-box lower bound. A measured contact can establish a zero minimum with unresolved pairs elsewhere. An incomplete model retains null whole-volume, hard-clash and clearance-violation totals.

Partial mode admits an intentionally limited set of complete polyhedral bodies. It withholds circular or indexed sweeps and advanced native BReps outside that certification. A missing Body stays unresolved. A successfully consumed host has zero volume. An independent source-based qualification must bind any known quantity to its STEP occurrence and preserve the unresolved denominator.
