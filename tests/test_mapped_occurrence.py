"""IFC4 mapped occurrences exercised through the same router entry point as the participant."""
import unittest

import numpy as np

import hamon_ifc_spatial_clash as geometry


def mapped_tetrahedron(kind,scale,origin,ax1,ax2,ax3,scale23,source_origin=(0,0,0),project_mm=False):
    units=1000 if project_mm else 1
    def point(x):
        return '('+','.join(str(float(v)*units) for v in x)+')'
    def direction(x):
        return '('+','.join(str(float(v)) for v in x)+')'
    vertices=((0.,0.,0.),(1.,0.,0.),(0.,2.,0.),(0.,0.,3.))
    p='(' + ','.join(point(x) for x in vertices)+')'
    def matarg(ref,vec):
        return '$' if vec is None else '#'+str(ref)
    trans=','.join([matarg(31,ax1),matarg(32,ax2),'#30',str(scale) if scale else '$',matarg(33,ax3)]+list(map(str,scale23)))
    axes='\n'.join('#%d=IFCDIRECTION(%s);'%(i,direction(x)) for i,x in [(31,ax1),(32,ax2),(33,ax3)] if x is not None)
    return f"""ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('ViewDefinition [CoordinationView]'),'2;1');
FILE_NAME('occurrence.ifc','2026-01-01T00:00:00',('test'),('test'),'geometry course','test','');
FILE_SCHEMA(('IFC4'));
ENDSEC;
DATA;
#1=IFCCARTESIANPOINT((0.,0.,0.));
#2=IFCAXIS2PLACEMENT3D(#1,$,$);
#3=IFCGEOMETRICREPRESENTATIONCONTEXT($,'Model',3,1.E-6,#2,$);
#4=IFCSIUNIT(*,.LENGTHUNIT.,{'.MILLI.' if project_mm else '$'},.METRE.);
#5=IFCUNITASSIGNMENT((#4));
#6=IFCPROJECT('0AzR0sseD4bBOcPJ50wSE1',$,'Course',$,$,$,$,(#3),#5);
#10=IFCCARTESIANPOINTLIST3D({p});
#11=IFCTRIANGULATEDFACESET(#10,$,.T.,((1,3,2),(1,2,4),(1,4,3),(2,3,4)),$);
#12=IFCSHAPEREPRESENTATION(#3,'Body','Tessellation',(#11));
#13=IFCCARTESIANPOINT({point(source_origin)});
#14=IFCAXIS2PLACEMENT3D(#13,$,$);
#15=IFCREPRESENTATIONMAP(#14,#12);
#30=IFCCARTESIANPOINT({point(origin)});
{axes}
#34={kind}({trans});
#35=IFCMAPPEDITEM(#15,#34);
#36=IFCSHAPEREPRESENTATION(#3,'Body','MappedRepresentation',(#35));
#37=IFCPRODUCTDEFINITIONSHAPE($,$,(#36));
#38=IFCLOCALPLACEMENT($,#2);
#40=IFCBEAM('0AzR0sseD4bBOcPJ50wSE2',$,'mapped',$,$,#38,#37,$,.BEAM.);
#51=IFCPRODUCTDEFINITIONSHAPE($,$,(#12));
#52=IFCDUCTSEGMENT('0AzR0sseD4bBOcPJ50wSE3',$,'fixed',$,$,#38,#51,$,.RIGIDSEGMENT.);
ENDSEC;
END-ISO-10303-21;
"""


class MappedOccurrenceTests(unittest.TestCase):
    def test_nested_aggregates_and_explicit_omissions_keep_their_positions(self):
        self.assertEqual(
            geometry._parse_step_args("((1,2),(3,4)),$,IFCLENGTHMEASURE(2.),'a,b'"),
            [[[1, 2], [3, 4]], None, ("IFCLENGTHMEASURE", [2.0]), "a,b"],
        )

    def test_source_and_target_placement_in_project_units(self):
        operator = "IFCCARTESIANTRANSFORMATIONOPERATOR3D"
        row = mapped_tetrahedron(operator, 2., (10, 1, 0),
                                 (0, 2, 0), (-3, 0, 0), (0, 0, 4), (),
                                 source_origin=(1, 0, 0), project_mm=True)
        mapped, fixed = geometry.load_ifc_solids(row)
        np.testing.assert_allclose(mapped.vertices, [(10, 3, 0), (10, 5, 0),
                                                     (6, 3, 0), (10, 3, 6)])
        np.testing.assert_allclose(fixed.vertices, [(0, 0, 0), (1, 0, 0),
                                                    (0, 2, 0), (0, 0, 3)])
        result = geometry.run_router_ifc_clash({"ifc_text": row, "cross_discipline_only": False})
        self.assertEqual(result["solid_count"], 2)
        self.assertEqual(result["all_pairs"][0]["status"], "clear")

    def test_nonuniform_scaling_and_reflection(self):
        operator = "IFCCARTESIANTRANSFORMATIONOPERATOR3DNONUNIFORM"
        row = mapped_tetrahedron(operator, 2., (5, 0, 0),
                                 None, (0, -1, 0), None, ("$", 3.))
        mapped, _ = geometry.load_ifc_solids(row)
        np.testing.assert_allclose(mapped.vertices, [(5, 0, 0), (7, 0, 0),
                                                     (5, -4, 0), (5, 0, 9)])
