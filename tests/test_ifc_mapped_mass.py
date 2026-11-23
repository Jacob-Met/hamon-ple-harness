"""Mapped rectangular occurrences checked by independent affine mass moments."""
import unittest
import re
import numpy as np
from hamon_conservation_cad_verifier import verify_step_cad_geometry


class Scene:
    def __init__(self):
        self.rows=[]
        self.identity=self.frame(np.eye(3), [0,0,0])
        self.context=self.add(f"IFCGEOMETRICREPRESENTATIONCONTEXT($,'Model',3,1.E-8,#{self.identity},$)")
    def add(self, text):
        self.rows.append(f"#{len(self.rows)+1}={text};")
        return len(self.rows)
    def vector(self, kind, values):
        return self.add(f"{kind}(("+','.join(str(float(v)) for v in values)+"))")
    def frame(self, rotation, translation):
        p=self.vector('IFCCARTESIANPOINT', translation)
        z=self.vector('IFCDIRECTION', rotation[:,2]); x=self.vector('IFCDIRECTION',rotation[:,0])
        return self.add(f"IFCAXIS2PLACEMENT3D(#{p},#{z},#{x})")
    def shape(self, item, representation='SweptSolid'):
        return self.add(f"IFCSHAPEREPRESENTATION(#{self.context},'Body','{representation}',(#{item}))")
    def map(self, item, rotation, translation, scale=1.):
        source=self.add(f"IFCREPRESENTATIONMAP(#{self.identity},#{self.shape(item)})")
        x=self.vector('IFCDIRECTION',rotation[:,0]); y=self.vector('IFCDIRECTION',rotation[:,1]); z=self.vector('IFCDIRECTION',rotation[:,2])
        p=self.vector('IFCCARTESIANPOINT',translation)
        operator=self.add(f"IFCCARTESIANTRANSFORMATIONOPERATOR3D(#{x},#{y},#{p},{float(scale)},#{z})")
        return self.add(f"IFCMAPPEDITEM(#{source},#{operator})")
    def product(self, item, rotation, translation, parent=None):
        frame=self.frame(rotation,translation)
        parent='$' if parent is None else f'#{parent}'
        placement=self.add(f"IFCLOCALPLACEMENT({parent},#{frame})")
        shape=self.shape(item,'MappedRepresentation')
        definition=self.add(f"IFCPRODUCTDEFINITIONSHAPE($,$,(#{shape}))")
        self.add(f"IFCBEAM('beam-{len(self.rows)}',$,$,$,$,#{placement},#{definition},$,$)")
        return placement
    def text(self):
        return 'DATA;\n'+'\n'.join(self.rows)+'\nENDSEC;'


RX=np.array([[1.,0,0],[0,0,-1],[0,1,0]])
RZ=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
MIRROR=np.diag([1.,-1.,1.])


def scene(signed=1, scale=1., nested=False, shift=0., mirror=False, repeated=True):
    s=Scene()
    pp=s.vector('IFCCARTESIANPOINT',[1,-2]); px=s.vector('IFCDIRECTION',[0,1])
    profile_frame=s.add(f"IFCAXIS2PLACEMENT2D(#{pp},#{px})")
    profile=s.add(f"IFCRECTANGLEPROFILEDEF(.AREA.,$,#{profile_frame},2.,4.)")
    body_frame=s.frame(RX,[2,3,4]); direction=s.vector('IFCDIRECTION',[3,0,4*signed])
    solid=s.add(f"IFCEXTRUDEDAREASOLID(#{profile},#{body_frame},#{direction},5.)")
    edges=RX@np.column_stack(([0.,2,0],[-4.,0,0],[3.,0,4*signed]))
    center=RX@np.array([2.5,-2,2*signed])+[2,3,4]
    rot=RZ@(MIRROR if mirror else np.eye(3))
    item=s.map(solid,rot,[5,6,7],scale)
    center=scale*rot@center+[5,6,7]; edges=scale*rot@edges
    if nested:
        item=s.map(item,RX,[-3,8,-2],.5)
        center=.5*RX@center+[-3,8,-2]; edges=.5*RX@edges
    parts=[]
    for rotation,translation in [(RZ,[shift+10,20,30])]+([(RX,[shift-4,5,-8])] if repeated else []):
        s.product(item,rotation,translation)
        parts.append((rotation@center+translation,rotation@edges))
    s.add(f"IFCEXTRUDEDAREASOLID(#{profile},#{body_frame},#{direction},50000.)")
    return s,parts


def expected(parts,density):
    volumes=[abs(float(np.linalg.det(edges))) for _,edges in parts]
    masses=np.asarray(volumes)*density
    origin=parts[0][0]
    center=origin+sum(m*(c-origin) for m,(c,_) in zip(masses,parts))/sum(masses)
    inertia=np.zeros((3,3));surface=0.;corners=[]
    for mass,(c,edges) in zip(masses,parts):
        covariance=edges@edges.T/12
        delta=c-center
        inertia+=mass*(np.trace(covariance)*np.eye(3)-covariance + (delta@delta)*np.eye(3)-np.outer(delta,delta))
        surface+=2*sum(np.linalg.norm(np.cross(edges[:,i],edges[:,j])) for i,j in [(0,1),(1,2),(2,0)])
        corners.extend(c+edges@np.array([i,j,k]) for i in [-.5,.5] for j in [-.5,.5] for k in [-.5,.5])
    corners=np.asarray(corners)
    return sum(volumes),sum(masses),center,inertia,surface,corners.max(axis=0)-corners.min(axis=0)


class MappedMass(unittest.TestCase):
    def check(self,s,parts,unit=1.):
        got=verify_step_cad_geometry({'step_text':s.text(),'density_kg_m3':1200.,'unit_scale_m':unit})
        volume,mass,center,inertia,area,extent=expected(parts,1200.)
        self.assertTrue(got['conservation_verified'],got)
        self.assertEqual(got['ifc_product_count'],len(parts))
        self.assertAlmostEqual(got['volume_m3']/volume,1.,places=12)
        self.assertAlmostEqual(got['mass_kg']/mass,1.,places=12)
        np.testing.assert_allclose(got['centroid_m'],center,rtol=0,atol=2e-9)
        np.testing.assert_allclose(got['inertia_tensor_centroid_kg_m2'],inertia,rtol=2e-12,atol=2e-14*np.max(abs(inertia)))
        np.testing.assert_allclose(got['aabb_extents_m'],extent,rtol=0,atol=2e-9)
        self.assertAlmostEqual(got['surface_area_m2']/area,1.,places=12)
    def test_shared_mapped_occurrences_and_signed_sweeps(self):
        for sign in [-1,1]:
            for scale in [.125,1.,2.]:
                with self.subTest(sign=sign,scale=scale):self.check(*scene(sign,scale))
    def test_nested_reflected_mapping_in_survey_coordinates(self):
        for sign in [-1,1]:
            for shift in [0.,729011.25]:
                with self.subTest(sign=sign,shift=shift):self.check(*scene(sign,2.,True,shift,True))
    def test_same_definition_and_same_location_are_distinct_products(self):
        s,parts=scene(repeated=False)
        mapped=next(int(row.split('=')[0][1:]) for row in reversed(s.rows) if '=IFCMAPPEDITEM(' in row)
        s.product(mapped,RZ,[10,20,30]); parts.append(parts[0])
        self.check(s,parts)


    def test_reciprocal_source_and_map_scale_preserves_physical_cube(self):
        for scale in (1e-110,1.,1e110):
            with self.subTest(scale=scale):
                s=Scene(); length=1/scale
                profile=s.add(f"IFCRECTANGLEPROFILEDEF(.AREA.,$,$,{length},{length})")
                direction=s.vector('IFCDIRECTION',[0,0,1])
                solid=s.add(f"IFCEXTRUDEDAREASOLID(#{profile},#{s.identity},#{direction},{length})")
                mapped=s.map(solid,np.eye(3),[0,0,0],scale)
                s.product(mapped,np.eye(3),[0,0,0])
                # The same one-metre cube for every source-coordinate scale.
                self.check(s,[(np.array([0.,0.,.5]),np.eye(3))])
                if scale==1.:
                    i=next(i for i,row in enumerate(s.rows) if '=IFCCARTESIANTRANSFORMATIONOPERATOR3D(' in row)
                    refs=list(map(int,re.findall(r'#(\d+)',s.rows[i])))
                    s.rows[i]=f'#{i+1}=IFCCARTESIANTRANSFORMATIONOPERATOR3D($,$,#{refs[3]},$,$);'
                    self.check(s,[(np.array([0.,0.,.5]),np.eye(3))])

    def test_mapping_lengths_follow_the_project_unit_once(self):
        s,parts=scene(-1,.5,True,0,True)
        self.check(s,[(center*.001,edges*.001) for center,edges in parts],unit=.001)

    def test_unqualified_or_partial_mapping_refuses(self):
        for defect in ('origin','cycle','missing_source','missing_target','zero_scale','negative_scale',
                       'nonuniform','partial_axes','missing_origin','zero_axis','nonorthogonal','extra_item','opening'):
            with self.subTest(defect=defect):
                s,_=scene(repeated=False)
                imap=next(i for i,row in enumerate(s.rows) if '=IFCMAPPEDITEM(' in row)
                iop=next(i for i,row in enumerate(s.rows) if '=IFCCARTESIANTRANSFORMATIONOPERATOR3D(' in row)
                isource=next(i for i,row in enumerate(s.rows) if '=IFCREPRESENTATIONMAP(' in row)
                mapped=imap+1
                if defect=='origin':
                    nonidentity=s.frame(np.eye(3),[.25,0,0])
                    s.rows[isource]=s.rows[isource].replace(f'(#{s.identity},',f'(#{nonidentity},')
                elif defect=='cycle':
                    source_shape=int(re.findall(r'#(\d+)',s.rows[isource])[-1])
                    s.rows[source_shape-1]=re.sub(r'\(#\d+\)\);$',f'(#{mapped}));',s.rows[source_shape-1])
                elif defect in ('missing_source','missing_target'):
                    source,target=map(int,re.findall(r'#(\d+)',s.rows[imap])[1:])
                    s.rows[imap]=f'#{mapped}=IFCMAPPEDITEM(#{999999 if defect=="missing_source" else source},#{999999 if defect=="missing_target" else target});'
                elif defect in ('zero_scale','negative_scale'):
                    s.rows[iop]=s.rows[iop].replace(',1.0,',',0.0,' if defect=='zero_scale' else ',-1.0,')
                elif defect=='nonuniform':
                    s.rows[iop]=s.rows[iop].replace('OPERATOR3D(','OPERATOR3DNONUNIFORM(').replace(');',',2.,3.);')
                elif defect=='partial_axes':
                    s.rows[iop]=re.sub(r'OPERATOR3D\(#\d+,','OPERATOR3D($,',s.rows[iop])
                elif defect in ('missing_origin','zero_axis','nonorthogonal'):
                    refs=list(map(int,re.findall(r'#(\d+)',s.rows[iop])))
                    if defect=='missing_origin':
                        s.rows[iop]=s.rows[iop].replace(f',#{refs[3]},',',#999999,')
                    else:
                        axis=refs[1]
                        s.rows[axis-1]=f'#{axis}=IFCDIRECTION((0.,0.,0.));' if defect=='zero_axis' else f'#{axis}=IFCDIRECTION((1.,1.,1.));'
                elif defect=='extra_item':
                    s.rows=[row.replace(f'(#{mapped}));',f'(#{mapped},#{mapped}));') if '=IFCSHAPEREPRESENTATION(' in row else row for row in s.rows]
                else:
                    s.add("IFCRELVOIDSELEMENT('void',$,$,$,#999998,#999999)")
                with self.assertRaises(ValueError):
                    verify_step_cad_geometry({'step_text':s.text(),'density_kg_m3':1200.})


if __name__=='__main__':unittest.main()
