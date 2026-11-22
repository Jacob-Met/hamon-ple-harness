import math
import unittest
import tempfile
from pathlib import Path

from hamon_dxf import _dimension, scan_dxf


def groups(kind, points, normal=(0., 0., 1.), cache=None):
    result = [(70, str(kind))]
    for code, xyz in {**points, 210: normal}.items():
        result.extend((code + offset, str(value)) for offset, value in zip((0, 10, 20), xyz))
    if cache is not None:
        result.append((42, str(cache)))
    return result


def vertex_point(degrees):
    a = math.radians(degrees)
    return (math.cos(a), math.sin(a), 0.)


def triangle(degrees, shape='arrow'):
    x,y,_ = vertex_point(degrees)
    radial, tangent = complex(x,y), complex(-y,x)
    offsets = {'arrow':(.08j+.02,.08j-.02), 'flat':(.04j,.08j), 'radial':(.08+.02j,.08-.02j)}[shape]
    corners = [radial, radial+offsets[0].real*radial+offsets[0].imag*tangent,
               radial+offsets[1].real*radial+offsets[1].imag*tangent]
    return [(p.real,p.imag,0.) for p in (*corners,corners[-1])]


def native_picture_file(path, shape):
    records = [(0,'SECTION'),(2,'BLOCKS'),(0,'BLOCK'),(2,'*D1'),(10,0),(20,0),
               (0,'ARC'),(10,0),(20,0),(40,1),(50,185),(51,35)]
    for angle in (45,180):
        records.append((0,'SOLID'))
        for code, point in zip((10,11,12,13),triangle(angle,shape)):
            records.extend([(code,point[0]),(code+10,point[1])])
    records.extend([(0,'ENDBLK'),(0,'ENDSEC'),(0,'SECTION'),(2,'ENTITIES'),(0,'DIMENSION'),(2,'*D1')])
    records.extend(groups(2,{13:(0,0,0),14:vertex_point(45),15:(0,0,0),10:vertex_point(180),16:vertex_point(356)}))
    records.extend([(0,'ENDSEC'),(0,'EOF')])
    path.write_text(''.join(f'{code}\n{value}\n' for code,value in records))


class AngularDimensionTests(unittest.TestCase):
    def test_two_lines_choose_marker_sector_and_keep_lengths_empty(self):
        points = {13: (-1, 0, 0), 14: (2, 0, 0), 15: (0, 0, 0), 10: vertex_point(55)}
        for marker, expected in ((30, 55), (75, 125), (210, 55), (255, 125)):
            with self.subTest(marker=marker):
                row = _dimension(groups(2, {**points,16:vertex_point(marker)}, cache=math.radians(expected)), 1, 0)
                self.assertEqual(row['geometry_status'], 'angular_geometry_and_cad_agree')
                self.assertAlmostEqual(row['angular']['usable_degrees'], expected)
                for key in ('cad_measurement_model_units', 'computed_model_units', 'usable_qualified_value'):
                    self.assertIsNone(row[key])

    def test_three_points_accept_reflex_and_refuse_contradictory_cache(self):
        points = {15: (0,0,0), 13: vertex_point(0), 14: vertex_point(55), 10: vertex_point(200)}
        for cache in (None, -1.):
            row = _dimension(groups(5, points, cache=cache), 1, 0)
            self.assertEqual(row['geometry_status'], 'angular_unverified_sector')
            self.assertAlmostEqual(row['angular']['marker_sector_degrees'], 305)
            self.assertIsNone(row['angular']['usable_degrees'])
        row = _dimension(groups(5, points, cache=math.radians(55)), 1, 0)
        self.assertEqual(row['geometry_status'], 'cad_measurement_mismatch')
        self.assertIsNone(row['angular']['usable_degrees'])

    def test_translated_tilted_plane_resolves_ocs_marker(self):
        # For normal +Y the OCS X axis points along -X and its Y axis along +Z.
        def wcs(xy):
            return (11.-xy[0], 2., 19.+xy[1])
        points = {13:wcs((-1.,0.)),14:wcs((1.,0.)),15:wcs((0.,0.)),10:wcs(vertex_point(55))}
        m=vertex_point(75)
        points[16]=(-11.+m[0],19.+m[1],2.)
        result=_dimension(groups(2, points, normal=(0.,1.,0.), cache=math.radians(125)), 1, 0)
        self.assertAlmostEqual(result['angular']['usable_degrees'], 125.)
        points[16]=(points[16][0],points[16][1],6.)
        self.assertIsNone(_dimension(groups(2, points, normal=(0.,1.,0.)), 1, 0)['angular']['usable_degrees'])

    def test_linked_major_arc_supplies_orientation_and_untrimmed_extent(self):
        points = {13:(0,0,0),14:vertex_point(45),15:(0,0,0),10:vertex_point(180),16:vertex_point(356)}
        # The arc is trimmed for arrows and text, so its 210-degree span is not the measurement.
        picture = {'arcs': [([0,0,0],1,185,35)], 'arrows': [triangle(45), triangle(180)]}
        result = _dimension(groups(2, points), 1, 0, picture)
        self.assertAlmostEqual(result['angular']['marker_sector_degrees'], 135.)
        self.assertAlmostEqual(result['angular']['usable_degrees'], 225.)
        self.assertEqual(result['geometry_status'], 'angular_picture_verified')
        self.assertIsNone(_dimension(groups(2, points, cache=math.radians(135)),1,0,picture)['angular']['usable_degrees'])
        picture['arcs'] = [([1,0,0],1,185,35)]
        self.assertIsNone(_dimension(groups(2, points, cache=math.radians(135)),1,0,picture)['angular']['usable_degrees'])

    def test_opposite_ray_wrong_sweep_and_missing_anchor_refuse(self):
        points = {13:(0,0,0),14:vertex_point(45),15:(0,0,0),10:vertex_point(180),16:vertex_point(356)}
        for arc, arrows in (((50,175),[45,180]),((185,35),[]),((185,35),[45,225]),((185,35),[45,180,225])):
            picture = {'arcs':[([0,0,0],1,*arc)], 'arrows':[triangle(a) for a in arrows]}
            self.assertIsNone(_dimension(groups(2, points),1,0,picture)['angular']['usable_degrees'])

    def test_native_solid_arrow_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'angular.dxf'
            for shape in ('arrow','flat','radial'):
                with self.subTest(shape=shape):
                    native_picture_file(path,shape)
                    row=scan_dxf(path)['dimensions'][0]
                    if shape=='arrow': self.assertAlmostEqual(row['angular']['usable_degrees'],225.)
                    else: self.assertIsNone(row['angular']['usable_degrees'])

    def test_missing_degenerate_and_boundary_inputs_stay_unqualified(self):
        base = {13:(-1.,0.,0.),14:(2.,0.,0.),15:(0.,0.,0.),10:vertex_point(55),16:vertex_point(20)}
        cases = [{k:v for k,v in base.items() if k!=16}, {**base,10:(1.,0.,0.)},
                 {**base,16:(0.,0.,0.)},{**base,16:vertex_point(55)},{**base,14:base[13]},
                 {**base,14:(2.,0.,1.)}]
        for p in cases:
            with self.subTest(points=p):
                self.assertIsNone(_dimension(groups(2,p),1,0)['angular']['usable_degrees'])
