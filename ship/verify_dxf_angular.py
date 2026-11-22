"""Replay selected published native CAD angular dimensions through the registered CLI."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


SAMPLES = {
    'librecad': ('44b25aae12aae38a07019076325dcad62aa32658', {'F4': 45.0, 'F5': 225.0}),
    'acadsharp': ('6bb05d6d8b14f6df64a7dfb2e46d1257e0d1337b',
                  {'515': 94.39870535499560, '516': 94.39870535499567}),
}


def verify(label, path, root):
    data = path.read_bytes()
    blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    expected_blob, expected_angles = SAMPLES[label]
    if blob != expected_blob:
        raise ValueError(f'{label} requires original CAD source {expected_blob}, received {blob}')
    raw = subprocess.check_output([sys.executable, str(root / 'engineering_toolkit.py'),
                                   'dxf-dimensions', str(path)])
    result = json.loads(raw)
    found = {r['handle']: r for r in result['dimensions'] if 'angular' in r}
    if set(found) != set(expected_angles):
        raise ValueError(f'{label} angular entity roster differs from the fixture')
    for handle, expected in expected_angles.items():
        r = found[handle]
        a = r['angular']
        if (r['geometry_status'] != 'angular_picture_verified' or a['cad_measurement_raw'] is not None
                or a['picture_sweep_degrees'] is None or a['usable_degrees'] is None
                or not math.isclose(a['usable_degrees'], expected, rel_tol=0., abs_tol=1e-7)
                or any(r[x] is not None for x in ('cad_measurement_model_units', 'computed_model_units',
                                                  'usable_qualified_value'))):
            raise ValueError(f'{label} {handle} failed its oriented native-picture gate: {r}')
        print(f'{label} {handle} {a["usable_degrees"]:.10f} degrees, block {a["picture_block"]}')
    if label == 'librecad' and not math.isclose(found['F5']['angular']['marker_sector_degrees'],135.):
        raise ValueError('LibreCAD F5 no longer exercises the competing 135-degree marker sector')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--librecad', required=True, type=Path, help='librecad/res/dxf/dim_sample.dxf')
    parser.add_argument('--acadsharp', required=True, type=Path, help='samples/sample_AC1009_ascii.dxf')
    args = parser.parse_args()
    for name in SAMPLES:
        verify(name, getattr(args,name), Path(__file__).resolve().parents[1])
