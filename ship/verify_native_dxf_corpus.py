"""Pinned independent CAD dimensional regression; not scored official PLE."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import hamon_dxf, subprocess, tempfile, math, json
CASES={
 'bai_a/CST-A-001_mounting_plate':('mm',15,15,(120,80,6,3)),
 'bai_b/CST-B-001_studio_plan':('mm',25,25,(6400,4900,1800)),
 'bai_c/CST-C-001_step_bearing':('inch',22,22,(12,3.5,0.75,1,2)),
}
def run(root):
    root=Path(root);out=[]
    for base,(unit,expected,usable,proofs) in CASES.items():
        dxf=root/(base+'.dxf');pdf=root/(base+'.pdf')
        assert not subprocess.run(['pdftotext','-f','1','-l','1',str(pdf),'-'],check=True,capture_output=True).stdout.strip(),pdf
        result=hamon_dxf.scan_dxf(dxf,limit=5000)
        assert result['total_dimensions']==expected,(dxf,result['total_dimensions'])
        assert result['geometry_supported_count']==usable,(dxf,result['geometry_supported_count'])
        assert result['declared_unit']==unit,(dxf,result['declared_unit'])
        assert result['pdf_equivalence_verified'] is False
        values=[d['usable_qualified_value'] for d in result['dimensions'] if d['usable_qualified_value'] is not None]
        for expected_value in proofs:
            assert any(math.isclose(v,expected_value,abs_tol=1e-8) for v in values),(dxf,expected_value,values)
        out.append({'drawing':base,'pdf_native_words':0,'dxf_dimensions':expected,
                    'geometry_supported':usable,'sha256':result['source_sha256']})
    path=root/'bai_a/CST-A-001_mounting_plate.dxf'
    original=hamon_dxf.scan_dxf(path)
    groups=path.read_text().splitlines()
    in_dim=False;modified=False
    for i in range(0,len(groups)-1,2):
        if groups[i].strip()=='0' and groups[i+1].strip()=='DIMENSION':in_dim=True
        elif in_dim and groups[i].strip()=='14':
            groups[i+1]=str(float(groups[i+1])+7)
            modified=True;break
    assert modified
    with tempfile.TemporaryDirectory() as td:
        changed=Path(td)/'changed.dxf';changed.write_text('\n'.join(groups)+'\n')
        mutation=hamon_dxf.scan_dxf(changed)
    assert original['source_sha256']!=mutation['source_sha256']
    assert original['dimensions'][0]['usable_qualified_value']!=mutation['dimensions'][0]['usable_qualified_value']
    print(json.dumps({'corpus':'haiha-ux/cad-sample-tests@150f4dce05a41a85d956189de8ded2cdd0d4a4d3',
      'pdf_native_words':0,'source_dimensions':62,'geometry_supported':62,
      'source_change_control':'PASS','samples':out},indent=2))
if __name__=='__main__':run(sys.argv[1])
