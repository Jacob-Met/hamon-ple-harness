import math
from hamon_ifc_spatial_clash import run_router_ifc_clash
from test_wired_router_e2e import SAMPLE_IFC


def _source(second=False):
    # Reuse the maintained beam/duct case, making the duct a subtraction volume.
    text = SAMPLE_IFC.replace("IFCFLOWSEGMENT(", "IFCOPENINGELEMENT(")
    marker = "#38=IFCRELVOIDSELEMENT('rel',$,$,$,#24,#37);"
    if second:
        marker += """
#39=IFCOPENINGELEMENT('second',$,'Second',$,$,#32,#36,$,.OPENING.);
#40=IFCRELVOIDSELEMENT('another',$,$,$,#24,#39);
"""
    return text.replace("ENDSEC;\nEND-ISO",marker+"\nENDSEC;\nEND-ISO")


def test_overlapping_subtractions_apply_once():
    for duplicate in (False,True):
        out=run_router_ifc_clash({"ifc_text":_source(duplicate)})
        assert out["ok"] and out["solid_count"]==1 and out["pair_count"]==0
        assert math.isclose(out["total_solid_volume_m3"], 6-0.6*0.6*1.5)


def test_unresolved_void_never_returns_gross_quantity():
    ambiguous=_source().replace("'SweptSolid',(#34)","'SweptSolid',(#34,#9998)")
    out=run_router_ifc_clash({"ifc_text":ambiguous})
    assert out["status"]=="unresolved_opening" and out["total_solid_volume_m3"] is None
