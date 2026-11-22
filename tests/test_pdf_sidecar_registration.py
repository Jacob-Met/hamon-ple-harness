from hamon_callout import _register_axes

def axes():
    return [
        {"axis":i%2, "points":[(x,y),(x+n,y) if i%2==0 else (x,y+n)],"span":n}
        for i,(x,y,n) in enumerate(((1,2,12),(4,6,11),(8,3,22),(7,20,18),(6,16,9),(2,8,14)))
    ]

def test_repeated_complete_placements_are_unknown():
    source=axes()
    shifted=[dict(r,points=[(x+100,y+40) for x,y in r["points"]]) for r in source]
    assert len(_register_axes(source,source))==len(source)
    assert not _register_axes(source,source+source)
    assert not _register_axes(source,source+shifted)
    assert not _register_axes(source[:4],source)

def test_partial_sheets_preserve_missing_locus():
    source=axes()
    assert len(_register_axes(source[:-1],source))==len(source)-1

def test_declared_clockwise_rotation_preserves_native_and_display_frames():
    native = axes()
    display = []
    for original in native:
        axis = 1 - original["axis"]
        points = sorted(((300-y, 20+x) for x,y in original["points"]), key=lambda p: p[axis])
        display.append(dict(original, axis=axis, points=points))
    joined = _register_axes(display, native, 90)
    assert [(j["pdf"], j["native"]) for j in joined] == list(zip(display, native))
    assert not _register_axes(display, native)
    assert not _register_axes(display, native + native, 90)
