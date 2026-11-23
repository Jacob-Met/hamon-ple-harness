"""Run the real provisioner and copper CLI across a loopback PLE WebSocket session."""
import argparse
import asyncio
import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
import tempfile

from websockets.asyncio.client import connect
from websockets.asyncio.server import serve


def check(ok, why):
    if not ok:
        raise RuntimeError(why)


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def due_inputs(raw, cases):
    sch = raw / "due-0.sch"
    check(digest(sch.read_bytes()) == "60f449078bea22dc298a0cad5b9733affb1214f6052f778e54babb08531e9734",
          "Original Due schematic differs")
    original = raw / "due-1.brd"
    check(digest(original.read_bytes()) == "fd7cc5f9a14623ff62a7555c826373fe8f842cca400f05ec4f7b4c5589e1780a",
          "Original Due board differs")
    boards = [("original", "witnessed", original)] + [(k, s, Path(b).resolve(strict=True)) for k, s, b in cases]
    return boards, {label: sch for label, _, _ in boards}


def stadium_inputs():
    """Supply two source-authored task files for each route in evaluator cwd."""
    from xml.etree import ElementTree as ET
    from verify_eagle_board_domain import rectangle, small_fixture, wire

    # Radius 1 mm around the horizontal spine joining (-1,0) to (1,0).
    points = [(-1,-1), (1,-1), (2,0), (1,1), (-1,1), (-2,0)]
    hole = [wire(a, b, curve=angle)
            for a, b, angle in zip(points, points[1:] + points[:1], [0,90,90,0,90,90])]
    courses = [
        ("clear", "witnessed", ("1.65",".95"), ("1.95",".65")),
        ("clipped", "unresolved", ("1.35","1.03"), ("2.03",".35")),
    ]
    # On y=2.6-x the clear stroke lies >.006 mm beyond the right cap.
    # The clipped stroke has clear endpoints, but its midpoint relative to
    # the cap is (.69,.69), radius ~.976 mm: the whole .012-mm width is cut.
    boards, schematics, files = [], {}, {}
    for label, status, start, end in courses:
        pads = [(name, *point, 1) for name, point in zip(("L", "R"), (start, end))]
        board, schematic = small_fixture(pads, rectangle(-4,4,-4,4) + hole)
        for pad in board.findall(".//smd"):
            pad.attrib.update(dx=".02", dy=".02")
        board.find(".//signals/signal").append(wire(start, end, layer=1, width=".012"))
        board_path, sch_path = Path(label + ".brd"), Path(label + ".sch")
        files[board_path] = ET.tostring(board, encoding="utf-8", xml_declaration=True)
        files[sch_path] = ET.tostring(schematic, encoding="utf-8")
        boards.append((label, status, board_path))
        schematics[label] = sch_path
    return boards, schematics, files


async def qualify(repo, raw, expected_module, cases, material_case, *, stadium=False):
    sys.path.insert(0, str(repo))
    import engineering_intake
    import taskbridge

    check(Path(engineering_intake.__file__).resolve().parent == repo, "Wrong provisioner revision")
    check(Path(taskbridge.__file__).resolve().parent == repo, "Wrong protocol revision")
    task_inputs = {}
    if stadium:
        boards, schematics, task_inputs = stadium_inputs()
        material_case = "clear"
        net, expected_contacts, expected_nets = "A", 2, 1
    else:
        boards, schematics = due_inputs(raw, cases)
        net, expected_contacts, expected_nets = "N$3", 605, 163
    check(len({k for k, _, _ in boards}) == len(boards), "Expected unique case labels")
    check(all(s in ("witnessed", "unresolved") for _, s, _ in boards), "Unexpected case verdict")
    check(any(label == material_case and status == "witnessed" for label, status, _ in boards),
          "Expected a named finite-material positive control")
    check(any(status == "unresolved" for _, status, _ in boards), "Expected a severed negative control")
    pins = {label: digest(task_inputs[board] if stadium else board.read_bytes()) for label, _, board in boards}
    if stadium:
        check(pins == {
            "clear": "da975996d2e4148e02ca69f64a3b218a73bcf83bcdea3ae3cc92b6c89fa74a2b",
            "clipped": "a503c2e8b68ebd08c325b4e550a9f224e57a2b9a62aeed91501110bcffc17fe3",
        }, "Frozen stadium fixture bytes differ")
    check(len(set(pins.values())) == len(boards), "Control boards must have distinct bytes")
    tool = "/tmp/hamon-engineering-toolkit.py"
    completed, written = [], {}
    runs = [(label, status, board, ()) for label, status, board in boards]
    runs += [(label, "unresolved", board, ("-S",))
             for label, _, board in boards if label in {"original", material_case}]
    commands = [shlex.join(["python3", *flags, tool, "eagle-netlist", str(schematics[label]), "--board", str(board),
                            "--routed-net", net]) for label, _, board, flags in runs]

    async def handler(session):
        advertised = await engineering_intake.provision_engineering_cli(session)
        check("eagle-netlist" in advertised, "Normal provisioner did not advertise EAGLE")
        for command, (label, wanted, board, flags) in zip(commands, runs):
            reply = await session.exec(command, timeout_sec=50)
            check(reply["exit_code"] == 0 and not reply["truncated"], "Task CLI failed: " + reply["output"][-1400:])
            result = json.loads(reply["output"])
            routed, flat = result["routed_continuity"], result["board_comparison"]
            pin = pins[label]
            check(routed["board_sha256"] == pin == flat["board_sha256"], "Wrong board " + label)
            check(flat["agreement"] and flat["exact_contacts"] == expected_contacts and len(flat["exact_nets"]) == expected_nets,
                  "Unexpected flat membership " + label)
            check(routed["status"] == wanted and bool(routed["witnesses"]) == (wanted == "witnessed"),
                  "Physical result differs on " + label + ": " + str(routed["status"]))
            reason = routed.get("material", {}).get("reason")
            domain = routed.get("board_domain", {})
            if stadium and not flags:
                check(domain.get("status") == "bounded" and domain.get("hole_wire_count") == 6
                      and domain.get("model") == "outer_outline_with_stadium_dimension_hole",
                      "The expected six-wire stadium was not resolved: " + label)
            if flags:
                check(routed.get("board_domain", {}).get("status") == "unresolved",
                      "Declared board outline must not become unbounded when its backend is missing")
                check(reason == "finite_width_geometry_dependency_unavailable",
                      "Missing backend must remain explicitly unresolved")
            elif label == material_case:
                mode = routed["material"]
                check(mode["model"] == "finite_outer_layer_regions" and mode["in_plane_clearance_nm"] == 1,
                      "Small hole used an unexpected material model")
                components = [step["material"] for witness in routed["witnesses"] for step in witness["steps"]
                              if step.get("material", {}).get("kind") == "component"]
                check(components and all(c.get("path_nm") for c in components), "Missing in-plane witness path")
            completed.append({"case": label, "site_enabled": not bool(flags), "status": wanted,
                              "material_reason": reason, "board_sha256": pin,
                              "contacts": flat["exact_contacts"], "nets": len(flat["exact_nets"]),
                              **({"board_domain": domain["status"], "witness_count": len(routed["witnesses"])} if stadium else {})})
        return "Verified provisioned EAGLE CLI"

    async def accept(ws):
        await taskbridge._session(ws, handler)

    with tempfile.TemporaryDirectory(prefix="ple-eagle-task-") as isolated:
        task = Path(isolated)
        for filename, payload in task_inputs.items():
            (task / filename).write_bytes(payload)
        env = {k: os.environ[k] for k in ("PATH", "LANG", "LC_ALL") if k in os.environ}
        python = [sys.executable, "-E", "-s", "-B"]
        runtime = subprocess.run([*python, "-c", "import shapely; print(shapely.__version__)"],
                                 cwd=task, env=env, capture_output=True, text=True, timeout=10, check=True)
        check(runtime.stdout.strip() == "2.1.2", "Copied task CLI requires Shapely 2.1.2")
        no_site = subprocess.run([*python, "-S", "-c",
                                  "from importlib.util import find_spec; print(find_spec('shapely') is None)"],
                                 cwd=task, env=env, capture_output=True, text=True, timeout=10, check=True)
        check(no_site.stdout.strip() == "True", "No-site interpreter unexpectedly finds Shapely")
        async with serve(accept, "127.0.0.1", 0, max_size=taskbridge.MAX_FRAME, ping_interval=None) as server:
            port = server.sockets[0].getsockname()[1]
            async with connect(f"ws://127.0.0.1:{port}", max_size=taskbridge.MAX_FRAME) as socket:
                await socket.send(json.dumps({"type": "task", "instruction": "Verify supplied stadium copper routes" if stadium else "Verify supplied Due copper routes",
                                              "workdir": str(task), "limits": {"agent_timeout_sec": 120},
                                              "mcp_servers": []}))
                while True:
                    frame = json.loads(await asyncio.wait_for(socket.recv(), 60))
                    kind = frame["type"]
                    if kind == "done":
                        check(frame["final_message"] == "Verified provisioned EAGLE CLI", "Unexpected final message")
                        break
                    if kind == "error":
                        raise RuntimeError(frame["message"])
                    reply = {"type": "result", "id": frame["id"], "ok": True}
                    if kind == "write_file":
                        remote = PurePosixPath(frame["path"])
                        check(remote.parent == PurePosixPath("/tmp") and remote.suffix == ".py",
                              "Unexpected provisioned destination")
                        destination = task / remote.name
                        with destination.open("ab" if frame.get("append") else "wb") as out:
                            out.write(base64.b64decode(frame["content_b64"], validate=True))
                        reply["size"] = destination.stat().st_size
                        source = "engineering_toolkit.py" if remote.name == "hamon-engineering-toolkit.py" else remote.name
                        written[source] = destination
                    elif kind == "exec":
                        check(frame["command"] in commands, "Unexpected command")
                        argv = shlex.split(frame["command"])
                        cli_index = argv.index(tool)
                        flags = argv[1:cli_index]
                        check(argv[0] == "python3" and flags in ([], ["-S"]), "Unexpected task executable")
                        proc = subprocess.run([*python, *flags, str(task / PurePosixPath(tool).name), *argv[cli_index + 1:]],
                                              cwd=task, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                              text=True, timeout=50)
                        reply.update(exit_code=proc.returncode, output=proc.stdout, truncated=False, timed_out=False)
                    else:
                        raise RuntimeError("Unexpected task frame: " + kind)
                    await socket.send(json.dumps(reply))
        for name, path in written.items():
            check(path.read_bytes() == (repo / name).read_bytes(), "Provisioned bytes differ for " + name)
        check(expected_module in written, "New material module missing from normal provisioning")
        stadium_modules = {}
        if stadium:
            for name in ("hamon_eagle_copper.py", "hamon_eagle_material.py"):
                check(name in written, "Missing provisioned stadium module: " + name)
                stadium_modules[name] = digest(written[name].read_bytes())
            for filename, payload in task_inputs.items():
                check((task / filename).read_bytes() == payload, "Task input changed: " + str(filename))
        check(len(completed) == len(runs), "Incomplete routed-net course")
        return {"head": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
                "child_shapely": runtime.stdout.strip(), "no_site_shapely_available": False,
                "material_case": material_case,
                "transport": "loopback WebSocket, official taskbridge.Session, local evaluator file/exec receiver",
                "cases": completed, "module": expected_module, "module_sha256": digest(written[expected_module].read_bytes()),
                "provisioned_module_count": len(written),
                **({"stadium_module_sha256": stadium_modules} if stadium else {})}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--stadium", action="store_true", help="Run the self-contained split-cap task protocol course")
    parser.add_argument("--expect-module", default="hamon_eagle_material.py")
    parser.add_argument("--material-case")
    parser.add_argument("--case", nargs=3, action="append", default=[], metavar=("LABEL", "STATUS", "DUE_BOARD"))
    args = parser.parse_args()
    if args.stadium:
        if args.raw or args.material_case or args.case:
            parser.error("--stadium cannot be combined with Due case arguments")
    elif not args.raw or not args.material_case:
        parser.error("the Due course requires --raw and --material-case")
    print(json.dumps(asyncio.run(qualify(args.repo.resolve(strict=True),
                                         args.raw.resolve(strict=True) if args.raw else None,
                                         args.expect_module, args.case, args.material_case, stadium=args.stadium)), indent=2))
