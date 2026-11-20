"""Cold raster/source-coordinate parity on pinned public engineering PDFs.

These drawings are exposed development controls, never organizer holdouts.
The oracle is an independent full-page native render, compared as decoded RGB.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from test_pdf_window import read_rgb


CORPUS = [
    ("bai_a/CST-A-001_mounting_plate.pdf", "803ad502e5fa982df1e07a43aa90d15b0423339d38e28da777c10a59c6f86fd0"),
    ("bai_b/CST-B-001_studio_plan.pdf", "a9c0ac00ddbbd0b9d2810ce2b16740046d6cdf56e56c01de4024289d7370bf59"),
    ("bai_c/CST-C-001_step_bearing.pdf", "09c463c6b56f4f1dc4274d1fbac2dd0365ef0cf742eb72c957334b49ecb3b04e"),
]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    dpi, region = 120, [40.5, 150.25, 645.75, 700.5]
    for relative, expected_hash in CORPUS:
        source = args.corpus / relative
        before = source.read_bytes()
        assert hashlib.sha256(before).hexdigest() == expected_hash, relative
        stem = source.stem
        full = args.output_dir / (stem + "-reference")
        subprocess.run(["pdftocairo", "-png", "-cropbox", "-singlefile", "-f", "1", "-l", "1",
                        "-r", str(dpi), str(source), str(full)],
                       capture_output=True, check=True, timeout=30)
        overview_path = args.output_dir / (stem + "-overview.png")
        window_path = args.output_dir / (stem + "-window.png")
        evidence = []
        for target, selected_region in [(overview_path, None), (window_path, region)]:
            command = [sys.executable, str(ROOT / "engineering_toolkit.py"), "pdf-window", str(source),
                       "--page", "1", "--dpi", str(dpi), "--output", str(target)]
            if selected_region is not None:
                command.extend(["--region", *map(str, selected_region)])
            start = time.monotonic()
            proc = subprocess.run(command, capture_output=True, text=True, timeout=90)
            assert proc.returncode == 0, proc.stdout + proc.stderr
            record = json.loads(proc.stdout)
            assert record["ok"] and record["source_pdf_sha256"] == expected_hash
            assert not record["physical_drawing_scale_verified"] and not record["ocr_performed"]
            record["observed_wall_ms"] = (time.monotonic() - start) * 1000
            assert target.stat().st_size == record["image"]["bytes"] <= 8_000_000
            assert hashlib.sha256(target.read_bytes()).hexdigest() == record["image"]["sha256"]
            evidence.append(record)
        reference_width, reference_height, reference = read_rgb(full.with_suffix(".png"))
        overview_width, overview_height, overview = read_rgb(overview_path)
        assert (overview_width, overview_height) == (reference_width, reference_height)
        assert overview == reference, (relative, "full decoded RGB differs")
        width, height, window = read_rgb(window_path)
        left, top = math.floor(region[0] * dpi / 72), math.floor(region[1] * dpi / 72)
        right, bottom = math.ceil(region[2] * dpi / 72), math.ceil(region[3] * dpi / 72)
        assert (width, height) == (right - left, bottom - top)
        expected = [row[left * 3:right * 3] for row in reference[top:bottom]]
        assert window == expected, (relative, "window decoded RGB differs")
        assert source.read_bytes() == before, (relative, "source changed")
        rows.append({"file": relative, "source_sha256": expected_hash,
                     "exact_reference_rgb": True, "exact_crop_rgb": True,
                     "source_unchanged": True, "evidence": evidence})
    receipt = {"schema": "hamon.ple.pdf-window-public-corpus.v1",
               "corpus_commit": "150f4dce05a41a85d956189de8ded2cdd0d4a4d3",
               "exposure": "previously exposed public development controls",
               "official_score": None, "ocr_accuracy_measured": False,
               "dpi": dpi, "region_svg": region, "passed": len(rows), "rows": rows}
    encoded = (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode()
    path = args.output_dir / "receipt.json"
    path.write_bytes(encoded)
    print(json.dumps({"passed": len(rows), "full_and_crop_rgb_exact": True,
                      "receipt": str(path), "sha256": hashlib.sha256(encoded).hexdigest(),
                      "official_score": None}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
