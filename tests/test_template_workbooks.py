"""Role/row controls reconstructed from the public template readout, not filed designs."""
import io
from pathlib import Path
import tempfile
import unittest
import zipfile

from openpyxl import Workbook, load_workbook

import archive_checkpoint
import office_package

# The historical participant printed these two schema rows at trajectory step 9.
COLUMNS = {
    "foundation_schedule.xlsx": ("FOUNDATION SCHEDULE", "pad foundation_tag foundation_type B_in L_in thickness_in bearing_depth_in bearing_stratum fc_psi max_wcm reinforcement_ratio bar_size bar_spacing_in pier_count pier_diameter_in pier_depth_ft area_replacement_ratio settlement_in governing_constraint".split()),
    "constraint_register.xlsx": ("CONSTRAINT REGISTER", "req_id document clause requirement value unit kind applies_to resolution".split()),
}


def install_templates(root):
    owner = root / "project" / "owner"
    owner.mkdir(parents=True)
    paths = {}
    for name, (title, columns) in COLUMNS.items():
        book = Workbook()
        book.active.title = title
        book.active.append(columns)
        book.create_sheet("instructions").append(["note"])
        prefix = "AV-CR-2604_" if name.startswith("foundation") else ""
        paths[name] = owner / (prefix + Path(name).stem + "_TEMPLATE.xlsx")
        book.save(paths[name])
    return paths


def authored_bytes(template, count, change=None):
    book = load_workbook(template)
    sheet = book.active
    for i in range(count):
        sheet.append([f"row-{i}", i + 1] + [i] * (sheet.max_column - 2))
    if change:
        change(book)
    out = io.BytesIO()
    book.save(out)
    return out.getvalue()


class TemplateWorkbooks(unittest.TestCase):
    def test_observed_templates_and_atomic_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = install_templates(root)
            minimums = dict(zip(COLUMNS, (36, 1)))
            spec = office_package.discover_table_contracts(root, minimums)
            self.assertFalse(any(c.get("unavailable") for c in spec.values()))
            stage = root / "stage"
            stage.mkdir()
            originals = {n: authored_bytes(p, minimums[n]) for n, p in files.items()}
            for n, payload in originals.items():
                (stage / n).write_bytes(payload)
            target = root / "delivery.zip"
            archive_checkpoint.publish(stage, target, spec)
            good = target.read_bytes()
            name = "foundation_schedule.xlsx"
            for payload in [originals["constraint_register.xlsx"], files[name].read_bytes(),
                            authored_bytes(files[name], 35),
                            authored_bytes(files[name], 36, lambda b: setattr(b.active, "title", "Other")),
                            authored_bytes(files[name], 36, lambda b: b.active.__setitem__("C1", "other"))]:
                (stage / name).write_bytes(payload)
                with self.assertRaises(ValueError):
                    archive_checkpoint.publish(stage, target, spec)
                self.assertEqual(target.read_bytes(), good)

    def test_ambiguous_reference_and_formula_anchor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "cost_TEMPLATE.xlsx"
            book = Workbook()
            book.active.append(["id", "amount", "double"])
            book.active.append(["example", 1, "=B2*2"])
            book.save(base)
            roles = {"cost.xlsx": 1}
            contract = office_package.discover_table_contracts(root, roles)["cost.xlsx"]
            for remove in (False, True):
                body = authored_bytes(base, 1, lambda b: b.active.__setitem__("C2", 2) if remove else None)
                with zipfile.ZipFile(io.BytesIO(body)) as archive:
                    if remove:
                        with self.assertRaises(ValueError):
                            office_package.verify_table_contract(archive, contract)
                    else:
                        office_package.verify_table_contract(archive, contract)
            (root / "second_cost_template.xlsx").write_bytes(base.read_bytes())
            self.assertTrue(office_package.discover_table_contracts(root, roles)["cost.xlsx"]["unavailable"])
