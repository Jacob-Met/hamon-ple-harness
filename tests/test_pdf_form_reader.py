import io
import unittest

from pypdf import PdfWriter
from pypdf.generic import ArrayObject, DecodedStreamObject, DictionaryObject, NameObject, NumberObject

from hamon_drawing_reader import build_minimal_vector_pdf, read_vector_pdf, run_router_drawing_reader


def numbers(values):
    return ArrayObject([NumberObject(value) for value in values])


def stream(data):
    value = DecodedStreamObject()
    value.set_data(data)
    return value


def resources(entries):
    return DictionaryObject({NameObject("/XObject"): DictionaryObject(
        {NameObject(name): value for name, value in entries}
    )})


def nested_sheet():
    writer = PdfWriter()
    hidden = stream(b"BT 1 0 0 1 3 4 Tm (UNPLACED) Tj ET")
    hidden.update({NameObject("/Subtype"): NameObject("/Form"),
                   NameObject("/BBox"): numbers([0, 0, 10, 10])})
    inner = stream(b"BT 1 0 0 1 3 4 Tm (CUT B) Tj ET "
                   b"BT 1 0 0 1 13 4 Tm (OUTSIDE) Tj ET -5 5 m 20 5 l S")
    inner.update({NameObject("/Subtype"): NameObject("/Form"),
                  NameObject("/BBox"): numbers([0, 0, 10, 10]),
                  NameObject("/Matrix"): writer._add_object(numbers([2, 0, 0, 2, 1, 2]))})
    inner_ref = writer._add_object(inner)
    outer = stream(b"q 1 0 0 1 2 3 cm /Detail Do Q")
    outer.update({NameObject("/Subtype"): NameObject("/Form"),
                  NameObject("/BBox"): numbers([0, 0, 50, 50]),
                  NameObject("/Resources"): writer._add_object(resources([("/Detail", inner_ref)]))})
    page = writer.add_blank_page(width=200, height=100)
    page[NameObject("/Resources")] = resources(
        [("/Detail", writer._add_object(outer)), ("/Unplaced", writer._add_object(hidden))]
    )
    page[NameObject("/Contents")] = writer._add_object(stream(
        b"q 1 0 0 1 10 0 cm /Detail Do Q q 0 1 -1 0 100 0 cm /Detail Do Q"
    ))
    writer.add_blank_page(width=123, height=456)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


class FormReaderTests(unittest.TestCase):
    def test_placed_forms_have_local_names_matrices_and_bounds(self):
        page, blank = read_vector_pdf(nested_sheet())
        self.assertEqual([(s.text, s.x_pt, s.y_pt) for s in page.texts],
                         [("CUT B", 19, 13), ("CUT B", 87, 9)])
        self.assertEqual([(s.x0_pt, s.y0_pt, s.x1_pt, s.y1_pt) for s in page.segments],
                         [(13, 15, 33, 15), (85, 3, 85, 23)])
        self.assertEqual(blank.texts, [])

    def test_physical_line_continuation_preserves_the_printed_word(self):
        # The public MnDOT 5-297.250 sheet wraps the literal ONLY in its stream.
        content = "BT (ONL" + chr(92) + chr(13) + "Y) Tj (REAL" + r"\rBREAK" + ") Tj ET"
        page = read_vector_pdf(build_minimal_vector_pdf([
            {"width_pt": 100, "height_pt": 100, "content_stream": content}
        ]))[0]
        self.assertEqual([span.text for span in page.texts], ["ONLY", "REAL\rBREAK"])

    def test_registered_page_selection_retains_original_sheet_number(self):
        result = run_router_drawing_reader({"pdf_bytes": nested_sheet(), "page_index": 1})
        self.assertEqual(result["page_count"], 2)
        self.assertEqual([p["page_index"] for p in result["pages"]], [1])
        self.assertEqual(result["pages"][0]["texts"], [])
        with self.assertRaisesRegex(ValueError, "page_index"):
            run_router_drawing_reader({"pdf_bytes": nested_sheet(), "page_index": 2})
