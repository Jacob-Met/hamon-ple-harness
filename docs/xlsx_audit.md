# Auditing workbook formulas

Run `python3 /tmp/hamon-engineering-toolkit.py xlsx-audit FILE.xlsx` inside the provisioned task container. The JSON lists worksheets in workbook order with their declared names, package paths and visibility. Both Strict and Transitional SpreadsheetML workbooks are supported. Declared non-worksheet sheets appear under `other_sheets`.

For each worksheet, `formulas_without_cached_values` counts formulas with no value element and formulas with an empty non-string value. Zero and Boolean false count as present values. An explicitly present empty string cache is counted separately under `formulas_with_empty_string_cache`. The formula total includes shared formula followers. Error counts cover all error-typed cells, and the report includes up to 50 examples per worksheet.

The audit reads the saved package. A cached result can be stale even when it is present. Recalculation requires a spreadsheet engine.

For direct numeric VLOOKUP and HLOOKUP formulas with explicit exact match, the
`lookup_audit` section compares the saved result with the first matching row or
column of a literal table. It records the source sheet, source cell, expected
number and cached number. Only direct cell-reference keys and A1 ranges in
this workbook qualify. A missing or nonnumeric cache, an ambiguous key before
the match, or a formula-derived dependency yields `unverified`. Lookup
counts cover top-level VLOOKUP, HLOOKUP and INDEX calls. The verifier checks up to
256 formulas and 2,048 key cells per formula. Above 100,000 workbook cells it
keeps the inventory and reports all lookups as unverified.

Array-formula and data-table result ranges are formula-derived even when a follower has no formula element. Such cells cannot qualify as literal lookup inputs. Missing, malformed, out-of-grid or inconsistent master ranges, and more than 128 ranges on one sheet, make dependencies from that sheet unverified. Valid disjoint ranges and unrelated sheets retain their literal inputs. Range containment never expands the worksheet grid.

Shared lookup formulas use their worksheet-local master and declared result range. The audit translates relative rows and columns for each saved follower and preserves absolute references. It checks the same literal-value and first-match rules as an ordinary lookup. Unresolved followers appear in each worksheet’s `unresolved_shared_formula_followers` count when a master is missing, ambiguous, or the sheet exceeds 1,024 shared groups.

For a top-level numeric `CONVERT(A1,"mm","in")`, `conversion_audit` follows same-sheet and named-sheet references to a numeric literal. It reconstructs each intermediate conversion, including cells with a missing or incorrect saved value. Each dependent formula must declare the unit produced by its predecessor. The result identifies the original worksheet, cell and number of conversions. For example, 1 m converts to 100 cm and back to 1 m even when the workbook has saved 999 and 9.99.

The supported case-sensitive Excel units are `m`, `mm`, `cm`, `km`, `in`, `ft`, `yd`, `g`, `kg`, `lbm`, `sec` and `hr`. The numeric comparison uses relative tolerance 1e-12 and zero absolute tolerance. A wrong numeric cache or a conversion between incompatible dimensions with a numeric cache makes `format-guard` fail. A missing numeric cache leaves that cell's comparison unverified. Cycles, unsupported expressions, unknown units, inconsistent source units, external references and shared or array-derived results also remain unverified. The audit checks up to 256 normal conversion formulas and indexes up to 100,000 workbook cells.

For vertical `INDEX(return_range,MATCH(key_cell,key_range,0))`, the two
single-column ranges may start at different rows or reside on different
local sheets. Both must contain the same number of rows. The numeric key,
all keys through the first exact match, and the selected return cell must
come from workbook literals. The first duplicate wins. The report identifies
the selected return cell and compares its numeric value to the cache.
Approximate or omitted match modes, external references, unknown keys and
unsupported INDEX forms remain unverified. The ordinary `format-guard`
rejects a proven cache mismatch.

## Arithmetic across worksheets

For ordinary, full-precision formulas, the audit reconstructs finite numeric literals, A1 cell references, parentheses, unary signs, and binary addition, subtraction, multiplication and division. References may use absolute or mixed anchors and a local or explicitly named worksheet. Each supported reference is followed to its numeric inputs. A stored intermediate result is never used as an input to the calculation.

`arithmetic_audit` reports matched, mismatched and unverified results. The comparison uses a relative tolerance of 1e-12. A proven mismatch fails the ordinary format guard. A present text or Boolean cache contradicts a grounded numeric result even when the text looks numeric. A missing cache remains unverified. A valid workbook can contain formulas outside this grammar; the guard reports outstanding counts under `unverified_formula_caches` for arithmetic, lookup and conversion formulas. Functions (including SUM), ranges, external references, strings, array/shared arithmetic, blank cells, cycles, near-cancelling sums/differences, subnormal values, and precision-as-displayed workbooks fall outside the check. Existing lookup and unit-conversion checks retain their own counts. Numeric arithmetic alone cannot establish a cell's physical units.

The arithmetic audit checks up to 256 formulas, 512 characters and 96 tokens per expression, and 40 nested dependencies. The workbook-wide 100,000-cell index limit also applies. Formulas that cannot be reconstructed within those limits are reported as unverified.
