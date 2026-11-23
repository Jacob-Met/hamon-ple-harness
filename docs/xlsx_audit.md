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
counts cover top-level VLOOKUP and HLOOKUP calls. The verifier checks up to
256 formulas and 2,048 key cells per formula. Above 100,000 workbook cells it
keeps the inventory and reports all lookups as unverified.

Array-formula and data-table result ranges are formula-derived even when a follower has no formula element. Such cells cannot qualify as literal lookup inputs. Missing, malformed, out-of-grid or inconsistent master ranges, and more than 128 ranges on one sheet, make dependencies from that sheet unverified. Valid disjoint ranges and unrelated sheets retain their literal inputs. Range containment never expands the worksheet grid.

Shared lookup formulas use their worksheet-local master and declared result range. The audit translates relative rows and columns for each saved follower and preserves absolute references. It checks the same literal-value and first-match rules as an ordinary lookup. Unresolved followers appear in each worksheet’s `unresolved_shared_formula_followers` count when a master is missing, ambiguous, or the sheet exceeds 1,024 shared groups.

For a top-level numeric `CONVERT(A1,"mm","in")`, `conversion_audit` checks the saved result against a numeric literal on the same worksheet or a named worksheet, including quoted names and absolute cell references. The supported case-sensitive Excel units are `m`, `mm`, `cm`, `km`, `in`, `ft`, `yd`, `g`, `kg`, `lbm`, `sec` and `hr`. Length, mass and time are separate dimensions. The numeric comparison uses relative tolerance 1e-12 and zero absolute tolerance. A wrong numeric cache or an incompatible pair of supported units with a numeric cache makes `format-guard` fail. Unsupported units, external references, derived source cells, shared formulas and missing numeric caches remain unverified. The conversion audit checks up to 256 formulas using the same 100,000-cell literal index as the lookup audit.
