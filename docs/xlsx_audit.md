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

For a top-level numeric `CONVERT(A1,"mm","in")`, `conversion_audit` follows same-sheet and named-sheet references through one dependency graph shared with arithmetic. The numeric argument can contain supported arithmetic, nested conversions, and references to arithmetic producers. Every value is reconstructed from literal inputs, including intermediates with a missing or incorrect saved value. For example, `CONVERT(2*3,"m","cm")` independently yields 600, and 1 m converts to 100 cm and back to 1 m even when the workbook has saved 999 and 9.99. A single input root is identified by worksheet and cell; an expression with several roots lists them under `root_cells`.

The supported case-sensitive Excel units are `m`, `mm`, `cm`, `km`, `in`, `ft`, `yd`, `g`, `kg`, `lbm`, `sec` and `hr`. The numeric comparison uses relative tolerance 1e-12 and zero absolute tolerance. A wrong numeric cache or a conversion between incompatible dimensions with a numeric cache makes `format-guard` fail. A present text or Boolean cache contradicts a grounded numeric conversion, even if the text looks numeric. A missing numeric cache leaves that cell's comparison unverified while its reconstructed value can still supply dependents. Cycles, unsupported expressions, unknown units, inconsistent declared source units, external references and shared or array-derived results also remain unverified. The audit checks up to 256 normal conversion formulas and indexes up to 100,000 workbook cells. A pure conversion chain may span all 256 indexed cells; mixed arithmetic dependencies retain the 40-level arithmetic bound. Precision-as-displayed workbooks and results outside the supported normal finite numeric range remain unverified.

The pound-mass unit has a measured engine-specific exception. The exact SI pound is 0.45359237 kg, while the [LibreOffice 24.2 source table](https://github.com/LibreOffice/core/blob/libreoffice-24.2.7.2/scaddins/source/analysis/analysishelper.cxx) uses 2.20462291469134 pounds per kg. The verifier independently reconstructs the graph with both numeric conventions. A cache that agrees only with the historical Calc constant has status `unverified`, reason `engine_specific_lbm_scale`, and separate `expected_si` and `expected_legacy_lbm` values. This also applies to downstream arithmetic; saved intermediates remain comparison data only. The ordinary tolerance is unchanged, an exact SI match remains matched, and values contradicting both supported conventions remain mismatches. Unit dimensions and rational SI scales continue to use the exact physical definition.

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

`arithmetic_audit` reports matched, mismatched and unverified results. The comparison uses a relative tolerance of 1e-12. A proven mismatch fails the ordinary format guard. A present text or Boolean cache contradicts a grounded numeric result even when the text looks numeric. A missing numeric cache remains unverified; a proven structural unit contradiction fails independently of cache presence or type. A valid workbook can contain formulas outside this grammar; the guard reports outstanding counts under `unverified_formula_caches` for arithmetic, lookup and conversion formulas. Functions other than the supported CONVERT form (including SUM), ranges, external references, strings outside literal unit arguments, array/shared arithmetic, blank cells, cycles, near-cancelling sums/differences, subnormal values, and precision-as-displayed workbooks fall outside the check. Lookup, arithmetic and unit-conversion checks retain their own counts. Numeric arithmetic alone cannot establish a cell's physical units.

The arithmetic audit checks up to 256 formulas, 512 characters and 96 tokens per expression, and 40 nested dependencies. The workbook-wide 100,000-cell index limit also applies. Formulas that cannot be reconstructed within those limits are reported as unverified.

### Arithmetic with explicit conversion units

A supported CONVERT supplies its independently reconstructed numeric value, physical dimension, and exact rational scale relative to SI. The evaluator carries these unit frames through negation, multiplication and division. Products and ratios simplify their scales exactly: centimetres times metres divided by centimetres has the metre frame. Such an arithmetic result can feed a subsequent conversion when both its dimension and scale agree with that conversion's declared source unit. When the reconstructed frame contradicts that declaration, the guard rejects the workbook and reports the actual and declared dimensions and rational scales. A missing or unsupported dependency still yields an unverified conversion. A bare formula number can scale a known quantity in multiplication or division. Numeric cells with no declared unit and bare additive offsets leave the resulting unit annotation unknown. A separate conditional frame records the necessary unit and scale for that addition to be physically valid. Later additive operands must agree with that frame; an unlabelled term cannot erase an earlier known length or centimetre scale. These conditions are reported as `conditional_si_dimension` and `conditional_unit_scale_si`, without inventing units for the unlabelled inputs. Multiplying or dividing by an input whose unit is unknown leaves the resulting condition unknown. Numeric cache checks still apply when a unit frame is unknown.

A sum or difference of two independently grounded, incompatible dimensions has status `dimension_mismatch` and fails `format-guard`, even when its stored number agrees with Excel arithmetic. Equal dimensions with unequal scales have status `unit_scale_mismatch`: 200 in a centimetre frame plus 2 in a metre frame is rejected even when the cache correctly contains Excel's number 202. An explicit CONVERT to a common frame makes the sum eligible. Downstream dependents carry `upstream_dimension_mismatch` or `upstream_unit_scale_mismatch` and the offending worksheet cell; they receive no invented expected result. A known `si_dimension` lists the exponents of kg, m, s, A and K, and `unit_scale_si` is the exact rational scale. Units are never inferred from cell labels, formatting or numeric magnitude. Explicit CONVERT source/target dimensional conflicts are checked on the parsed expression before numeric dependency evaluation, so a missing input, a cycle, or a missing or nonnumeric cached result cannot hide that contradiction.

## Independent spreadsheet-engine qualification

`python ship/verify_xlsx_unit_graph.py --out OUTPUT_DIR --min-memory-gib 4` writes a workbook with 76 uncached formulas, invokes LibreOffice Calc, and compares the newly saved values with SymPy's independent unit definitions. It covers all 62 compatible ordered unit pairs in the supported catalogue plus arithmetic producers, nested conversions and composite unit frames. The observed Calc 24.2 replay has 71 exact-SI matches and five historical pound-scale results; the latter are explicitly unverified. The report records both independent predictions for those five cases. A separate control copy corrupts four caches, including one pound conversion, and must produce exactly four proven mismatches. The untouched source, engine-produced workbook, corrupted control and JSON report are retained separately. The hosted `workbook-oracle` job installs both independent tools, runs only for relevant source changes, and refuses to run below its memory requirement. This qualification is a synthetic verifier test; it does not establish an official PLE task score.

