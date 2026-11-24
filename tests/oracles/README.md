# Independent Bishop reference

`xslope_bishop_source.py` is the unchanged `test/indep_bishop_check.py` from
[njones61/xslope](https://github.com/njones61/xslope/blob/28f16877a9c204ed421d3e2d6e353609ec8d0c8b/test/indep_bishop_check.py),
commit `28f16877a9c204ed421d3e2d6e353609ec8d0c8b`.
It is distributed under Apache 2.0. `XSLOPE_LICENSE.txt` and `XSLOPE_NOTICE`
reproduce the upstream license and notice. Copyright 2025 Norman L. Jones.

The tests call only its `bishop_independent` function, with a small ndarray
column adapter. No reference numerical calculations have been modified.
The other upstream diagnostic functions remain present but are not invoked.

The geometric test inputs are derived from the same commit's
`docs/inputs/slope/xslope_simple1.xlsx`. Source cells and explicit changes to the
control material are recorded in `physical_profile` in `test_bishop_slope.py`.
Only the same slice table is used for numerical comparison. The test does not
claim agreement with a differently discretized published example.
