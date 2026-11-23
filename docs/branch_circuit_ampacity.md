# Ordinary branch-circuit conductor ampacity

`ampacity_derating` checks the two conductor-sizing conditions in **NFPA 70, 2017 Edition, §210.19(A)(1)(a) and (b)**, including its explicitly listed 100%-rated assembly exception. Section 110.14(C) governs the temperature limits at conductor and equipment terminations and allows higher-temperature insulation ratings to be used for adjustment and correction. This is a supplied-rating calculation for ordinary branch circuits covered by that clause.

| Input | Meaning |
| --- | --- |
| `I_continuous_A`, `I_noncontinuous_A` | Nonnegative continuous and noncontinuous loads. Noncontinuous load defaults to zero. |
| `base_ampacity_A` | Unadjusted ampacity at the conductor insulation rating applicable to the chosen installation |
| `termination_ampacity_A` | Optional unadjusted ampacity of **the selected conductor** at the lowest applicable termination temperature rating. With a single supplied rating, `base_ampacity_A` serves as the common insulation/termination rating. |
| `temp_factor`, `bundling_factor` | Positive ambient correction and current-carrying-conductor adjustment of the insulation rating. A cool ambient may give a temperature factor above one; bundling is at most one. |
| `listed_100_percent_assembly` | Boolean, default false. True asserts that the complete assembly, including its protecting overcurrent device, is listed for 100% operation. |

First compare both unadjusted ratings against 125% of the continuous load plus all noncontinuous load. The listed assembly uses 100% of both loads instead. Separately compare the adjusted insulation ampacity against the actual load, continuous plus noncontinuous. Both checks must pass. For example, 80 A continuous needs an unadjusted 100 A for an ordinary assembly. A 100 A common rating corrected to 91 A still carries the actual 80 A. A 95 A common rating corrected to 102.6 A still lacks the required unadjusted rating.

`I_derated_allowable_A` reports the adjusted insulation ampacity. `I_installed_allowable_A` also caps it at the termination rating. `I_design_required_A` reports the applicable unadjusted sizing demand. `unadjusted_sizing_ok` and `adjusted_load_ok` show the two checks. `I_required_table_A` inverts both inequalities to give the required insulation rating, subject to the termination requirement.

The caller supplies the applicable tabular values, temperature limits, adjustment factors and any listing. OCPD selection, special-purpose loads, conductor lookup and complete installation compliance require their own checks. The edition named here does not determine which edition a jurisdiction has adopted.

[NFPA 70 (2017 Edition), published Code text, City of Bisbee copy](https://www.bisbeeaz.gov/DocumentCenter/View/11395/2017-NEC-Code-2?bidId=) · [Schneider Electric's 2017 Code conductor ampacity bulletin](https://productinfo.se.com/na-std-ref/5c0fee4b347bdf0001de4d55/Conductor%20Ampacity%20Tables/English/Data%20Bulletin%20-%20Conductor%20Ampacity%20Tables%20%28bookmap%29_0000264293.ditamap/%24/NEC210.19ConductorsMinimumAmpacityA-F20BBAC1)
