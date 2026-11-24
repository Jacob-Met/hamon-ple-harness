# Detention-basin level-pool routing

Use `hamon_conservation_cad_verifier` with `mode=civil_geotech` and `spec.calculation=level_pool_routing`. Supply `initial_stage_m`, a `rating_curve` of rows with `stage_m`, `storage_m3` and `outflow_m3_s`, and an `inflow_hydrograph` of rows with `time_s` and `inflow_m3_s`.

The reservoir has a horizontal water surface and one fixed total outlet rating. Stage and storage increase strictly. Outflow can remain constant or increase with stage. The initial stage can lie between rating rows. Each sampled inflow interval may have a different duration. For each interval the solver linearly interpolates storage and total outflow on the common stage coordinate, then solves the Modified Puls volume balance. All supplied stages share a datum. The solver declines times for which no in-range rating satisfies the balance. Specify zero inflow samples explicitly to evaluate the recession after a storm.

The hydrograph reports stage, storage and outflow at the supplied times. The peak and lag refer to these samples, with the first occurrence chosen for tied peaks. The returned continuity and rating residuals replay the reported values. Certification refers to this discrete balance; peak accuracy also depends on the hydrograph time step and outlet data. A blocked outlet needs its own total rating. Tailwater-dependent outlets, gate operations, seepage and spatially varying water surfaces need additional modeling.

The reproducible external benchmark uses the Los Angeles County Hydrology Manual (January 2006), Chapter 8, Figure 8.2 and Tables 8.1 and 8.2. The 220 cfs peak inflow yields about 200.6 cfs peak outflow at 70 minutes. All 43 published outflow ordinates, including the 220-minute recession after inflow ceases, are checked at the tables' 0.1 cfs reporting precision. The manual supplies a worked hydrologic example.

- https://planning.lacity.gov/eir/CrossroadsHwd/deir/files/references/G12.pdf
- https://www.hec.usace.army.mil/confluence/hmsdocs/hmstrm/reservoir-modeling/reservoir-modeling-concepts-and-equations
