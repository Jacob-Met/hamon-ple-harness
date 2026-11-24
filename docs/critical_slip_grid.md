# Critical circular slip search

The registered civil engineering verifier accepts `calculation=critical_slip_circle_grid`. It tests the Cartesian product of three explicit coordinate lists and reports the lowest factor with verified equilibrium and compressive basal contact. The output includes the circle, sliding span, and slice inputs for that result. It also counts geometry exclusions, contact exclusions and unresolved equilibria. The search makes no continuous optimality or site-safety determination.

Supply a dry homogeneous soil, a piecewise-linear ground profile with strictly increasing x, and a horizontal basal cutoff. Lengths are in metres, effective cohesion is in Pa, effective friction angle is in degrees and bulk unit weight is in N/m³. Specify the downhill sliding direction. Trial masses need two distinct lower-circle intersections with the ground and cannot penetrate below the basal cutoff. Slice weights use two-point Gaussian quadrature of vertical soil thickness, split at every ground-profile vertex. Base inclinations use the midpoint tangent. The existing Bishop or uncorrected simplified Janbu verifier independently reconstructs force equilibrium for each eligible mass.

A positive `dry_tension_crack_depth_m` truncates the rear of each sliding mass at the first point, from the uphill endpoint, where ground-to-circle vertical depth equals the prescribed depth. The new rear face transmits no force. A circle unable to reach that crack depth is excluded. The user supplies the depth. Wet cracks, stratified ground, groundwater, surcharge and seismic loads are outside this model.

## Published external comparison

ACADS problem 1 in the [Rocscience Slide2 verification manual](https://static.rocscience.cloud/assets/verification-and-theory/Slide2/Slide_SlopeStabilityVerification.pdf) gives this dry slope, with floor elevation 20 m.

| Ground x (m) | 20 | 30 | 50 | 70 |
| --- | ---: | ---: | ---: | ---: |
| Elevation (m) | 25 | 25 | 35 | 35 |

For cohesion 3 kPa, friction angle 19.6° and unit weight 20 kN/m³, Figure 1.2 reports simplified Bishop 0.987, center (29.104, 55.527) m, radius 30.527 m, and slip endpoints at x = 30.027 and 51.700 m. The receiver independently searches 735 circles from the geometry alone. A separate calculation evaluates the rounded published circle and checks both intersections. The manual's 0.990 value uses **corrected** Janbu. External comparison to that number is inapplicable to this implementation.

Run `python -m unittest discover -s tests -p test_critical_slip_grid.py` to exercise the geometry, refinement, mirrored slide, changed cohesion, prescribed crack, and invalid inputs. After installing the repository's locked Pi dependencies, `python ship/verify_critical_slip_grid.py` checks both methods via the live local registered-tool path and a loopback model fixture.
