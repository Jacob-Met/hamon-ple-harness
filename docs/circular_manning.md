# Uniform gravity flow in a circular conduit

Use `manning_channel` with `cross_section="circular"`, `diameter_m`, `Q_m3_s`, `manning_n` and `slope_S`. The route uses SI Manning friction with a uniform positive bed slope. For the reported Froude number, `gravity_m_s2` defaults to 9.80665. The previous trapezoidal geometry remains available through `bottom_width_m` and `side_slope_z`.

The wetted perimeter and flow area grow at different rates as water approaches the crown. Maximum uniform discharge occurs near 0.93818 times the diameter. A greater depth reduces discharge until the conduit becomes full. This gives two free-surface normal depths for discharges strictly between the full-section and maximum capacities. Both appear in `normal_depth_solutions`, ordered from shallower to deeper. Each includes depth, crown clearance, exposed water width, mean velocity and Froude regime. A tiny air gap may remain measurable in the clearance even when depth rounds to the diameter. [Derivation of the maximum-discharge depth](https://ponce.sdsu.edu/what_is_the_relative_depth.html)

| Requested discharge | Free-surface solutions |
| --- | --- |
| Below full-section flow | One on the increasing branch |
| At full-section flow | One on the increasing branch, plus a separately reported full-section boundary |
| Between full-section flow and maximum | One on each branch |
| At maximum | One at the peak |
| Above maximum | No uniform free-surface solution |

The reported binary64 capacities designate the two physical endpoints. A full-section boundary carries no pressure-flow or free-surface Froude certificate. The computation uses circular-segment geometry for the two branches and replays conveyance by integrating horizontal chord widths from the smaller reported vertical distance.
