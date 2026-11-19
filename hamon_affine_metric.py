"""Empirical 2-D physical metric from independently dimensioned PDF vectors.

For an image-space displacement v=(dx,dy), the squared physical length is
v.T @ G @ v. The SPD tensor G captures unknown affine anisotropy and shear,
unlike one scalar drawing scale. Four or more directional measurements enable
a genuinely held-out leave-one-dimension-out falsification of G. This module
does NOT establish that a text label belongs to a dimension line: the caller
must source-qualify that correspondence before fitting.
"""
from __future__ import annotations

import math


def _fit_three(rows):
    """Fit normalized squared direction cosines by bounded Gaussian elimination."""
    gram = [[0.0]*3 for _ in range(3)]
    rhs = [0.0]*3
    for row in rows:
        ux, uy = row["unit_direction"]
        features = (ux*ux, 2.0*ux*uy, uy*uy)
        observed = (row["physical_m"]/row["svg_units"])**2
        for i in range(3):
            rhs[i] += features[i]*observed
            for j in range(3):
                gram[i][j] += features[i]*features[j]
    scale = max(abs(x) for line in gram for x in line)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("affine reference directions have no information")
    matrix = [[gram[i][j]/scale for j in range(3)] + [rhs[i]/scale]
              for i in range(3)]
    for col in range(3):
        pivot = max(range(col,3), key=lambda k:abs(matrix[k][col]))
        # Fail closed on near-dependent direction families rather than
        # computing an unstable off-axis answer from a singular Gram matrix.
        if abs(matrix[pivot][col]) < 0.001:
            raise ValueError("affine reference directions are ill-conditioned")
        matrix[col],matrix[pivot] = matrix[pivot],matrix[col]
        divider = matrix[col][col]
        for j in range(col,4):
            matrix[col][j] /= divider
        for i in range(3):
            if i == col:
                continue
            ratio = matrix[i][col]
            for j in range(col,4):
                matrix[i][j] -= ratio*matrix[col][j]
    gx,gxy,gy = (matrix[i][3] for i in range(3))
    if not all(math.isfinite(x) for x in (gx,gxy,gy)):
        raise ValueError("affine metric has non-finite coefficients")
    eig_radius = math.hypot(gx-gy,2*gxy)
    lo,hi = (gx+gy-eig_radius)/2,(gx+gy+eig_radius)/2
    if lo <= 0 or hi <= 0:
        raise ValueError("dimension references imply non-positive physical metric")
    condition = math.sqrt(hi/lo)
    if not math.isfinite(condition) or condition>10:
        raise ValueError("affine metric distortion exceeds 10x; inspect labels and geometry")
    return (gx,gxy,gy),condition


def _length_metres(metric, delta):
    dx,dy=delta
    gx,gxy,gy=metric
    q = dx*dx*gx + 2*dx*dy*gxy + dy*dy*gy
    if not math.isfinite(q) or q<=0:
        raise ValueError("affine metric yields nonpositive or non-finite target length")
    return math.sqrt(q)


def fit_affine_metric(observations, directions, max_relative_residual):
    """Return SPD physical tensor + held-out errors under strict source evidence.

    Args:
        observations: list of {svg_units,physical_m} from independent labels.
        directions: matching unit SVG directions.
        max_relative_residual: upper bound on EACH in-fit AND LOO relative error.

    Four nonparallel directions, four distinct source labels and a 0.1% Gram
    pivot threshold are mandatory. With only three references an affine
    tensor could interpolate any three lengths exactly, hiding bad evidence.
    """
    if len(observations)<4 or len(observations)>64 \
            or len(directions)!=len(observations):
        raise ValueError("affine metric requires 4..64 independent directional dimensions")
    if isinstance(max_relative_residual,bool) \
            or not isinstance(max_relative_residual,(float,int)) \
            or not math.isfinite(max_relative_residual) \
            or not 0<=max_relative_residual<=0.1:
        raise ValueError("invalid affine metric relative tolerance")
    rows=[]
    for i,(obs,direction) in enumerate(zip(observations,directions)):
        width,length=obs["svg_units"],obs["physical_m"]
        if (not math.isfinite(width) or width <= 1e-9 or width>1e9
                or not math.isfinite(length) or length<=1e-9 or length>1e9):
            raise ValueError(f"affine reference {i}: physical/svg length outside bound")
        ux,uy=direction
        if not all(math.isfinite(x) for x in direction) \
                or abs(math.hypot(ux,uy)-1)>1e-7:
            raise ValueError(f"affine reference {i}: non-unit SVG direction")
        # Distinct angle orientations modulo 180 degrees; no second
        # independent dimension may claim the same directional evidence.
        if any(abs(ux*existing["unit_direction"][0]+
                   uy*existing["unit_direction"][1])>0.996
               for existing in rows):
            raise ValueError("affine metric requires four distinct reference orientations")
        rows.append({"svg_units":width,"physical_m":length,
                     "unit_direction":(ux,uy)})
    metric,condition=_fit_three(rows)
    fitted=[abs(_length_metres(metric, tuple(
                x*row["svg_units"] for x in row["unit_direction"]))
            -row["physical_m"])/row["physical_m"] for row in rows]
    # All rows are frozen before any holdout solve. The omitted observation
    # cannot silently influence the fit used to validate itself.
    heldout=[]
    for i,row in enumerate(rows):
        train=rows[:i]+rows[i+1:]
        independent,_=_fit_three(train)
        delta=tuple(x*row["svg_units"] for x in row["unit_direction"])
        heldout.append(abs(_length_metres(independent,delta)-
                           row["physical_m"])/row["physical_m"])
    if max(fitted)>max_relative_residual+1e-12 \
            or max(heldout)>max_relative_residual+1e-12:
        raise ValueError("inconsistent affine dimension anchors on held-out directions")
    return {
        "metric":metric,
        "metric_tensor_m2_per_svg_unit2":
            [[metric[0],metric[1]],[metric[1],metric[2]]],
        "distortion_principal_scale_ratio":condition,
        "reference_relative_residuals":fitted,
        "heldout_relative_residuals":heldout,
        "max_relative_residual":max(fitted),
        "max_heldout_relative_residual":max(heldout),
        "reference_count":len(rows),
    }
