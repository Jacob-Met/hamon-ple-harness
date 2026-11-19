# Unit-safe engineering calculation DAG

Task container CLI: `python3 /tmp/hamon-engineering-toolkit.py calc-pack design_checks.json`.

Input JSON has schema `hamon.engineering.calcpack.v1`, an object `quantities` containing named `{expr, to?, source?}` entries, and an array `checks` containing `{id,lhs,rhs,relation,relative_tolerance?,source?}`. Quantities are recursively evaluated independent of declaration order. Dimensionally inconsistent expressions, cycles, unsafe AST, ambiguous `lb`, non-positive conversion units, and oversized graphs are errors (exit 2). Violated checks produce exit 3 and a JSON result with named SI residuals, utilization when RHS positive, and full supplied expressions. An algebraic pass produces exit 0.

Example: `{"schema":"hamon.engineering.calcpack.v1","quantities":{"q":{"expr":"10*kN/m","source":"provided line load"},"L":{"expr":"5*m","source":"provided span"},"M":{"expr":"q*L**2/2","to":"kN*m","source":"cantilever UDL bending"},"Mallow":{"expr":"250*MPa*0.001*m**3/1.5","source":"provided allowable"}},"checks":[{"id":"bending","lhs":"M","relation":"<=","rhs":"Mallow","source":"project criterion"}]}`.

**The check is not an engineering design or code-compliance certification.** It verifies algebraically that the supplied formulas and thresholds are mutually consistent; it does not establish their physics, units as recorded in source documents, load cases, applicable standards, reliability, or engineering adequacy. Validate those separately before using results.
