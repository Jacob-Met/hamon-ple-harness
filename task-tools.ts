// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * Routes pi's read, write, edit, and bash tools to the task container through
 * the taskbridge Relay (TASK_RELAY_URL). The tools resolve relative paths and
 * run commands in the task's working directory (TASK_WORKDIR).
 */
import {
  type ExtensionAPI,
  createBashToolDefinition,
  createEditToolDefinition,
  createReadToolDefinition,
  createWriteToolDefinition,
} from '@earendil-works/pi-coding-agent';
import {readFile} from 'node:fs/promises';
import http from 'node:http';
import {tmpdir} from 'node:os';
import path from 'node:path';

const RELAY = process.env.TASK_RELAY_URL ?? "";
const WORKDIR = process.env.TASK_WORKDIR ?? "/";
const MAX_EXEC_SEC = 600;
const IMAGES: Record<string, string> = {
	".png": "image/png",
	".jpg": "image/jpeg",
	".jpeg": "image/jpeg",
	".gif": "image/gif",
	".webp": "image/webp",
};

interface RelayResponse {
	ok?: boolean;
	error?: string;
	truncated?: boolean;
	size?: number;
	content_b64?: string;
	output?: string;
	timed_out?: boolean;
	exit_code: number;
}

function relay(route: string, body: object): Promise<RelayResponse> {
	const url = new URL(`${RELAY}/${route}`);
	const data = Buffer.from(JSON.stringify(body));
	return new Promise((resolve, reject) => {
		const req = http.request(
			{
				hostname: url.hostname,
				port: url.port,
				path: url.pathname,
				method: "POST",
				headers: { "Content-Type": "application/json", "Content-Length": data.length },
			},
			(res) => {
				const chunks: Buffer[] = [];
				res.on("data", (chunk) => chunks.push(chunk));
				res.on("end", () => {
					try {
						const out = JSON.parse(Buffer.concat(chunks).toString() || "{}") as RelayResponse;
						if (res.statusCode !== 200) reject(new Error(out.error ?? `relay answered ${res.statusCode}`));
						else if (out.ok === false) reject(new Error(out.error));
						else resolve(out);
					} catch (err) {
						reject(err);
					}
				});
			},
		);
		req.on("error", reject);
		req.end(data);
	});
}

function quote(s: string): string {
	return `'${s.replace(/'/g, `'\\''`)}'`;
}

/**
 * Wraps `tool` so relative paths (e.g., "../data.csv") resolve and commands
 * start in `workdir`.
 *
 * `pi` uses the cwd of the context passed to a tool (which defaults to pi's own
 * scratch directory), so the tool receives a derived context whose `cwd` is
 * `workdir`. `Object.create` preserves the context's other properties, which pi
 * reads lazily.
 */
// tslint:disable-next-line:no-any
export function inTask(tool: any, workdir: string): any {
	return {
		...tool,
		// tslint:disable-next-line:no-any
		execute: (id: string, params: unknown, signal: unknown, onUpdate: unknown, ctx: any) =>
			tool.execute(id, params, signal, onUpdate, ctx && Object.create(ctx, { cwd: { value: workdir } })),
	};
}

/**
 * Returns a bash call's time limit in seconds: pi's `timeout` when it is a
 * positive number (or numeric string), capped at `max`. Zero, a negative
 * number, or an omitted value returns `max`, since 0 represents "no limit" in
 * pi.
 */
export function execLimit(timeout: unknown, max: number): number {
	const seconds = typeof timeout === "string" && timeout.trim() !== "" ? Number(timeout) : timeout;
	const usable = typeof seconds === "number" && Number.isFinite(seconds) && seconds > 0;
	return usable ? Math.max(1, Math.ceil(Math.min(seconds, max))) : max;
}

/**
 * Formats `exec` output as shown by a bash call, appending a truncation notice
 * at the end when the evaluator cut the middle of the output.
 */
export function execOutput(output: string, truncated: boolean): string {
	const note = "\n[the output was over 100 KB, so its middle was cut; write it to a file and read that in parts]\n";
	return truncated ? output + note : output;
}

/**
 * Returns true if `p` is pi's own temporary log file holding a long command's
 * output (`<tmp>/pi-bash-<16 hex>.log`), which stays local to the harness.
 */
export function isLocalTemp(p: string, tmp: string): boolean {
	return path.dirname(p) === tmp && /^pi-[a-z]+-[0-9a-f]{16}\.log$/.test(path.basename(p));
}

/**
 * Registers taskbridge-backed read, write, edit, and bash tools on the pi
 * coding agent extension API.
 */
// tslint:disable-next-line:no-default-export
function _origRegisterTaskTools_registerTaskTools(pi: ExtensionAPI) {
	// pi saves a long command's output to a local temp file and points the model at it; reads of that file stay local.
	const isLocal = (p: string) => isLocalTemp(p, tmpdir());

	const readOp = async (p: string) => {
		if (isLocal(p)) return readFile(p);
		const r = await relay("read_file", { path: p });
		if (r.truncated) throw new Error(`${p} is ${r.size} bytes, over the 10 MB a read returns; use bash on it`);
		return Buffer.from(r.content_b64 ?? "", "base64");
	};
	const writeOp = async (p: string, content: string) => {
		await relay("write_file", { path: p, content_b64: Buffer.from(content).toString("base64") });
	};
	// No separate access check: readOp and writeOp report a missing or unwritable file, one round trip fewer.
	const access = async () => {};
	// tslint:disable-next-line:no-any
	const register = (tool: any) => pi.registerTool(inTask(tool, WORKDIR));

	register(
		createReadToolDefinition(WORKDIR, {
			operations: {
				readFile: readOp,
				access,
				detectImageMimeType: async (p) => IMAGES[path.extname(p).toLowerCase()] ?? null,
			},
		}),
	);
	// The relay's write_file creates missing parent directories.
	register(createWriteToolDefinition(WORKDIR, { operations: { writeFile: writeOp, mkdir: async () => {} } }));
	register(createEditToolDefinition(WORKDIR, { operations: { readFile: readOp, writeFile: writeOp, access } }));
	register(
		createBashToolDefinition(WORKDIR, {
			operations: {
				exec: async (command, cwd, { onData, signal, timeout }) => {
					if (signal?.aborted) throw new Error("aborted");
					const limit = execLimit(timeout, MAX_EXEC_SEC);
					const run = relay("exec", { command: `cd ${quote(cwd)} || exit 1\n${command}`, timeout_sec: limit });
					// The relay cannot cancel an exec: an abort stops the wait, and the command runs on to its time limit.
					const abort = new Promise<never>((_, reject) =>
						signal?.addEventListener("abort", () => reject(new Error("aborted")), { once: true }),
					);
					const r = await Promise.race([run, abort]);
					const output = execOutput(r.output ?? "", Boolean(r.truncated));
					if (output) onData(Buffer.from(output));
					if (r.timed_out) throw new Error(`timeout:${limit}`);
					return { exitCode: r.exit_code };
				},
			},
		}),
	);

	pi.on("before_agent_start", (event) => {
		event.systemPromptOptions.cwd = WORKDIR;
	});
}


import { spawn as _pleSpawn } from "node:child_process";
import * as _pleFs from "node:fs";
import * as _plePath from "node:path";
import { fileURLToPath as _pleFileURLToPath } from "node:url";

export const PLE_DOMAIN_TOOL_NAMES = [
  "hamon_ifc_spatial_clash",
  "hamon_drawing_reader",
  "hamon_conservation_cad_verifier",
  "check_deliverable_preflight",
];

const _PLE_HERE_DIR = _plePath.dirname(_pleFileURLToPath(import.meta.url));

async function _pleReadTaskBytes(taskPath: string): Promise<Buffer> {
  // Task files live in a different container. The authenticated relay is the
  // only authority for these bytes, including when a local namesake exists.
  const result = await relay("read_file", { path: taskPath });
  if (result.truncated) {
    throw new Error(`Task file ${taskPath} was truncated; use task bash to inspect a bounded portion`);
  }
  if (typeof result.content_b64 !== "string") {
    throw new Error(`Task read for ${taskPath} returned no file bytes`);
  }
  const bytes = Buffer.from(result.content_b64, "base64");
  if (bytes.toString("base64") !== result.content_b64 ||
      (typeof result.size === "number" && result.size !== bytes.length)) {
    throw new Error(`Task read for ${taskPath} returned incomplete or invalid file bytes`);
  }
  return bytes;
}

function _pleRunPythonJson(scriptName: string, payload: Record<string, unknown>): Promise<any> {
  const pyBin =
    process.env.TASK_HARNESS_PYTHON ||
    (_pleFs.existsSync("/opt/harness/venv/bin/python") ? "/opt/harness/venv/bin/python" : "python3");
  const srcDir = process.env.TASK_HARNESS_SRC || _PLE_HERE_DIR;
  const scriptPath = _plePath.join(srcDir, scriptName);
  return new Promise((resolve, reject) => {
    const child = _pleSpawn(pyBin, [scriptPath, "--json-stdin"], {
      stdio: ["pipe", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => { stdout += chunk.toString(); });
    child.stderr.on("data", (chunk) => { stderr += chunk.toString(); });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(`Python tool ${scriptName} exited ${code}: ${stderr || stdout}`));
        return;
      }
      try {
        const lines = stdout.trim().split("\n");
        resolve(JSON.parse(lines[lines.length - 1]));
      } catch (err) {
        reject(new Error(`Invalid JSON from ${scriptName}: ${stdout}`));
      }
    });
    child.stdin.write(JSON.stringify(payload));
    child.stdin.end();
  });
}

function _registerPleWave3DomainTools(pi: any) {
  pi.registerTool({
    name: "hamon_ifc_spatial_clash",
    label: "IFC4 Spatial Clash & Solid Takeoff",
    description: "Run deterministic IFC4 solid volume takeoff, AABB broadphase, and cross-discipline spatial clash/clearance analysis. The model minimum uses measured solid distances. Each measured pair includes closest_points_m in element order. For a pruned clear pair, distance_lower_bound_m is available and min_distance_m and the witness fields are null.",
    parameters: {
      type: "object",
      properties: {
        path: { type: "string", description: "Workspace-relative path to .ifc model" },
        required_clearance_m: { type: "number", description: "Minimum required clearance in meters (default 0.05)" },
        cross_discipline_only: { type: "boolean", description: "Filter to cross-discipline clashes only (default true)" },
      },
      required: ["path"],
    },
    async execute(_id: string, params: any) {
      const raw = await _pleReadTaskBytes(String(params.path));
      const details = await _pleRunPythonJson("hamon_ifc_spatial_clash.py", {
        mode: "clash",
        ifc_b64: raw.toString("base64"),
        required_clearance_m: params.required_clearance_m ?? 0.05,
        cross_discipline_only: params.cross_discipline_only ?? true,
      });
      return {
        content: [{ type: "text", text: JSON.stringify(details, null, 2) }],
        details,
      };
    },
  });

  pi.registerTool({
    name: "hamon_drawing_reader",
    label: "Engineering PDF Vector Drawing Reader",
    description: "Extract calibrated vector linework, scale ratios, title-block metadata, and text callouts from engineering PDF drawings.",
    parameters: {
      type: "object",
      properties: {
        path: { type: "string", description: "Workspace-relative path to .pdf sheet" },
        page_index: { type: "number", description: "Optional 0-based page index filter" },
        scale_override: { type: "string", description: "Optional scale string (e.g. '1:50' or '0.25 in = 1 ft') or numeric ratio" },
      },
      required: ["path"],
    },
    async execute(_id: string, params: any) {
      const raw = await _pleReadTaskBytes(String(params.path));
      const details = await _pleRunPythonJson("hamon_drawing_reader.py", {
        pdf_b64: raw.toString("base64"),
        page_index: params.page_index ?? null,
        scale_override: params.scale_override ?? null,
      });
      return {
        content: [{ type: "text", text: JSON.stringify(details, null, 2) }],
        details,
      };
    },
  });

  pi.registerTool({
    name: "hamon_conservation_cad_verifier",
    label: "Deterministic Engineering & Conservation Verifier",
    description: "Verify CAD/STEP geometry, dimensional homogeneity, structural mechanics, circuits, geotechnics and thermal-fluid balances. Also solve PDEs, control systems, psychrometrics, FEM, compressible flow/acoustics, RF/RLC, stress/vessels/buckling, reactor kinetics, thermodynamic cycles, heat exchangers/radiation, Lagrangian/orbital mechanics and electrochemistry/separations. Set the outer mode to the solver family and put calculation inputs in spec. For a semi-rigid diaphragm use structural with spec.calculation=diaphragm_membrane, nodes_m [[x,y],...], rectangular panels {nodes:[SW,SE,NE,NW],Dx_N_m,Dy_N_m,G_N_m}, optional bars {nodes:[a,b],EA_N}, springs {node,axis:x or y,k_N_m}, and loads {node,force_N:[Fx,Fy]}. Use zero-based node indices and omit cells at openings. Optional displacement_precision=compensated supports up to 64 displacement DOFs and 1024 strain rows. When verified_displacement_representation is binary64_primary_plus_remainder, replay each displacement as the exact sum of the matching displacements_m and displacement_remainders_m binary64 components. Returns element-center shear, axial bar tension and signed forces transmitted to the wall springs in input order. Provide allowable_tension_N and allowable_compression_N on a bar for a signed capacity check, with optional demand_factor. Optional stations {edge_nodes:[start,end],fraction,tie_rule} retain all equally near exterior panel centers. Use first_from_start or max_abs to resolve ties. For three-phase Thevenin equivalents use circuit with spec.calculation=symmetrical_fault, fault_type (3PH/SLG/LL/DLG), prefault_positive_voltage_pu, sequence_impedances_pu with zero/positive/negative, and optional fault_impedance_pu, faulted_phases and proposed_phase_currents_pu. Each phasor has real and imag per-unit components. For LL, impedance lies between the two terminals. For DLG, both terminals join one impedance to ground. For balanced per-unit AC load flow use circuit with spec.calculation=ac_power_flow, buses (id, type slack/PV/PQ, vm_pu on slack/PV, p_injection_pu on PV/PQ and q_injection_pu on PQ), and lines (id, from, to, r_pu, x_pu; optional line_charging_b_pu, tap_ratio, phase_shift_deg). PV reactive limits are net injections in q_min_injection_pu and q_max_injection_pu. Use proposed_voltages keyed by bus id with vm_pu and va_deg to check an answer. For an adiabatic rotor use thermal_fluid with spec.calculation=euler_rotor, machine_kind (pump, compressor or turbine), rotation_rpm, and inlet/outlet stations with radius_m, meridional_velocity_m_s and absolute_tangential_velocity_m_s or signed absolute_flow_angle_deg/relative_flow_angle_deg. Angles are measured from the meridional axis toward rotation. A blade metal angle is an optional comparison. For a conservation certificate supply measured_rotor_static_enthalpy_change_J_kg, or mass_flow_kg_s and measured_fluid_torque_N_m. Optional stage_boundary_speeds_m_s (inlet/outlet), isentropic_specific_work_J_kg and pump suction_flange add reaction, efficiency and NPSH checks. For rated pump duty use thermal_fluid with spec.calculation=pump_duty, a single curve of {flow_m3_s, head_m} points, total_flow_m3_s and optional identical_pumps sharing flow equally. Supply required_head_m for a known duty, or equipped_branch to recompute candidate losses. The latter takes common_head_m (stage head at total flow excluding these accessories), hose_length_m, hose_diameter_m, hose_roughness_m, kinematic_viscosity_m2_s, strainer_reference_flow_m3_s, strainer_reference_head_m and optional gravity_m_s2. Each identical branch carries its share of flow. Pass per_pump_input_W together with available_panel_W to also check the panel. Inspect duty_verified and checks at the prescribed flow. For passive water pipe networks use thermal_fluid with spec.calculation=pipe_network. Supply nodes with id and one fixed_head_m reservoir or signed demand_m3_s, and pipes with id, from, to, length_m, diameter_m, optional roughness_m and minor_loss_K. Set network-wide gravity_m_s2 for local gravitational acceleration (default 9.80665 m/s²). Positive flow follows the declared pipe orientation. For compression-only rigid strip contact use civil_geotech with spec.calculation=rigid_strip_contact, base_width_m, vertical_resultant_N_m (positive downward) and signed moment_Nm_m about the base center. The model is p(x)=max(0,a+b*x), with integral(x*p)=M; positive M loads the positive-x edge. It returns the full or partial contact interval and peak pressure, and refuses finite pressure equilibrium at or beyond a base edge. This is contact equilibrium, separate from bearing capacity. For centered shallow foundation bearing, use civil_geotech with spec.calculation=bearing_capacity, bearing_method (terzaghi, meyerhof, vesic), footing_B_m, depth_Df_m, gamma_N_m3, phi_deg, c_prime_Pa and FS. For groundwater supply water_table_depth_m and gamma_sat_N_m3. Optional footing_L_m defines a finite rectangle. For eccentric vertical loading on a finite rectangle or square, select bearing_method=meyerhof and eccentricity_method=meyerhof_effective_area. Supply positive column_load_N and signed eccentricity_B_m and/or eccentricity_L_m on the named plan axes. All bearing factors use the effective rectangle, and q_applied_Pa is the equivalent uniform pressure over its area. Terzaghi supports strip, square or circular geometry through footing_shape. To check supplied answers, use proposed_q_ult_Pa and proposed_q_allow_Pa for gross capacities or proposed_q_net_ult_Pa and proposed_q_net_allow_Pa for net capacities. For field compaction use civil_geotech with calculation=compaction_acceptance, gamma_d_max, w_opt_pct, min_rc (fraction), plus w_tol_pct or both w_tol_below_pct and w_tol_above_pct in percentage points. For a WSDOT M 46-01.48 (January 2026), T 99/T 180 Annex A reference in the default range of more than 5 to at most 30 percent oversize, replace gamma_d_max and w_opt_pct with oversize_reference containing coarse_dry_mass_pct (retained on the method sieve), fine_gamma_d_max, fine_w_opt_pct, coarse_particle_dry_density (oven-dry bulk specific gravity × water density, in the same units as all other densities), and coarse_w_pct. Moistures are percent by dry mass. Comparisons use the unrounded derived_reference; supply the direct reference when the project requires rounded values. Each test needs a unique id, w_pct, and either gamma_dry or gamma_wet in the same density units as the reference. This checks supplied measurements against that reference. For primary consolidation, use civil_geotech with spec.calculation=consolidation_1d, H_clay_m, e0, Cc, sigma_v0_eff_Pa, delta_sigma_v_Pa, cv_m2_s, time_s and drainage=single or double. Add sigma_preconsolidation_Pa and Cr for overconsolidated clay. To derive the increment instead, supply stress_point {x_m,y_m,depth_m} below the common loading plane and rectangular_loads of {center_x_m,center_y_m,width_m,length_m,net_pressure_Pa}. These axis-aligned uniform net pressures superpose in a homogeneous isotropic elastic half-space. An accompanying delta_sigma_v_Pa is checked against the derived increment. For reversible linear consolidation under staged uniform surcharge, use calculation=consolidation_history_1d with H_clay_m, cv_m2_s, mv_per_Pa (constant volume compressibility), drainage, time_s and load_history of {time_s,surcharge_Pa}. Start at zero time, interpolate linearly, repeat a time for an instantaneous jump and hold the last ordinate. A positive first ordinate is an instantaneous initial load. Returns settlement and signed mean excess pore pressure. For a smooth vertical Rankine wall, set calculation=rankine_layers, pressure_state=active or passive, and list layers from the surface with thickness_m and phi_deg. Supply gamma_above_water_N_m3 in dry soil and gamma_saturated_N_m3 or gamma_submerged_N_m3 in submerged soil. Measure the retained-side water_table_depth_m and optional exposed-face front_water_table_depth_m down from the same wall top. For opposing earth, supply front_layers from exposed grade to the common toe. Their total thickness fixes the grade, optionally checked with front_surface_depth_m. The two faces mobilize opposite active/passive states, with independent uniform surcharge_Pa and front_surcharge_Pa. Positive net thrust acts toward the exposed face. The signed moment uses height above the toe. A pure couple has no force location. For pde/control/psychrometrics/fem, spec.mode selects the calculation (e.g. mode=control with spec.mode=state_space and A/B/C matrices). Other new families require spec.subkind (e.g. mode=compressible_acoustics with spec.subkind=normal_shock and M1), except reactor_kinetics requires spec.reactor_type. For a single CSTR specified by volume, set spec.steady_state_branch to all to enumerate every steady conversion, or an ascending zero-based index to select one. Multiple states without an explicit branch are refused. Prescribed-conversion sizing remains single-valued; steady-state enumeration does not establish dynamic stability. A completed call is not evidence of a successful check: inspect the returned passed/verified/conservation_verified flags and diagnostics.",
    parameters: {
      type: "object",
      properties: {
        mode: { type: "string", description: "Solver family: auto, step_cad, structural, circuit, civil_geotech, thermal_fluid, dimensional, pde, control, psychrometrics, fem, compressible_acoustics, rf_transmission_rlc, solid_mohr_vessel_buckling, reactor_kinetics, thermo_cycles, exchangers_radiation, lagrangian_orbital, electrochem_separations." },
        path: { type: "string" },
        spec: { type: "object", description: "Calculation parameters in the selected solver's units, including explicit mode/subkind/reactor_type for the new families. A supplied domain must match the outer mode. Empty or selector-only new-family specs are rejected." },
      },
    },
    async execute(_id: string, params: any) {
      const payload: Record<string, unknown> = { mode: params?.mode ?? "auto", spec: params?.spec ?? {} };
      if (params?.path) {
        const raw = await _pleReadTaskBytes(String(params.path));
        payload.file_b64 = raw.toString("base64");
        payload.filename = _plePath.basename(String(params.path));
      }
      const details = await _pleRunPythonJson("hamon_conservation_cad_verifier.py", payload);
      return { content: [{ type: "text", text: JSON.stringify(details, null, 2) }], details };
    },
  });

  pi.registerTool({
    name: "check_deliverable_preflight",
    label: "Deliverable Preflight Gate",
    description: "Validate output deliverable existence, archive/file integrity, non-empty content, finite numerics, and required task IDs before submission.",
    parameters: {
      type: "object",
      properties: {
        path: { type: "string", description: "Task-container path to a deliverable file, directory, or .zip archive" },
        expected_task_ids: { type: "array", items: { type: "string" }, description: "Optional required task IDs" },
      },
      required: ["path"],
    },
    async execute(_id: string, params: any) {
      const taskPath = String(params.path);
      const srcDir = process.env.TASK_HARNESS_SRC || _PLE_HERE_DIR;
      const source = _pleFs.readFileSync(_plePath.join(srcDir, "hamon_preflight.py"), "utf8");
      const payload = JSON.stringify({
        path: taskPath,
        expected_task_ids: params.expected_task_ids ?? null,
      });
      // Inspect files and directories in their original task filesystem. Copying
      // or re-zipping directories would change the existing symlink checks.
      const result = await relay("exec", {
        command: `python3 -I -c ${quote(source)} ${quote(payload)}`,
        timeout_sec: 120,
      });
      if (result.truncated || result.timed_out || result.exit_code !== 0) {
        throw new Error(`Task preflight did not complete: ${result.output ?? ""}`);
      }
      const details = JSON.parse(result.output ?? "");
      return {
        content: [{ type: "text", text: JSON.stringify(details, null, 2) }],
        details,
      };
    },
  });
}

export default function registerTaskTools(pi: any) {
  _origRegisterTaskTools_registerTaskTools(pi);
  _registerPleWave3DomainTools(pi);
}
