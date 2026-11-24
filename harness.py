# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pi coding agent harness (`@earendil-works/pi-coding-agent`).

Runs one headless Pi session per task. Pi runs in the serve container while the
task's files live in the task container. `task-tools.ts` replaces Pi's built-in
`read`, `write`, `edit`, and `bash` tools with versions that operate on the task
container through `taskbridge.Relay`, and the task's MCP servers are proxied
through the same relay. Pi's agent loop handles model calls, tool execution,
retries, and conversation compaction, returning its last response as the final
message.

The model is configured from the environment variables set by the evaluator in
`models.json` for each task. Claude and Gemini models use their native APIs
under Pi's built-in provider names (`anthropic` and `google`) so Pi's catalog
supplies their limits, thinking support, and image input capabilities. Any other
model uses the OpenAI-compatible `MODEL_BASE_URL` and `MODEL_API_KEY` under a
custom `model` provider with `UNKNOWN_MODEL` limits.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import pathlib
import secrets
import subprocess
import tempfile
from time import monotonic, time
from typing import Any

import taskbridge
import engineering_intake
import delivery_guard
import archive_contract
import delivery_materialize
import hamon_pareto

HERE = pathlib.Path(__file__).resolve().parent
# submission.yaml installs this CLI into the harness source tree with npm ci.
PI = str(HERE / "node_modules" / ".bin" / "pi")
# pi's --thinking level: off, minimal, low, medium, high, xhigh or max
THINKING = "max"
# stop pi this long before the deadline (less on short tasks) so its answer
# arrives
STOP_BEFORE_SEC = 90
# time_share caps this reserve at one quarter of the remaining task time.
DELIVERY_RESERVE_SEC = 30 * 60
MIN_DELIVERY_RUN_SEC = 60
STARTUP_OBSERVATION_SECONDS = 29
STARTUP_TASK_RESERVE_SECONDS = 12
CATALOG_TIMEOUT_SEC = 30
# Request features the Google Cloud route for Claude refuses. pi leaves them
# out, and the Anthropic API works without them too.
COMPAT = {
    "supportsStrictTools": False,
    "supportsMidConvoEffort": False,
    "supportsMidConvoSystemMessages": False,
    "supportsMidConvoToolChanges": False,
}
# A model pi's catalog does not know gets these limits; tune them for yours.
UNKNOWN_MODEL = {"input": ["text"], "contextWindow": 128000, "maxTokens": 16384}
NOTE = (
    "You are working on an engineering task. Your tools act on the task's Linux"
    " container, whose working directory is {workdir}; it has no internet"
    " access. Read the whole instruction and the files it points to, put every"
    " deliverable exactly where and in the format the instruction asks, and"
    " check what you wrote. When you are done, reply with a short summary of"
    " what you delivered and where."
)


def provider() -> tuple[str, dict[str, Any]]:
  """Returns the Pi provider name and its `models.json` entry.

  Returns:
    A tuple of `(provider_name, provider_config)`.
  """
  if os.environ.get("ANTHROPIC_BASE_URL"):
    return "anthropic", {
        "baseUrl": os.environ["ANTHROPIC_BASE_URL"],
        "api": "anthropic-messages",
        "apiKey": "$ANTHROPIC_API_KEY",
        "compat": COMPAT,
    }
  if os.environ.get("GOOGLE_GEMINI_BASE_URL"):
    return "google", {
        "baseUrl": os.environ["GOOGLE_GEMINI_BASE_URL"] + "/v1beta",
        "api": "google-generative-ai",
        "apiKey": "$GEMINI_API_KEY",
    }
  return "model", {
      "baseUrl": os.environ["MODEL_BASE_URL"],
      "api": "openai-completions",
      "apiKey": "$MODEL_API_KEY",
  }


def in_catalog(name: str, model: str, env: dict[str, str]) -> bool:
  """Checks whether Pi's built-in model catalog recognizes `model`.

  If the probe hangs beyond `CATALOG_TIMEOUT_SEC` or fails, returns `False` so
  the model is configured explicitly with `UNKNOWN_MODEL` limits.

  Args:
    name: Provider name (`anthropic`, `google`, or `model`).
    model: Model identifier string.
    env: Subprocess environment variables for the Pi CLI probe.

  Returns:
    `True` if `(name, model)` appears in `pi --list-models`, else `False`.
  """
  try:
    out = subprocess.run(
        [PI, "--offline", "--list-models", model],
        env=env,
        capture_output=True,
        text=True,
        timeout=CATALOG_TIMEOUT_SEC,
        check=False,
    ).stdout
  except (subprocess.TimeoutExpired, OSError):
    return False
  return any(line.split()[:2] == [name, model] for line in out.splitlines())


def configure(
    agent_dir: pathlib.Path,
    relay: taskbridge.Relay,
    session: taskbridge.Session,
    env: dict[str, str],
) -> tuple[str, str]:
  """Writes Pi's model, settings, and MCP configuration files for a task.

  Args:
    agent_dir: Directory where Pi reads its configuration files.
    relay: Local HTTP relay for the task container and its MCP servers.
    session: Active task session.
    env: Environment dictionary passed to Pi subprocesses.

  Returns:
    A tuple of `(provider_name, model_name)`.
  """
  model = os.environ["MODEL_NAME"]
  name, config = provider()
  models = agent_dir / "models.json"
  # `pi --list-models` reads the provider here
  models.write_text(json.dumps({"providers": {name: config}}))
  if not in_catalog(name, model, env):
    config["models"] = [{"id": model, "name": model, **UNKNOWN_MODEL}]
    models.write_text(json.dumps({"providers": {name: config}}))
  settings = {
      "defaultProjectTrust": "never",
      "quietStartup": True,
      "cacheWarming": "off",
      "retry": {"maxRetries": 6, "maxAgentDelayMs": 60000},
  }
  (agent_dir / "settings.json").write_text(json.dumps(settings))
  servers = {
      s["name"]: {"url": relay.mcp_url(s["name"]), "timeout": 300}
      for s in session.mcp_servers
  }
  (agent_dir / "mcp.json").write_text(json.dumps({"mcpServers": servers}))
  return name, model


def message_digest(message: dict[str, Any]) -> bytes:
  """Compare live and persisted Pi messages without keeping another tool payload."""
  encoded = json.dumps(message, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
  return hashlib.sha256(encoded.encode("ascii")).digest()


def empty_budget_abort(message):
  """Recognize the pinned SDK's empty abort, including its complete zero usage."""
  if (message.get("role") != "assistant" or message.get("content") != []
      or message.get("stopReason") not in {"error", "aborted"}):
    return False
  usage = message.get("usage")
  counts = {"input", "output", "cacheRead", "cacheWrite", "totalTokens"}
  costs = {"input", "output", "cacheRead", "cacheWrite", "total"}
  if not isinstance(usage, dict) or set(usage) != counts | {"cost"}:
    return False
  cost = usage["cost"]
  if not isinstance(cost, dict) or set(cost) != costs:
    return False
  return all(type(value) in (int, float) and value == 0
             for value in [*(usage[key] for key in counts), *cost.values()])


def acknowledged_tool_session(
    directory: pathlib.Path, assistant: bytes, observed: dict[str, bytes],
    expected: set[str], history: bytes | None, *, budget_id: str | None = None,
) -> pathlib.Path | None:
  """Find the one linear private session with this exact assistant and all results."""
  if history is None or not expected or observed.keys() != expected:
    return None
  files = list(directory.glob("*.jsonl"))
  if len(files) != 1 or files[0].is_symlink():
    return None
  parent = None
  seen: set[str] = set()
  matched = False
  cutoff = False
  aborted_tail = False
  results: dict[str, bytes] = {}
  try:
    with files[0].open(encoding="utf-8") as stream:
      header = json.loads(next(stream))
      if header.get("type") != "session":
        return None
      replay = message_digest(header)
      for line in stream:
        entry = json.loads(line)
        entry_id = entry.get("id")
        if not isinstance(entry_id, str) or entry_id in seen or entry.get("parentId") != parent:
          return None
        seen.add(entry_id)
        parent = entry_id
        if entry.get("type") != "message":
          if matched:
            data = entry.get("data") or {}
            if (cutoff or not budget_id or entry.get("type") != "custom"
                or entry.get("customType") != "ple_delivery_boundary"
                or data.get("budget_id") != budget_id
                or data.get("tool_call_ids") != sorted(expected)
                or results != observed):
              return None
            cutoff = True
          continue
        message = entry["message"]
        digest = message_digest(message)
        replay = hashlib.sha256(replay + digest).digest()
        if cutoff:
          if aborted_tail or not empty_budget_abort(message):
            return None
          aborted_tail = True
        elif message.get("role") == "assistant":
          matched = digest == assistant
          results.clear()
        elif matched:
          if message.get("role") != "toolResult":
            return None
          call_id = message.get("toolCallId")
          if call_id not in expected or call_id in results:
            return None
          results[call_id] = digest
  except (OSError, UnicodeError, ValueError, KeyError, TypeError, AttributeError, StopIteration):
    return None
  return files[0] if matched and replay == history and results == observed and (not cutoff or aborted_tail) else None


class Run:
  """Tracks JSON stream events from a single Pi subprocess run."""

  def __init__(self, session: taskbridge.Session, model: str, *, budgeted=False):
    self.session, self.model, self.replies = session, model, 0
    self.budget_steps = 0
    self.budget_id = secrets.token_hex(16) if budgeted else None
    self.tool_turn: bytes | None = None
    self.expected_calls: set[str] = set()
    self.observed_results: dict[str, bytes] = {}
    self.invalid_tail = False
    self.history: bytes | None = None
    self.archive_watch = None

  async def on_event(self, event: dict[str, Any]) -> None:
    """Processes a parsed JSON event from Pi, updating final message and usage.

    Args:
      event: Decoded JSON event dictionary from Pi's stdout stream.
    """
    if self.archive_watch:
      await self.archive_watch.observe(self.session, event)
    kind, message = event.get("type"), event.get("message") or {}
    if kind == "session":
      self.history = message_digest(event)
    if kind == "message_end" and self.history is not None:
      if message.get("role") in ("system", "user", "assistant", "toolResult"):
        self.history = hashlib.sha256(self.history + message_digest(message)).digest()
    if kind == "message_end" and message.get("role") == "toolResult":
      if self.tool_turn is not None:
        call_id = message.get("toolCallId")
        if call_id not in self.expected_calls or call_id in self.observed_results:
          self.invalid_tail = True
        else:
          self.observed_results[call_id] = message_digest(message)
    elif kind == "message_end" and message.get("role") == "assistant":
      self.replies += 1
      if (self.budget_id and self.tool_turn is not None and not self.invalid_tail
          and self.expected_calls and self.observed_results.keys() == self.expected_calls
          and empty_budget_abort(message)):
        # Preserve the completed batch only as a candidate. The private session
        # must still prove the matching cutoff marker and exact full history.
        return
      self.budget_steps += 1
      calls = [b for b in message.get("content", []) if b.get("type") == "toolCall"]
      self.tool_turn = message_digest(message) if calls else None
      self.expected_calls = {b["id"] for b in calls if isinstance(b.get("id"), str) and b["id"]}
      self.observed_results = {}
      self.invalid_tail = bool(calls) and (
          len(self.expected_calls) != len(calls) or message.get("stopReason") != "toolUse")
      text = "\n".join(
          b.get("text", "")
          for b in message.get("content") or []
          if b.get("type") == "text"
      )
      if calls:
        self.session.final_message = ""
      elif text.strip():
        self.session.final_message = text.strip()
      if message.get("stopReason") == "error":
        await self.session.log(
            f"pi: model error: {message.get('errorMessage')}"
        )
      # input_tokens counts cached prompt tokens too
      u = message.get("usage") or {}
      await self.session.usage(
          self.model,
          input_tokens=u.get("input", 0)
          + u.get("cacheRead", 0)
          + u.get("cacheWrite", 0),
          output_tokens=u.get("output", 0),
          cache_read_tokens=u.get("cacheRead", 0),
      )
    elif kind == "tool_execution_start":
      await self.session.log(
          f"pi: {event.get('toolName')} {json.dumps(event.get('args'))[:300]}"
      )
    elif kind in ("auto_retry_start", "compaction_start"):
      await self.session.log(f"pi: {kind} {json.dumps(event)[:300]}")

  async def on_line(self, line: bytes) -> None:
    """Parses a raw stdout line from Pi as JSON and forwards it to `on_event`.

    Args:
      line: Raw line bytes emitted by Pi.
    """
    try:
      event = json.loads(line)
    except ValueError:
      return
    if isinstance(event, dict):
      await self.on_event(event)


def delivery_step_cap(session):
  value = getattr(session, "limits", {}).get("max_steps")
  return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def delivery_budget_exhausted(session, steps):
  cap = delivery_step_cap(session)
  return cap is not None and steps >= cap


def delivery_process_env(env, session, steps, reserve, *, delivery=False, budget_id=None):
  """Configure the cooperative Pi boundary using this pass's actual budget."""
  remaining = session.time_left()
  work = taskbridge.time_share(remaining, reserve)
  now = time()
  config = {"phase": "delivery" if delivery else "work",
            "completed_steps": steps, "max_steps": delivery_step_cap(session),
            "phase_deadline_ms": None if work is None else (now + work) * 1000,
            "task_deadline_ms": None if remaining is None else (now + remaining) * 1000}
  if budget_id is not None:
    config["budget_id"] = budget_id
  return {**env, "PLE_DELIVERY_BUDGET": json.dumps(config, separators=(",", ":"))}


async def solve(session: taskbridge.Session) -> str:
  """Runs Pi on a single task and returns the final response.

  Creates an isolated temporary agent home, configures tools and MCP servers to
  communicate through `taskbridge.Relay`, and feeds the task instruction on
  standard input.

  Args:
    session: Active task session.

  Returns:
    The final message produced by Pi.

  Raises:
    RuntimeError: If Pi exits before the model produces any assistant reply.
  """
  with tempfile.TemporaryDirectory(
      prefix="pi-",
      dir=os.environ.get("HOME") or None,
      ignore_cleanup_errors=True,
  ) as scratch:
    agent_dir, work = (
        pathlib.Path(scratch) / "agent",
        pathlib.Path(scratch) / "work",
    )
    agent_dir.mkdir()
    work.mkdir()
    async with taskbridge.Relay(session) as relay:
      env = relay.env()
      env.update(
          PI_CODING_AGENT_DIR=str(agent_dir),
          PI_OFFLINE="1",
          PI_SKIP_VERSION_CHECK="1",
          PI_TELEMETRY="0",
          NODE_NO_WARNINGS="1",
          TASK_WORKDIR=session.workdir,
      )
      # One read-only project inventory over the evaluator's task protocol.
      # This is ephemeral task data; never store or log its contents.
      engineering_inventory = await engineering_intake.intake_for_prompt(session)
      delivery_targets = delivery_guard.extract_targets(
          session.instruction, session.workdir
      )
      terminal = delivery_guard.TerminalDelivery(session.instruction, delivery_targets)
      working_copy = terminal.add_working_copy(session.workdir)
      if working_copy:
        delivery_targets = [{"path": working_copy, "kind": "file"}]
      session.fallback_summary = terminal.summary
      # The same MODEL_NAME configures Pi below. Check first whether the whole
      # instruction already has its explicitly requested, verified JSON answer.
      pareto_plan = hamon_pareto.route_from_intake_prompt(
          engineering_inventory,
          instruction=session.instruction,
          mcp_server_count=len(getattr(session, "mcp_servers", None) or []),
          deliverable_count=len(delivery_targets),
          model_api=os.environ.get("MODEL_NAME", ""),
      )
      if pareto_plan and pareto_plan.get("deliver_verified_json"):
        return pareto_plan["presolve_result_json"]
      engineering_cli_note = await engineering_intake.provision_engineering_cli(session)
      # Share one observation allowance before the participant starts. Later
      # delivery verification retains its own existing per-operation limits.
      observation_start = monotonic()
      observation_budget = STARTUP_OBSERVATION_SECONDS
      left = session.time_left()
      if left is not None:
        observation_budget = min(observation_budget, max(0, left - STARTUP_TASK_RESERVE_SECONDS))
      observation_deadline = observation_start + observation_budget
      startup_targets = delivery_targets
      if archive_contract.declared_members(session.instruction, delivery_targets):
        startup_targets = [{**delivery_targets[0], "observe_loose_baseline": True}]
      before_delivery = await delivery_guard.snapshot(
          session, startup_targets, deadline=observation_deadline
      )
      terminal.before = before_delivery
      checked_delivery = bool(delivery_targets) and before_delivery is not None
      archive_plan = await archive_contract.provision(
          session, delivery_targets, deadline=observation_deadline
      ) if checked_delivery else None
      session_dir = pathlib.Path(scratch) / "delivery-session"
      session_args = ["--session-dir", str(session_dir)]
      # A generator written during this task can deterministically finish an
      # exact requested artifact; inherited unchanged scripts are never run.
      materialize_candidates = delivery_materialize.candidates(
          delivery_targets,
          archive_members=archive_contract.declared_members(session.instruction, delivery_targets)
              if engineering_cli_note else (),
          witness=bool(engineering_cli_note and working_copy and terminal.filing
                       and "witness" in session.instruction.lower()
                       and "commissioning" in session.instruction.lower()),
      )
      before_generators = await delivery_guard.snapshot(
          session, delivery_materialize.snapshot_targets(materialize_candidates),
          deadline=observation_deadline,
      ) if checked_delivery else None
      attempted_generators = set()
      delivery_note = (
          "\n\nOUTPUT DELIVERY CONTRACT (literal destinations from the instruction):\n"
          + "\n".join("  " + item["path"] for item in delivery_targets)
          + "\nWrite a complete initial deliverable early, then verify its actual"
            " file integrity and required format. Input examples are NOT outputs."
          if delivery_targets else ""
      )
      if working_copy:
        delivery_note = terminal.copy_instruction(working_copy)
      delivery_note += delivery_materialize.draft_note(materialize_candidates)
      first_request = session.instruction
      if archive_plan:
        first_request += archive_contract.first_instruction(*archive_plan)
      name, model = await asyncio.to_thread(
          configure, agent_dir, relay, session, env
      )
      if pareto_plan and "presolve_result_json" in pareto_plan:
        if "presolve_source_sha256" in pareto_plan:
          first_request += ("\n\nDeterministic presolve for instruction SHA-256 "
                            + pareto_plan["presolve_source_sha256"])
        first_request += ("\n\nDeterministic engineering calculation for the inline request (JSON tool output):\n"
                          + pareto_plan["presolve_result_json"])
      if pareto_plan and "presolve_abstention" in pareto_plan:
        await session.log("passive_fin_presolve: " + pareto_plan["presolve_abstention"])
      thinking_level = pareto_plan["thinking_tier"] if pareto_plan else THINKING
      reserve_window = (
          pareto_plan["delivery_reserve_sec"]
          if (checked_delivery and pareto_plan)
          else (DELIVERY_RESERVE_SEC if checked_delivery else STOP_BEFORE_SEC)
      )
      argv = [
          PI,
          "--print",
          "--mode",
          "json",
          *session_args,
          "-nc",
          "-ns",
          "-np",
          "-na",
          "-e",
          str(HERE / "task-tools.ts"),
          "--provider",
          name,
          "--model",
          model,
          "--thinking",
          thinking_level,
          "--append-system-prompt",
          NOTE.format(workdir=session.workdir) + engineering_inventory + engineering_cli_note + delivery_note,
      ]
      if checked_delivery:
        argv.extend(["-e", str(HERE / "delivery-budget.ts")])
      run = Run(session, model, budgeted=checked_delivery)
      if archive_plan:
        run.archive_watch = archive_contract.FirstPublication(
            archive_plan, target=delivery_targets[0]["path"],
            members=archive_contract.declared_members(session.instruction, delivery_targets),
        )
      done = await taskbridge.run_process(
          argv,
          session,
          env=(delivery_process_env(env, session, run.budget_steps, reserve_window, budget_id=run.budget_id)
               if checked_delivery else env),
          cwd=str(work),
          on_line=run.on_line,
          stdin=first_request.encode(),
          stop_before_sec=reserve_window,
          log_prefix="pi: ",
      )
      first_publication = run.archive_watch
      run.archive_watch = None
      tool_session = None
      if run.tool_turn is not None and not run.invalid_tail:
        tool_session = acknowledged_tool_session(
            session_dir, run.tool_turn, run.observed_results, run.expected_calls, run.history,
            budget_id=run.budget_id)
      can_continue = (run.tool_turn is None or tool_session is not None) and not delivery_budget_exhausted(session, run.budget_steps)
      participant_followup = False
      if delivery_targets and before_delivery is not None:
        after_delivery = await delivery_guard.snapshot(session, delivery_targets)
        terminal.observe(after_delivery)
        issues = delivery_guard.compare(before_delivery, after_delivery)
        if issues and archive_plan:
          # A completed staging directory can be committed even if Pi ended after
          # an unacknowledged tool request. Independently re-read the graded path.
          terminal.invalidate()
          recovery = archive_contract.recovery_command(
              archive_plan[1], delivery_targets[0]["path"], before_delivery, after_delivery)
          if await archive_contract.run_command(session, recovery):
            published = await delivery_guard.snapshot(session, delivery_targets)
            if published:
              terminal.observe(published)
              after_delivery = published
              issues = delivery_guard.compare(before_delivery, published)
        refinement_time = taskbridge.time_share(session.time_left(), STOP_BEFORE_SEC) if archive_plan else 0
        if (archive_plan and not issues and can_continue and run.tool_turn is None
            and all(row.get("state") != "unknown" for row in before_delivery.values())
            and after_delivery and all(row.get("state") == "ready" for row in after_delivery.values())
            and (refinement_time is None or refinement_time >= MIN_DELIVERY_RUN_SEC)
            and await archive_contract.run_command(session, archive_plan[1] + " --checkpoint")
            and ((refinement_time := taskbridge.time_share(session.time_left(), STOP_BEFORE_SEC)) is None
                 or refinement_time >= MIN_DELIVERY_RUN_SEC)):
          stage, command = archive_plan
          await session.log("delivery_gate: complete archive precedes optional refinement")
          resume = any(session_dir.glob("*.jsonl"))
          refine = (
              "A complete archive is on disk. Improve the engineering solution if justified. "
              f"Work only on staged members in {stage}; preserve the graded archive while computing. "
              "After validating an improvement, atomically publish it with:\n" + command +
              "\nIf there is no validated improvement, leave the complete version in place.")
          if not resume:
            refine = session.instruction + "\n\n" + refine
          participant_followup = True
          terminal.invalidate()
          try:
            await taskbridge.run_process(
                [*argv, "--continue"] if resume else argv, session,
                env=delivery_process_env(env, session, run.budget_steps, STOP_BEFORE_SEC, delivery=True, budget_id=run.budget_id), cwd=str(work),
                on_line=run.on_line, stdin=refine.encode(), stop_before_sec=STOP_BEFORE_SEC,
                log_prefix="pi-archive-refine: ")
          except taskbridge.Cancelled:
            raise
          except Exception as exc:
            await session.log("pi-archive-refine: stopped with " + type(exc).__name__)
          recovered = await archive_contract.recover_after_refinement(session, command)
          await session.log("delivery_gate: archive_after_refinement=" + (recovered or "unverified"))
          if recovered in {"published", "restored"}:
            detail = ("The saved complete version was restored." if recovered == "restored"
                      else "The complete staged version was published.")
            session.final_message = (f"Recovered a complete archive at {delivery_targets[0]['path']}. "
                                     + detail + " Required members and supported format checks passed.")
          elif recovered is None:
            session.final_message = (f"The archive at {delivery_targets[0]['path']} could not be "
                                     "verified after the refinement attempt.")
        if issues:
          await session.log("delivery_gate: incomplete=" + str(len(issues)))
          # A cheap task-native deterministic pass precedes another model call.
          if materialize_candidates and before_generators is not None:
            after_generators = await delivery_guard.snapshot(
                session,
                delivery_materialize.snapshot_targets(materialize_candidates),
            )
            selected = delivery_materialize.plan(
                materialize_candidates, issues,
                before_generators, after_generators,
                attempted=attempted_generators,
            )
            if selected:
              terminal.invalidate()
              observed = await delivery_materialize.materialize(session, selected)
              # Unknown execution may already have changed task files. Never
              # automatically repeat that script version after a model repair.
              for item in selected[:len(observed)]:
                attempted_generators.add((
                    item["script"], after_generators[item["script"]].get("fingerprint"),
                ))
              await session.log(
                  "delivery_gate: materializers="
                  + json.dumps(observed, separators=(",",":"))[:800]
              )
              after_delivery = await delivery_guard.snapshot(
                  session, delivery_targets,
              )
              terminal.observe(after_delivery)
              issues = delivery_guard.compare(before_delivery, after_delivery)
              await session.log("delivery_gate: post_materialize=" + str(len(issues)))
          delivery_time = taskbridge.time_share(session.time_left(), STOP_BEFORE_SEC)
          if (issues or (working_copy and terminal.filing)) and can_continue and not delivery_budget_exhausted(session, run.budget_steps) and (delivery_time is None or delivery_time >= MIN_DELIVERY_RUN_SEC):
            # Each task has its own temporary session directory. Continue the
            # actual calculation history if Pi persisted it before the cutoff.
            resume = any(session_dir.glob("*.jsonl"))
            terminal.invalidate()
            instruction = delivery_guard.repair_instruction(issues) if issues else "\n\nThe working copy was recovered locally."
            if working_copy:
              instruction += terminal.copy_instruction(working_copy)
            if archive_plan:
              instruction += archive_contract.first_instruction(*archive_plan)
            if not resume:
              instruction = session.instruction + instruction
            participant_followup = True
            run.archive_watch = first_publication
            await taskbridge.run_process(
                [*argv, "--continue"] if resume else argv,
                session,
                env=delivery_process_env(env, session, run.budget_steps, STOP_BEFORE_SEC, delivery=True, budget_id=run.budget_id),
                cwd=str(work),
                on_line=run.on_line,
                stdin=instruction.encode(),
                stop_before_sec=STOP_BEFORE_SEC,
                log_prefix="pi-delivery-repair: ",
            )
            run.archive_watch = None
            after_repair = await delivery_guard.snapshot(session, delivery_targets)
            terminal.observe(after_repair)
            still_missing = delivery_guard.compare(before_delivery, after_repair)
            if still_missing and archive_plan:
              # The final repair can finish the staged members just before its cutoff.
              # The participant's last turn is not responsible for committing them.
              recovery = archive_contract.recovery_command(
                  archive_plan[1], delivery_targets[0]["path"], before_delivery, after_repair)
              if await archive_contract.run_command(session, recovery):
                after_repair = await delivery_guard.snapshot(session, delivery_targets)
                terminal.observe(after_repair)
                still_missing = delivery_guard.compare(before_delivery, after_repair)
            await session.log(
                "delivery_gate: residual=" + str(len(still_missing))
            )
      # A complete deliverable still needs a final reply after an acknowledged
      # tool turn. Its history proof cannot be reused after another model run.
      if tool_session is not None and not participant_followup and not delivery_budget_exhausted(session, run.budget_steps):
        remaining = taskbridge.time_share(session.time_left(), STOP_BEFORE_SEC)
        if remaining is None or remaining >= MIN_DELIVERY_RUN_SEC:
          terminal.invalidate()
          await session.log("delivery_gate: continuing acknowledged tool results")
          await taskbridge.run_process(
              [*argv, "--session", str(tool_session)], session,
              env=delivery_process_env(env, session, run.budget_steps, STOP_BEFORE_SEC, delivery=True, budget_id=run.budget_id), cwd=str(work),
              on_line=run.on_line, stop_before_sec=STOP_BEFORE_SEC, log_prefix="pi-finalize: ",
              stdin=(b"Continue from the recorded tool results. Check what was actually delivered, "
                     b"finish the requested work within the remaining time, and report its status."))
      # The repair/finalization turn may have written a generator without running
      # it. Recover only fresh versions for known incomplete explicit outputs,
      # then independently inspect the actual artifacts before returning done.
      if materialize_candidates and before_generators is not None:
        left = session.time_left()
        budget = 90 if left is None else min(90, max(0, left - 1))
        final_deadline = monotonic() + budget
        if budget >= 24:
          if not terminal.observed:
            terminal.observe(await delivery_guard.snapshot(
                session, delivery_targets, deadline=final_deadline - 10))
          final_issues = delivery_guard.compare(before_delivery, terminal.after)
          if final_issues:
            generator_probes = delivery_materialize.snapshot_targets(materialize_candidates)
            member_paths = sorted(first_publication.required_outputs) if archive_plan else []
            if len(generator_probes) + len(member_paths) > delivery_guard.MAX_TARGETS:
                member_paths = []
            loose_probes = [{"path": path, "kind": "file"} for path in member_paths]
            after_generators = await delivery_guard.snapshot(
                session, generator_probes + loose_probes, deadline=final_deadline - 10,
            )
            selected = delivery_materialize.plan(
                materialize_candidates, final_issues, before_generators, after_generators,
                attempted=attempted_generators,
            )
            if selected:
              terminal.invalidate()
              observed = await delivery_materialize.materialize(
                  session, selected, reserve_sec=10, deadline=final_deadline,
              )
              if member_paths and any(row["state"] == "executed" for row in observed):
                await archive_contract.publish_generated(
                    session, archive_plan[1], member_paths, after_generators, deadline=final_deadline - 10,
                )
              terminal.observe(await delivery_guard.snapshot(
                  session, delivery_targets, deadline=final_deadline,
              ))
              # Telemetry cannot prevent the independent terminal observation
              # or consume the remaining delivery deadline.
              log_budget = min(1, final_deadline - monotonic())
              if log_budget > 0:
                try:
                  await asyncio.wait_for(session.log("delivery_gate: final_materializers="
                      + json.dumps(observed, separators=(",", ":"))[:800]), log_budget)
                except taskbridge.Cancelled:
                  raise
                except Exception:
                  pass
      if delivery_targets and not terminal.observed:
        left = session.time_left()
        budget = 25 if left is None else min(25, max(0, left - 1))
        if budget >= 1:
          terminal.observe(await delivery_guard.snapshot(session, delivery_targets, deadline=monotonic() + budget))
  if not run.replies:
    raise RuntimeError(
        f"pi exited with {done.code} before the model answered:"
        f" {done.stderr[-2000:]}"
    )
  if run.tool_turn is not None:
    note = "(pi stopped after requesting a tool, before giving a final answer)"
    observation = terminal.summary("")
    return f"{observation}\n{note}" if observation else note
  return terminal.summary(session.final_message or "(pi stopped without a final message)")


def resolve_wired_domain_tools(prompt: str = "", file_paths=()) -> list[str]:
    from hamon_pareto import select_domain_tools
    return list(select_domain_tools(prompt=prompt, file_paths=file_paths))
