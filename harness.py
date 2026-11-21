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
import json
import os
import pathlib
import shutil
import subprocess
import tempfile
from typing import Any

import taskbridge
import engineering_intake
import symbolic_route
import delivery_guard
import delivery_materialize
import hamon_pareto

HERE = pathlib.Path(__file__).resolve().parent
# npm ci, from package-lock.json
PI = shutil.which("pi") or str(HERE / "node_modules" / ".bin" / "pi")
# pi's --thinking level: off, minimal, low, medium, high, xhigh or max
THINKING = "max"
# stop pi this long before the deadline (less on short tasks) so its answer
# arrives
STOP_BEFORE_SEC = 90
# time_share caps this reserve at one quarter of the remaining task time.
DELIVERY_RESERVE_SEC = 30 * 60
MIN_DELIVERY_RUN_SEC = 60
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


class Run:
  """Tracks JSON stream events from a single Pi subprocess run."""

  def __init__(self, session: taskbridge.Session, model: str):
    self.session, self.model, self.replies = session, model, 0

  async def on_event(self, event: dict[str, Any]) -> None:
    """Processes a parsed JSON event from Pi, updating final message and usage.

    Args:
      event: Decoded JSON event dictionary from Pi's stdout stream.
    """
    kind, message = event.get("type"), event.get("message") or {}
    if kind == "message_end" and message.get("role") == "assistant":
      self.replies += 1
      text = "\n".join(
          b.get("text", "")
          for b in message.get("content") or []
          if b.get("type") == "text"
      )
      if text.strip():
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
      discovered = {}
      engineering_inventory = await engineering_intake.intake_for_prompt(session, discovered=discovered)
      engineering_cli_note = await engineering_intake.provision_engineering_cli(session)
      if discovered and engineering_cli_note:
        engineering_cli_note = await symbolic_route.prepare(session, discovered, engineering_cli_note)
      delivery_targets = delivery_guard.extract_targets(
          session.instruction, session.workdir
      )
      before_delivery = await delivery_guard.snapshot(session, delivery_targets)
      checked_delivery = bool(delivery_targets) and before_delivery is not None
      session_dir = pathlib.Path(scratch) / "delivery-session"
      session_args = (
          ["--session-dir", str(session_dir)] if checked_delivery else ["--no-session"]
      )
      # A generator written during this task can deterministically finish an
      # exact requested artifact; inherited unchanged scripts are never run.
      materialize_candidates = delivery_materialize.candidates(delivery_targets)
      before_generators = await delivery_guard.snapshot(
          session, delivery_materialize.snapshot_targets(materialize_candidates)
      )
      delivery_note = (
          "\n\nOUTPUT DELIVERY CONTRACT (literal destinations from the instruction):\n"
          + "\n".join("  " + item["path"] for item in delivery_targets)
          + "\nWrite a complete initial deliverable early, then verify its actual"
            " file integrity and required format. Input examples are NOT outputs."
          if delivery_targets else ""
      )
      name, model = await asyncio.to_thread(
          configure, agent_dir, relay, session, env
      )
      pareto_plan = hamon_pareto.route_from_intake_prompt(
          engineering_inventory,
          mcp_server_count=len(getattr(session, "mcp_servers", None) or []),
          deliverable_count=len(delivery_targets),
          model_api=model,
      )
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
      run = Run(session, model)
      done = await taskbridge.run_process(
          argv,
          session,
          env=env,
          cwd=str(work),
          on_line=run.on_line,
          stdin=session.instruction.encode(),
          stop_before_sec=reserve_window,
          log_prefix="pi: ",
      )
      if delivery_targets and before_delivery is not None:
        after_delivery = await delivery_guard.snapshot(session, delivery_targets)
        issues = delivery_guard.compare(before_delivery, after_delivery)
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
            )
            if selected:
              observed = await delivery_materialize.materialize(session, selected)
              await session.log(
                  "delivery_gate: materializers="
                  + json.dumps(observed, separators=(",",":"))[:800]
              )
              after_delivery = await delivery_guard.snapshot(
                  session, delivery_targets,
              )
              issues = delivery_guard.compare(before_delivery, after_delivery)
              await session.log("delivery_gate: post_materialize=" + str(len(issues)))
          delivery_time = taskbridge.time_share(session.time_left(), STOP_BEFORE_SEC)
          if issues and (delivery_time is None or delivery_time >= MIN_DELIVERY_RUN_SEC):
            # Each task has its own temporary session directory. Continue the
            # actual calculation history if Pi persisted it before the cutoff.
            resume = any(session_dir.glob("*.jsonl"))
            instruction = delivery_guard.repair_instruction(issues)
            if not resume:
              instruction = session.instruction + instruction
            await taskbridge.run_process(
                [*argv, "--continue"] if resume else argv,
                session,
                env=env,
                cwd=str(work),
                on_line=run.on_line,
                stdin=instruction.encode(),
                stop_before_sec=STOP_BEFORE_SEC,
                log_prefix="pi-delivery-repair: ",
            )
            after_repair = await delivery_guard.snapshot(session, delivery_targets)
            still_missing = delivery_guard.compare(before_delivery, after_repair)
            await session.log(
                "delivery_gate: residual=" + str(len(still_missing))
            )
  if not run.replies:
    raise RuntimeError(
        f"pi exited with {done.code} before the model answered:"
        f" {done.stderr[-2000:]}"
    )
  return session.final_message or "(pi stopped without a final message)"


def resolve_wired_domain_tools(prompt: str = "", file_paths=()) -> list[str]:
    from hamon_pareto import select_domain_tools
    return list(select_domain_tools(prompt=prompt, file_paths=file_paths))
