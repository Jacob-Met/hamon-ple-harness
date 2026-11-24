/** Reserve delivery by deferring a complete proposed tool batch before execution. */
export class DeliveryBudget {
  constructor(config, clock = () => Date.now()) {
    this.clock = clock;
    this.steps = Number.isInteger(config.completed_steps) && config.completed_steps >= 0 ? config.completed_steps : 0;
    this.maxSteps = Number.isInteger(config.max_steps) && config.max_steps > 0 ? config.max_steps : null;
    this.delivery = config.phase === "delivery";
    this.phaseDeadline = Number.isFinite(config.phase_deadline_ms) ? config.phase_deadline_ms : null;
    this.taskDeadline = Number.isFinite(config.task_deadline_ms) ? config.task_deadline_ms : null;
    this.lastBoundary = this.clock();
    this.durations = [];
    this.expected = new Set();
    this.decided = false;
    this.reason = null;
    this.blocked = new Set();
  }

  assistant(message) {
    if (message?.role !== "assistant") return;
    this.steps += 1;
    this.expected = new Set();
    this.decided = false;
    this.reason = null;
    this.blocked.clear();
    const calls = (message.content ?? []).filter(item => item.type === "toolCall");
    const ids = calls.map(call => call.id);
    if (message.stopReason === "toolUse" && ids.every(id => typeof id === "string" && id) &&
        new Set(ids).size === ids.length) this.expected = new Set(ids);
  }

  beforeTool(callId) {
    if (!this.expected.has(callId)) return null;
    if (this.decided) return this.reason;
    // Decide once for the whole batch. Later siblings cannot be interrupted
    // merely because an earlier sibling consumed more time than predicted.
    this.decided = true;
    const limit = this.maxSteps === null ? null
      : this.delivery ? this.maxSteps : Math.max(1, this.maxSteps - 4);
    if (limit !== null && this.steps >= limit) {
      this.reason = this.delivery ? "step limit" : "delivery step reserve";
    } else {
      this.reason = this.timeReason(this.clock());
    }
    return this.reason;
  }

  timeReason(now) {
    if (this.phaseDeadline === null) return null;
    const nextTurn = Math.max(0, now - this.lastBoundary, ...this.durations);
    const cutoff = !this.delivery && this.taskDeadline !== null
      ? Math.min(this.phaseDeadline, this.taskDeadline - 4 * nextTurn)
      : this.phaseDeadline;
    return now + nextTurn >= cutoff
      ? (this.delivery ? "final time reserve" : "delivery time reserve") : null;
  }

  finish(event) {
    const results = event.toolResults ?? [];
    const observed = new Set(results.map(result => result.toolCallId));
    if (!this.expected.size || observed.size !== results.length ||
        results.length !== this.expected.size || !results.every(result => result.role === "toolResult") ||
        ![...this.expected].every(id => observed.has(id))) return false;
    if (!this.decided) this.beforeTool(this.expected.values().next().value);
    const now = this.clock(), elapsed = now - this.lastBoundary;
    // A first slow tool can exceed every earlier observation. Once all
    // siblings have genuine results, hand off before requesting another turn.
    this.reason ??= this.timeReason(now);
    this.lastBoundary = now;
    if (elapsed > 0 && Number.isFinite(elapsed)) {
      this.durations.push(elapsed);
      if (this.durations.length > 4) this.durations.shift();
    }
    return true;
  }
}

export default function registerDeliveryBudget(pi) {
  let config;
  try { config = JSON.parse(process.env.PLE_DELIVERY_BUDGET ?? "null"); }
  catch { return; }
  if (!config || !["work", "delivery"].includes(config.phase)) return;
  const startWall = Date.now(), startMono = performance.now();
  const budget = new DeliveryBudget(config, () => startWall + performance.now() - startMono);
  pi.on("message_end", event => { budget.assistant(event.message); });
  pi.on("tool_call", event => {
    const reason = budget.beforeTool(event.toolCallId);
    if (reason) {
      budget.blocked.add(event.toolCallId);
      return {block: true, terminate: true,
        reason: "The " + reason + " was reached. This operation was deferred; continue in the delivery pass."};
    }
  });
  pi.on("turn_end", (event, context) => {
    const complete = budget.finish(event);
    if (!complete || !budget.reason) return;
    process.stderr.write("delivery_budget: " + budget.reason + "\n");
    // Pi validates unknown tools and malformed arguments before tool_call.
    // Their results cannot carry our termination vote. Stop only after every
    // declared result is recorded, and mark the exact private-session boundary.
    if (budget.blocked.size !== budget.expected.size && typeof config.budget_id === "string") {
      pi.appendEntry("ple_delivery_boundary", {budget_id: config.budget_id,
        tool_call_ids: [...budget.expected].sort(), phase: config.phase, steps: budget.steps});
      context.abort();
    }
  });
}
