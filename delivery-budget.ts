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
  }

  assistant(message) {
    if (message?.role !== "assistant") return;
    this.steps += 1;
    this.expected = new Set();
    this.decided = false;
    this.reason = null;
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
    } else if (this.phaseDeadline !== null) {
      const now = this.clock();
      const nextTurn = Math.max(0, now - this.lastBoundary, ...this.durations);
      const cutoff = !this.delivery && this.taskDeadline !== null
        ? Math.min(this.phaseDeadline, this.taskDeadline - 4 * nextTurn)
        : this.phaseDeadline;
      if (now + nextTurn >= cutoff) {
        this.reason = this.delivery ? "final time reserve" : "delivery time reserve";
      }
    }
    return this.reason;
  }

  finish(event) {
    const results = event.toolResults ?? [];
    const observed = new Set(results.map(result => result.toolCallId));
    if (!this.expected.size || observed.size !== results.length ||
        results.length !== this.expected.size || !results.every(result => result.role === "toolResult") ||
        ![...this.expected].every(id => observed.has(id))) return;
    const now = this.clock(), elapsed = now - this.lastBoundary;
    this.lastBoundary = now;
    if (elapsed > 0 && Number.isFinite(elapsed)) {
      this.durations.push(elapsed);
      if (this.durations.length > 4) this.durations.shift();
    }
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
    if (reason) return {block: true, terminate: true,
      reason: "The " + reason + " was reached. This operation was deferred; continue in the delivery pass."};
  });
  pi.on("turn_end", event => {
    budget.finish(event);
    if (budget.reason) process.stderr.write("delivery_budget: " + budget.reason + "\n");
  });
}
