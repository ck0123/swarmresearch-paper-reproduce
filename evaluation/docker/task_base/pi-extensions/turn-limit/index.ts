import * as fs from "node:fs";
import type { ExtensionAPI, ExtensionContext } from "@mariozechner/pi-coding-agent";

const DEFAULT_HARD_LIMIT_TURN = 20;
const EXIT_DELAY_MS = 250;

interface SessionState {
  baseTurnCount: number | null;
  turnCount: number;
  warnedTurns: Set<number>;
  terminating: boolean;
}

interface SessionMessage {
  role?: string;
}

interface SessionLine {
  type?: string;
  message?: SessionMessage;
}

interface SessionManagerLike {
  getSessionFile?: () => string | null | undefined;
}

interface ContextLike {
  sessionManager?: SessionManagerLike;
}

const sessionStates = new Map<string, SessionState>();

function hardLimitTurn(): number {
  const rawValue = process.env.PI_TURN_LIMIT ?? "";
  const parsed = Number.parseInt(rawValue, 10);
  if (Number.isFinite(parsed) && parsed > 0) return parsed;
  return DEFAULT_HARD_LIMIT_TURN;
}

function warningTurn(): number {
  return Math.max(1, hardLimitTurn() - 10);
}

function getSessionFile(ctx: ExtensionContext | ContextLike | undefined): string | undefined {
  if (!ctx || typeof ctx !== "object") return undefined;
  const sessionManager = ctx.sessionManager;
  if (!sessionManager || typeof sessionManager.getSessionFile !== "function") return undefined;
  return sessionManager.getSessionFile() ?? undefined;
}

function defaultState(): SessionState {
  return {
    baseTurnCount: null,
    turnCount: 0,
    warnedTurns: new Set<number>(),
    terminating: false,
  };
}

function readTurnCount(sessionFile: string): number {
  if (!fs.existsSync(sessionFile)) return 0;

  let turns = 0;
  const raw = fs.readFileSync(sessionFile, "utf-8");
  for (const line of raw.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed) continue;

    let parsed: SessionLine;
    try {
      parsed = JSON.parse(trimmed) as SessionLine;
    } catch {
      continue;
    }

    if (parsed.type !== "message") continue;
    const message = parsed.message;
    if (!message || message.role !== "assistant") continue;
    turns += 1;
  }

  return turns;
}

function buildWarningText(): string {
  const limit = hardLimitTurn();
  const turnsLeft = Math.max(1, limit - warningTurn());
  return [
    `You have ${turnsLeft} turns left before the ${limit}-turn limit.`,
    "Finish the experiment now and run exactly one final command:",
    './task-eval --commit-message "concise experiment description"',
    "Do not run any more local tests or exploratory commands.",
  ].join(" ");
}

function sendSteerWarning(pi: ExtensionAPI): void {
  pi.sendUserMessage(buildWarningText(), { deliverAs: "steer" });
}

function scheduleTermination(state: SessionState): void {
  if (state.terminating) return;
  state.terminating = true;

  const timer = setTimeout(() => {
    process.exit(0);
  }, EXIT_DELAY_MS);

  timer.unref?.();
}

function refreshTurnLimit(pi: ExtensionAPI, ctx: ExtensionContext | undefined): void {
  const sessionFile = getSessionFile(ctx);
  if (!sessionFile) return;

  const state = sessionStates.get(sessionFile) ?? defaultState();
  const rawTurnCount = readTurnCount(sessionFile);
  if (state.baseTurnCount === null) {
    state.baseTurnCount = rawTurnCount;
  }
  const turnCount = Math.max(0, rawTurnCount - state.baseTurnCount);
  const warning = warningTurn();

  if (turnCount >= warning && !state.warnedTurns.has(warning)) {
    sendSteerWarning(pi);
    state.warnedTurns.add(warning);
  }

  state.turnCount = Math.max(state.turnCount, turnCount);

  if (turnCount >= hardLimitTurn()) {
    scheduleTermination(state);
  }

  sessionStates.set(sessionFile, state);
}

export default function registerTurnLimitExtension(pi: ExtensionAPI): void {
  const captureSession = (_ctx: ExtensionContext | undefined) => {
    // Intentionally no-op; session state is keyed lazily on refresh.
  };

  pi.on("session_start", (_event, ctx) => captureSession(ctx));
  pi.on("session_before_switch", (_event, ctx) => captureSession(ctx));
  pi.on("session_switch", (_event, ctx) => captureSession(ctx));
  pi.on("session_before_fork", (_event, ctx) => captureSession(ctx));
  pi.on("session_fork", (_event, ctx) => captureSession(ctx));
  pi.on("session_before_compact", (_event, ctx) => captureSession(ctx));
  pi.on("session_compact", (_event, ctx) => captureSession(ctx));

  pi.on("message", (_event, ctx) => {
    refreshTurnLimit(pi, ctx);
  });

  pi.on("tool_result", (_event, ctx) => {
    refreshTurnLimit(pi, ctx);
  });

  pi.on("session_shutdown", (_event, ctx) => {
    refreshTurnLimit(pi, ctx);
  });
}
