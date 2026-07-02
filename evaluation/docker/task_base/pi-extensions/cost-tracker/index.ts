import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import type { ExtensionAPI, ExtensionContext } from "@mariozechner/pi-coding-agent";

interface ExtensionConfig {
  summaryPath?: string;
  skipNestedSubagent?: boolean;
}

interface CostValue {
  total?: unknown;
}

interface Usage {
  cost?: number | CostValue | undefined;
}

interface SessionContentItem {
  type?: string;
  text?: string;
}

interface SessionMessage {
  role?: string;
  toolName?: string;
  toolCallId?: string;
  usage?: Usage;
  model?: unknown;
  details?: unknown;
  content?: SessionContentItem[];
}

interface SessionLine {
  id?: string;
  type?: string;
  timestamp?: string;
  parentSession?: string;
  thinkingLevel?: unknown;
  isError?: unknown;
  message?: SessionMessage;
}

interface ParsedSubagentInvocation {
  agent?: string;
  task?: string;
  branch?: string;
  cost: number;
  exitCode?: number;
  timestamp?: number;
  mode?: string;
  runId?: string;
}

interface ParsedSolutionGeneratorInvocation {
  toolCallId?: string;
  branch?: string;
  cost: number;
  model?: string;
  timestamp?: number;
  status?: "success" | "error";
  error?: string;
}

interface SubagentResultShape {
  agent?: unknown;
  task?: unknown;
  branch?: unknown;
  usage?: Usage;
  exitCode?: unknown;
  timestamp?: unknown;
  mode?: unknown;
  runId?: unknown;
}

interface SessionSummary {
  sessionFile: string;
  updatedAt: number;
  sessionKind: "orchestrator" | "solution_generator" | "unknown";
  parentSessionFile?: string;
  orchestratorCost: number;
  subagentInvocationCount: number;
  subagentSum: number;
  subagents: ParsedSubagentInvocation[];
  totalCost: number;
  orchestratorModel?: string;
  solutionGeneratorInvocations?: ParsedSolutionGeneratorInvocation[];
}

interface SummaryDocument {
  version: 1;
  updatedAt: number;
  sessions: Record<string, SessionSummary>;
  aggregates: {
    totalOrchestratorCost: number;
    orchestratorSessionCount: number;
    averageOrchestratorCost: number;
    totalSolutionGeneratorCost: number;
    solutionGeneratorSessionCount: number;
    averageSolutionGeneratorCost: number;
    totalSubagentCost: number;
    subagentInvocationCount: number;
    averageSubagentCost: number;
    totalCost: number;
  };
}

interface ExtensionState {
  summaryPath: string;
  skipNestedSubagent: boolean;
  lastSessionFile?: string;
}

interface ParsedSession {
  lines: SessionLine[];
  sessionFile: string;
  parentSessionFile?: string;
  assistantMessages: Map<string, { cost: number; model?: string }>;
  subagentInvocations: Map<string, ParsedSubagentInvocation>;
  solutionGeneratorInvocations: Map<string, ParsedSolutionGeneratorInvocation>;
}

const parsedSessionCache = new Map<string, ParsedSession>();

function expandTilde(targetPath: string): string {
  if (!targetPath.startsWith("~/")) return targetPath;
  return path.join(os.homedir(), targetPath.slice(2));
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function toNumber(value: unknown): number {
  return isFiniteNumber(value) ? value : 0;
}

function toOptionalNumber(value: unknown): number | undefined {
  return isFiniteNumber(value) ? value : undefined;
}

function getCostFromUsage(usage: unknown): number {
  if (!usage || typeof usage !== "object") return 0;
  const raw = usage as Usage;
  if (isFiniteNumber(raw.cost)) return raw.cost;
  if (!raw.cost || typeof raw.cost !== "object") return 0;
  return toNumber((raw.cost as CostValue).total);
}

function getSummaryConfig(): ExtensionConfig {
  const configPath = path.join(os.homedir(), ".pi", "agent", "extensions", "cost-tracker", "config.json");
  try {
    if (!fs.existsSync(configPath)) return {};
    return JSON.parse(fs.readFileSync(configPath, "utf-8")) as ExtensionConfig;
  } catch {
    return {};
  }
}

function parseSessionLines(sessionFile: string): SessionLine[] {
  if (!fs.existsSync(sessionFile)) return [];
  try {
    return fs
      .readFileSync(sessionFile, "utf-8")
      .split(/\r?\n/)
      .filter((line) => line.trim())
      .flatMap((line) => {
        try {
          const parsed = JSON.parse(line) as SessionLine;
          return parsed && typeof parsed === "object" ? [parsed] : [];
        } catch {
          return [];
        }
      });
  } catch {
    return [];
  }
}

function getSessionsRoot(sessionFile: string): string | undefined {
  const normalized = path.resolve(sessionFile);
  const marker = `${path.sep}.pi${path.sep}agent${path.sep}sessions${path.sep}`;
  const index = normalized.indexOf(marker);
  if (index === -1) return undefined;
  return normalized.slice(0, index + marker.length - 1);
}

function resolveWorkspaceSessionPath(sessionFile: string, workspacePath: string | undefined): string | undefined {
  if (!workspacePath) return undefined;
  const sessionsRoot = getSessionsRoot(sessionFile);
  if (!sessionsRoot) return undefined;
  const normalizedWorkspace = path.posix.normalize(workspacePath);
  const prefix = "/workspace/.pi/agent/sessions";
  if (!normalizedWorkspace.startsWith(prefix)) return undefined;
  const relative = normalizedWorkspace.slice(prefix.length).replace(/^\/+/, "");
  return path.resolve(path.join(sessionsRoot, relative));
}

function parseSubagentFromResult(result: unknown): ParsedSubagentInvocation[] {
  if (!result || typeof result !== "object") return [];
  const parsed = result as SubagentResultShape;
  return [{
    agent: typeof parsed.agent === "string" ? parsed.agent : undefined,
    task: typeof parsed.task === "string" ? parsed.task : undefined,
    branch: typeof parsed.branch === "string" ? parsed.branch : undefined,
    cost: getCostFromUsage(parsed.usage),
    exitCode: toOptionalNumber(parsed.exitCode),
    timestamp: toOptionalNumber(parsed.timestamp),
    mode: typeof parsed.mode === "string" ? parsed.mode : undefined,
    runId: typeof parsed.runId === "string" ? parsed.runId : undefined,
  }];
}

function parseSubagentResults(results: unknown): ParsedSubagentInvocation[] {
  if (!Array.isArray(results)) return parseSubagentFromResult(results);
  return results.flatMap((item) => parseSubagentFromResult(item));
}

function parseSubagentMessageDetails(details: unknown): ParsedSubagentInvocation[] {
  if (!details || typeof details !== "object") return [];
  const data = details as {
    results?: unknown;
    subagents?: unknown;
    outcomes?: unknown;
    usage?: unknown;
    agent?: unknown;
    task?: unknown;
    branch?: unknown;
    exitCode?: unknown;
    mode?: unknown;
    runId?: unknown;
  };

  if (data.results !== undefined) return parseSubagentResults(data.results);
  if (data.subagents !== undefined) return parseSubagentResults(data.subagents);
  if (data.outcomes !== undefined) return parseSubagentResults(data.outcomes);

  return parseSubagentFromResult({
    agent: data.agent,
    task: data.task,
    branch: data.branch,
    usage: data.usage,
    exitCode: data.exitCode,
    mode: data.mode,
    runId: data.runId,
  });
}

function isSubagentToolName(toolName: string | undefined): boolean {
  return (
    toolName === "subagent" ||
    toolName === "managed_subagent" ||
    toolName === "managed_subagents" ||
    toolName === "librarian_subagent"
  );
}

function getToolError(details: unknown): string | undefined {
  if (!details || typeof details !== "object") return undefined;
  const parsed = details as { error?: unknown };
  return typeof parsed.error === "string" ? parsed.error : undefined;
}

function parseSolutionGeneratorInvocation(line: SessionLine): ParsedSolutionGeneratorInvocation | null {
  const message = line.message;
  if (!message || message.role !== "toolResult" || message.toolName !== "solution_generator") return null;
  const details = message.details;
  if (!details || typeof details !== "object") return null;
  const parsed = details as {
    branch?: unknown;
    usage?: unknown;
    model?: unknown;
    timestamp?: unknown;
  };

  const error = getToolError(details);
  const status: "success" | "error" = line.isError === true || !!error ? "error" : "success";

  return {
    toolCallId: typeof message.toolCallId === "string" ? message.toolCallId : undefined,
    branch: typeof parsed.branch === "string" ? parsed.branch : undefined,
    cost: getCostFromUsage(parsed.usage),
    model: typeof parsed.model === "string" ? parsed.model : undefined,
    timestamp: toOptionalNumber(parsed.timestamp),
    status,
    error,
  };
}

function getSubagentKey(invocation: ParsedSubagentInvocation): string {
  if (invocation.runId) {
    return JSON.stringify({
      runId: invocation.runId,
      agent: invocation.agent ?? "",
      branch: invocation.branch ?? "",
      task: invocation.task ?? "",
      mode: invocation.mode ?? "",
      timestamp: invocation.timestamp ?? null,
    });
  }
  return JSON.stringify({
    agent: invocation.agent ?? "",
    branch: invocation.branch ?? "",
    task: invocation.task ?? "",
    cost: invocation.cost,
    exitCode: invocation.exitCode ?? null,
    mode: invocation.mode ?? "",
  });
}

function getSolutionGeneratorKey(invocation: ParsedSolutionGeneratorInvocation): string {
  return JSON.stringify({
    toolCallId: invocation.toolCallId ?? "",
    branch: invocation.branch ?? "",
    cost: invocation.cost,
    model: invocation.model ?? "",
    timestamp: invocation.timestamp ?? null,
    status: invocation.status ?? "success",
    error: invocation.error ?? "",
  });
}

function getMessageText(message: SessionMessage | undefined): string {
  if (!message || !Array.isArray(message.content)) return "";
  return message.content
    .filter((item) => item.type === "text" && typeof item.text === "string")
    .map((item) => item.text ?? "")
    .join("");
}

function classifySession(lines: SessionLine[]): "orchestrator" | "solution_generator" | "unknown" {
  let firstUserText = "";
  let thinkingLevel: string | undefined;

  for (const line of lines) {
    if (!thinkingLevel && line.type === "thinking_level_change" && typeof line.thinkingLevel === "string") {
      thinkingLevel = line.thinkingLevel;
    }
    if (!firstUserText && line.type === "message" && line.message?.role === "user") {
      firstUserText = getMessageText(line.message);
    }
  }

  if (
    firstUserText.includes("## Experiment Task") ||
    firstUserText.includes("When asked to write or improve a C++ solution")
  ) {
    return "solution_generator";
  }
  if (firstUserText.includes("Use the research skill.") || thinkingLevel === "high") {
    return "orchestrator";
  }
  return "unknown";
}

function parseSession(sessionFile: string): ParsedSession {
  const resolvedFile = path.resolve(sessionFile);
  const cached = parsedSessionCache.get(resolvedFile);
  if (cached) return cached;

  const lines = parseSessionLines(resolvedFile);
  let parentSessionFile: string | undefined;
  const assistantMessages = new Map<string, { cost: number; model?: string }>();
  const subagentInvocations = new Map<string, ParsedSubagentInvocation>();
  const solutionGeneratorInvocations = new Map<string, ParsedSolutionGeneratorInvocation>();

  for (const line of lines) {
    if (!parentSessionFile && line.type === "session") {
      parentSessionFile = resolveWorkspaceSessionPath(resolvedFile, line.parentSession);
    }

    if (line.type !== "message") continue;
    const message = line.message;
    if (!message) continue;

    if (message.role === "assistant" && typeof line.id === "string") {
      assistantMessages.set(line.id, {
        cost: getCostFromUsage(message.usage),
        model: typeof message.model === "string" ? message.model : undefined,
      });
    }

    if (message.role === "toolResult" && isSubagentToolName(message.toolName)) {
      for (const invocation of parseSubagentMessageDetails(message.details)) {
        subagentInvocations.set(getSubagentKey(invocation), invocation);
      }
    }

    const sgInvocation = parseSolutionGeneratorInvocation(line);
    if (sgInvocation) {
      solutionGeneratorInvocations.set(getSolutionGeneratorKey(sgInvocation), sgInvocation);
    }
  }

  const parsed: ParsedSession = {
    lines,
    sessionFile: resolvedFile,
    parentSessionFile,
    assistantMessages,
    subagentInvocations,
    solutionGeneratorInvocations,
  };

  parsedSessionCache.set(resolvedFile, parsed);
  return parsed;
}

function readMessageIncrement(parsed: ParsedSession): { cost: number; model?: string } {
  const parent = parsed.parentSessionFile ? parseSession(parsed.parentSessionFile) : undefined;
  const parentIds = parent?.assistantMessages ?? new Map<string, { cost: number; model?: string }>();

  let cost = 0;
  let model: string | undefined;
  for (const [id, info] of parsed.assistantMessages.entries()) {
    if (parentIds.has(id)) continue;
    cost += info.cost;
    if (!model && info.model) model = info.model;
  }

  return { cost, model };
}

function diffMap<T>(current: Map<string, T>, parent: Map<string, T> | undefined): T[] {
  const result: T[] = [];
  for (const [key, value] of current.entries()) {
    if (parent?.has(key)) continue;
    result.push(value);
  }
  return result;
}

function buildSummaryForSession(sessionFile: string): SessionSummary {
  const parsed = parseSession(sessionFile);
  const parent = parsed.parentSessionFile ? parseSession(parsed.parentSessionFile) : undefined;
  const orchestrator = readMessageIncrement(parsed);
  const subagents = diffMap(parsed.subagentInvocations, parent?.subagentInvocations);
  const solutionGeneratorInvocations = diffMap(
    parsed.solutionGeneratorInvocations,
    parent?.solutionGeneratorInvocations,
  );

  const subagentSum = subagents.reduce((sum, subagent) => sum + toNumber(subagent.cost), 0);
  const solutionGeneratorSum = solutionGeneratorInvocations.reduce((sum, invocation) => sum + invocation.cost, 0);

  return {
    sessionFile: parsed.sessionFile,
    updatedAt: Date.now(),
    sessionKind: classifySession(parsed.lines),
    parentSessionFile: parsed.parentSessionFile,
    orchestratorCost: orchestrator.cost,
    orchestratorModel: orchestrator.model,
    subagentInvocationCount: subagents.length,
    subagentSum,
    subagents,
    solutionGeneratorInvocations: solutionGeneratorInvocations.length > 0 ? solutionGeneratorInvocations : undefined,
    totalCost: orchestrator.cost + subagentSum + solutionGeneratorSum,
  };
}

function defaultSummary(): SummaryDocument {
  return {
    version: 1,
    updatedAt: Date.now(),
    sessions: {},
    aggregates: {
      totalOrchestratorCost: 0,
      orchestratorSessionCount: 0,
      averageOrchestratorCost: 0,
      totalSolutionGeneratorCost: 0,
      solutionGeneratorSessionCount: 0,
      averageSolutionGeneratorCost: 0,
      totalSubagentCost: 0,
      subagentInvocationCount: 0,
      averageSubagentCost: 0,
      totalCost: 0,
    },
  };
}

function readExistingSummary(summaryPath: string): SummaryDocument {
  if (!fs.existsSync(summaryPath)) return defaultSummary();
  try {
    const raw = fs.readFileSync(summaryPath, "utf-8");
    const parsed = JSON.parse(raw) as Partial<SummaryDocument>;
    const sessions = parsed.sessions && typeof parsed.sessions === "object" ? parsed.sessions : {};
    const aggregates = parsed.aggregates && typeof parsed.aggregates === "object" ? parsed.aggregates : {};

    return {
      version: 1,
      updatedAt: isFiniteNumber(parsed.updatedAt) ? parsed.updatedAt : Date.now(),
      sessions: sessions as Record<string, SessionSummary>,
      aggregates: {
        totalOrchestratorCost: toNumber(aggregates.totalOrchestratorCost),
        orchestratorSessionCount: Math.max(0, Math.floor(toNumber(aggregates.orchestratorSessionCount))),
        averageOrchestratorCost: toNumber(aggregates.averageOrchestratorCost),
        totalSolutionGeneratorCost: toNumber(aggregates.totalSolutionGeneratorCost),
        solutionGeneratorSessionCount: Math.max(0, Math.floor(toNumber(aggregates.solutionGeneratorSessionCount))),
        averageSolutionGeneratorCost: toNumber(aggregates.averageSolutionGeneratorCost),
        totalSubagentCost: toNumber(aggregates.totalSubagentCost),
        subagentInvocationCount: Math.max(0, Math.floor(toNumber(aggregates.subagentInvocationCount))),
        averageSubagentCost: toNumber(aggregates.averageSubagentCost),
        totalCost: toNumber(aggregates.totalCost),
      },
    };
  } catch {
    return defaultSummary();
  }
}

function writeSummaryAtomic(summaryPath: string, summary: SummaryDocument): void {
  const dir = path.dirname(summaryPath);
  fs.mkdirSync(dir, { recursive: true });
  const temp = `${summaryPath}.${Date.now()}.${Math.random().toString(16).slice(2)}.tmp`;
  summary.updatedAt = Date.now();
  fs.writeFileSync(temp, JSON.stringify(summary, null, 2), "utf-8");
  fs.renameSync(temp, summaryPath);
}

function recalcAggregates(sessions: Record<string, SessionSummary>): SummaryDocument["aggregates"] {
  const aggregates = {
    totalOrchestratorCost: 0,
    orchestratorSessionCount: 0,
    averageOrchestratorCost: 0,
    totalSolutionGeneratorCost: 0,
    solutionGeneratorSessionCount: 0,
    averageSolutionGeneratorCost: 0,
    totalSubagentCost: 0,
    subagentInvocationCount: 0,
    averageSubagentCost: 0,
    totalCost: 0,
  };

  for (const session of Object.values(sessions)) {
    if (!session || typeof session !== "object") continue;
    if (session.sessionKind === "orchestrator" && !session.parentSessionFile) {
      aggregates.totalOrchestratorCost += toNumber(session.orchestratorCost);
      aggregates.orchestratorSessionCount += 1;
      aggregates.totalSubagentCost += toNumber(session.subagentSum);
      aggregates.subagentInvocationCount += Math.max(0, Math.floor(toNumber(session.subagentInvocationCount)));
    }

    if (Array.isArray(session.solutionGeneratorInvocations)) {
      const sgCost = session.solutionGeneratorInvocations.reduce((sum, invocation) => sum + toNumber(invocation.cost), 0);
      aggregates.totalSolutionGeneratorCost += sgCost;
      aggregates.solutionGeneratorSessionCount += session.solutionGeneratorInvocations.length;
    }
  }

  if (aggregates.orchestratorSessionCount > 0) {
    aggregates.averageOrchestratorCost = aggregates.totalOrchestratorCost / aggregates.orchestratorSessionCount;
  }
  if (aggregates.solutionGeneratorSessionCount > 0) {
    aggregates.averageSolutionGeneratorCost =
      aggregates.totalSolutionGeneratorCost / aggregates.solutionGeneratorSessionCount;
  }
  if (aggregates.subagentInvocationCount > 0) {
    aggregates.averageSubagentCost = aggregates.totalSubagentCost / aggregates.subagentInvocationCount;
  }

  aggregates.totalCost =
    aggregates.totalOrchestratorCost + aggregates.totalSolutionGeneratorCost + aggregates.totalSubagentCost;
  return aggregates;
}

function isNestedSubagentSession(skipNested: boolean): boolean {
  if (!skipNested) return false;
  const depth = Number(process.env.PI_SUBAGENT_DEPTH ?? 0);
  return Number.isFinite(depth) && depth > 0;
}

function listWorkspaceSessionFiles(sessionFile: string): string[] {
  const sessionDir = path.dirname(path.resolve(sessionFile));
  try {
    return fs
      .readdirSync(sessionDir)
      .filter((name) => name.endsWith(".jsonl"))
      .map((name) => path.join(sessionDir, name))
      .sort();
  } catch {
    return [path.resolve(sessionFile)];
  }
}

function rebuildWorkspaceSummaries(sessionFile: string, summaryPath: string): void {
  const payload = readExistingSummary(summaryPath);
  parsedSessionCache.clear();
  for (const workspaceSessionFile of listWorkspaceSessionFiles(sessionFile)) {
    payload.sessions[path.resolve(workspaceSessionFile)] = buildSummaryForSession(workspaceSessionFile);
  }
  payload.aggregates = recalcAggregates(payload.sessions);
  writeSummaryAtomic(summaryPath, payload);
}

function summaryPathFromConfig(config: ExtensionConfig): string {
  const configured =
    config.summaryPath ?? path.join(os.homedir(), ".pi", "agent", "extensions", "cost-tracker", "cost-summary.json");
  return expandTilde(configured);
}

function getSessionFile(ctx: ExtensionContext): string | undefined {
  if (!ctx || typeof ctx !== "object") return undefined;
  if (ctx.sessionManager && typeof ctx.sessionManager.getSessionFile === "function") {
    return ctx.sessionManager.getSessionFile() ?? undefined;
  }
  return undefined;
}

export default function registerCostTrackerExtension(pi: ExtensionAPI): void {
  const config = getSummaryConfig();
  const state: ExtensionState = {
    summaryPath: summaryPathFromConfig(config),
    skipNestedSubagent: config.skipNestedSubagent !== false,
  };

  const setCurrentSession = (ctx: ExtensionContext | undefined) => {
    const file = getSessionFile(ctx ?? ({} as ExtensionContext));
    if (file) state.lastSessionFile = file;
  };

  const refreshCurrentSessionSummary = (ctx: ExtensionContext | undefined) => {
    if (isNestedSubagentSession(state.skipNestedSubagent)) return;
    const sessionFile = getSessionFile(ctx ?? ({} as ExtensionContext)) ?? state.lastSessionFile;
    if (!sessionFile) return;
    try {
      rebuildWorkspaceSummaries(sessionFile, state.summaryPath);
    } catch {
      // best-effort incremental update handler
    }
  };

  pi.on("session_start", (_event, ctx) => setCurrentSession(ctx));
  pi.on("session_before_switch", (_event, ctx) => setCurrentSession(ctx));
  pi.on("session_switch", (_event, ctx) => setCurrentSession(ctx));
  pi.on("session_before_fork", (_event, ctx) => setCurrentSession(ctx));
  pi.on("session_fork", (_event, ctx) => setCurrentSession(ctx));
  pi.on("session_before_compact", (_event, ctx) => setCurrentSession(ctx));
  pi.on("session_compact", (_event, ctx) => setCurrentSession(ctx));
  pi.on("message", (_event, ctx) => {
    setCurrentSession(ctx);
    refreshCurrentSessionSummary(ctx);
  });
  pi.on("tool_result", (_event, ctx) => {
    setCurrentSession(ctx);
    refreshCurrentSessionSummary(ctx);
  });
  pi.on("session_shutdown", (_event, ctx) => {
    refreshCurrentSessionSummary(ctx);
  });
}
