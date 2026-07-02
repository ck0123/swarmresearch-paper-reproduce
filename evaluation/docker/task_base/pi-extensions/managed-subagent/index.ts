import { execFileSync, spawn } from "node:child_process";
import * as fs from "node:fs";
import * as path from "node:path";
import type { ExtensionAPI, ExtensionContext } from "@mariozechner/pi-coding-agent";
import { Type } from "@sinclair/typebox";

const DEFAULT_WORKER_MODEL = "minimax.minimax-m2.5";
const DEFAULT_WORKER_THINKING = "high";
const DEFAULT_WORKER_TIMEOUT_MS = 900_000;
const DEFAULT_WORKER_INSTRUCTIONS = `# Instructions

You are a researcher working on open research problem. Your goal is to write a solution program in one-shot. If you miss any step or do not follow its rules, your response will be rejected.

You are working in a designated git worktree. IMPORTANT GUARDRAIL: Do not access or modify any other worktree, branch checkout, or repository path.

**STEP 1: Read prompt.md, findings.md (if it exists), and any initial programs in your branch**

**STEP 2: Implement the experiment in one-shot**
- One idea per experiment. Implement the idea efficiently. 

IMPORTANT GUARDRAIL: Do NOT execute your program locally to get an initial score or fix bugs. You MUST only use the standard evaluator to execute your program. 

**Step 3: Create/edit findings.md as needed**
- \`findings.md\` concisely records failures and successes. It is strictly factual and neutral. 
- Only make edits if you learned new facts. Otherwise, keep as is.

IMPORTANT GUARDRAIL: Do NOT discuss next steps or suggestions on what is necessary to improve the score. Simply record the basic facts of your solution and its evaluation. 

**STEP 4: Run the standard evaluator once**
- You are only allowed to run the evaluator once.
- Run exactly one final command:
  \`\`\`
  ./task-eval --commit-message "concise experiment description"
  \`\`\`
- Do NOT run the evaluation with a timeout.
- Do NOT include "| score =" in your commit message. Just a concise description of the experiment.
- Do not run \`git commit\` manually. The evaluator will commit your workspace with the evaluated score.
- After the evaluator prints its output, do not continue, do not run any commands, and do not write any further messages. This marks the end of your experiment.
`;
const MANAGED_SUBAGENT_PROMPT_FILENAME = "managed_subagent_prompt.md";

interface Usage {
  cost?: number | { total?: number } | undefined;
}

interface WorkerResult {
  text: string;
  cost: number;
  model?: string;
  rawStdout: string;
  stderr: string;
  exitCode: number | null;
  signal: string | null;
}

interface SessionManagerLike {
  getSessionFile?: () => string | null | undefined;
}

interface ManagedSubagentDetails {
  mode: string;
  runId: string;
  branch: string;
  exitCode: number | null;
  signal: string | null;
  usage: { cost: number };
  model?: string;
  initialCommitSha?: string;
  branchAdvanced: boolean;
  commitSha?: string;
  commitMessage?: string;
  artifacts: {
    inputPath: string;
    outputPath: string;
    metadataPath: string;
  };
  error?: string;
}

interface ManagedSubagentOutcome {
  branchName: string;
  summary: string;
  details: ManagedSubagentDetails;
  isError: boolean;
}

function getWorkerModel(): string | undefined {
  return process.env.MANAGED_SUBAGENT_MODEL ?? process.env.PI_MODEL ?? DEFAULT_WORKER_MODEL;
}

function getProvider(): string | undefined {
  return process.env.MANAGED_SUBAGENT_PROVIDER ?? process.env.PI_PROVIDER ?? "amazon-bedrock";
}

function getThinking(): string {
  return process.env.MANAGED_SUBAGENT_THINKING ?? DEFAULT_WORKER_THINKING;
}

function getWorkerTimeoutMs(): number {
  const raw = process.env.MANAGED_SUBAGENT_TIMEOUT_MS;
  if (!raw) return DEFAULT_WORKER_TIMEOUT_MS;
  const n = parseInt(raw, 10);
  return Number.isFinite(n) && n > 0 ? n : DEFAULT_WORKER_TIMEOUT_MS;
}

function envFlagDisabled(name: string): boolean {
  return ["0", "false", "no", "off"].includes((process.env[name] ?? "").trim().toLowerCase());
}

function getPiAgentDir(): string {
  return process.env.PI_CODING_AGENT_DIR ?? path.join(process.env.HOME ?? "/agent-home", ".pi");
}

function readConfiguredWorkerInstructions(): string {
  const configuredPath = process.env.MANAGED_SUBAGENT_PROMPT_PATH;
  const promptPath = configuredPath && configuredPath.trim()
    ? configuredPath.trim()
    : path.join(getPiAgentDir(), MANAGED_SUBAGENT_PROMPT_FILENAME);

  try {
    const text = fs.readFileSync(promptPath, "utf-8").trim();
    if (text) return text;
  } catch {
    // The config prompt is optional; fall back to the built-in default.
  }

  return DEFAULT_WORKER_INSTRUCTIONS;
}

function gitExec(args: string[], cwd: string): string {
  return execFileSync("git", args, { cwd, encoding: "utf-8" });
}

function ensureDir(dir: string): void {
  fs.mkdirSync(dir, { recursive: true });
}

function randomRunId(): string {
  return Math.random().toString(16).slice(2, 10);
}

function safeBranchName(branchName: string): boolean {
  return /^[A-Za-z0-9][A-Za-z0-9._/-]*$/.test(branchName) && !branchName.includes("..") && !branchName.endsWith("/");
}

function branchExists(cwd: string, branchName: string): boolean {
  try {
    execFileSync("git", ["rev-parse", "--verify", `refs/heads/${branchName}`], { cwd, stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
}

function currentBranch(cwd: string): string {
  return gitExec(["branch", "--show-current"], cwd).trim();
}

function ensureWorktree(cwd: string, branchName: string): string {
  const worktreePath = path.join(cwd, ".worktrees", branchName);
  ensureDir(path.dirname(worktreePath));

  if (fs.existsSync(worktreePath)) {
    const checkedOut = currentBranch(worktreePath);
    if (checkedOut !== branchName) {
      throw new Error(`existing worktree ${worktreePath} is on ${checkedOut || "detached HEAD"}, expected ${branchName}`);
    }
    return worktreePath;
  }

  gitExec(["worktree", "add", worktreePath, branchName], cwd);
  return worktreePath;
}

function removeWorktree(cwd: string, worktreePath: string): void {
  try {
    gitExec(["worktree", "remove", "--force", worktreePath], cwd);
  } catch {
    fs.rmSync(worktreePath, { recursive: true, force: true });
  }
}

function readPromptMd(cwd: string): string {
  try {
    return fs.readFileSync(path.join(cwd, "prompt.md"), "utf-8");
  } catch {
    return "";
  }
}

function stripOrchestratorInstructions(prompt: string): string {
  const markers = [
    "\n---\n\n# Swarm Research Instructions",
    "\n---\n\n# Managed Swarm Research Instructions",
    "\n---\n\nYou are a research orchestrator working on an open research problem.",
  ];
  const markerIndexes = markers
    .map((marker) => prompt.indexOf(marker))
    .filter((index) => index !== -1);
  if (markerIndexes.length === 0) return prompt;
  return prompt.slice(0, Math.min(...markerIndexes)).trimEnd();
}

function buildWorkerPrompt(cwd: string, worktreePath: string, branchName: string): string {
  const promptMd = stripOrchestratorInstructions(readPromptMd(cwd));
  return [
    `Your assigned branch is \`${branchName}\`.`,
    `Your git worktree is \`${worktreePath}\`.`,
    "This worktree has already been created for you.",
    "Run all commands from that worktree only.",
    "",
    readConfiguredWorkerInstructions(),
    promptMd ? `\n---\n\n# Problem\n\n${promptMd}` : "",
  ].join("\n");
}

function getCostFromUsage(usage: unknown): number {
  if (!usage || typeof usage !== "object") return 0;
  const raw = usage as Usage;
  if (typeof raw.cost === "number" && Number.isFinite(raw.cost)) return raw.cost;
  if (raw.cost && typeof raw.cost === "object" && typeof raw.cost.total === "number" && Number.isFinite(raw.cost.total)) {
    return raw.cost.total;
  }
  return 0;
}

function getPiInvocation(args: string[]): { command: string; args: string[] } {
  const currentScript = process.argv[1];
  if (currentScript && fs.existsSync(currentScript)) {
    return { command: process.execPath, args: [currentScript, ...args] };
  }
  const execName = path.basename(process.execPath).toLowerCase();
  return /^(node|bun)(\.exe)?$/.test(execName) ? { command: "pi", args } : { command: process.execPath, args };
}

function getSessionFile(ctx: ExtensionContext): string | undefined {
  const manager = (ctx as { sessionManager?: SessionManagerLike }).sessionManager;
  if (manager && typeof manager.getSessionFile === "function") return manager.getSessionFile() ?? undefined;
  return undefined;
}

function getArtifactsDir(ctx: ExtensionContext, cwd: string): string {
  const sessionFile = getSessionFile(ctx);
  if (sessionFile) return path.join(path.dirname(sessionFile), "managed-subagent-artifacts");
  return path.join(cwd, ".pi", "agent", "sessions", "--workspace--", "managed-subagent-artifacts");
}

function writeArtifact(filePath: string, content: string): void {
  ensureDir(path.dirname(filePath));
  fs.writeFileSync(filePath, content, "utf-8");
}

function latestCommit(cwd: string, branchName: string): { sha: string; subject: string } | undefined {
  try {
    return {
      sha: gitExec(["rev-parse", branchName], cwd).trim(),
      subject: gitExec(["log", "-1", "--format=%s", branchName], cwd).trim(),
    };
  } catch {
    return undefined;
  }
}

function appendCommitMessageToFindings(worktreePath: string, commitSubject: string): boolean {
  const findingsPath = path.join(worktreePath, "findings.md");
  const existing = fs.existsSync(findingsPath) ? fs.readFileSync(findingsPath, "utf-8") : "";
  if (existing.includes(commitSubject)) return false;

  const entry = `- ${commitSubject}\n`;
  const nextContent = existing.trim()
    ? `${existing.endsWith("\n") ? existing : `${existing}\n`}${entry}`
    : `# Findings\n\n${entry}`;
  fs.writeFileSync(findingsPath, nextContent, "utf-8");
  gitExec(["add", "findings.md"], worktreePath);
  gitExec(["commit", "--amend", "--no-edit"], worktreePath);
  return true;
}

function clampConcurrency(raw: unknown, taskCount: number): number {
  if (taskCount <= 0) return 0;
  const n = typeof raw === "number" && Number.isFinite(raw) ? Math.floor(raw) : taskCount;
  return Math.max(1, Math.min(n, taskCount));
}

async function runWorker(prompt: string, worktreePath: string, stateDir: string, signal?: AbortSignal): Promise<WorkerResult> {
  const model = getWorkerModel();
  const provider = getProvider();
  const args = [
    "--mode", "json",
    "-p",
    prompt,
    "--thinking", getThinking(),
    "--no-extensions",
    "--no-skills",
    "--no-prompt-templates",
    "--no-themes",
    "--session-dir", path.join(stateDir, "sessions"),
  ];
  if (model) args.push("--model", model);
  if (provider) args.push("--provider", provider);

  ensureDir(stateDir);
  ensureDir(path.join(stateDir, "sessions"));

  return new Promise((resolve, reject) => {
    const invocation = getPiInvocation(args);
    const home = process.env.HOME ?? "/agent-home";
    const env = {
      ...process.env,
      HOME: home,
      PWD: worktreePath,
      PI_CODING_AGENT_DIR: path.join(stateDir, ".pi"),
      AWS_SHARED_CREDENTIALS_FILE:
        process.env.AWS_SHARED_CREDENTIALS_FILE ?? path.join(home, ".aws", "credentials"),
      AWS_CONFIG_FILE: process.env.AWS_CONFIG_FILE ?? path.join(home, ".aws", "config"),
      TASK_EVAL_AUTO_COMMIT: envFlagDisabled("MANAGED_SUBAGENT_AUTO_COMMIT") ? "0" : "1",
      TASK_EVAL_HARD_STOP_AFTER_COMMIT: envFlagDisabled("MANAGED_SUBAGENT_HARD_STOP_AFTER_COMMIT") ? "0" : "1",
      MANAGED_SUBAGENT_ACTIVE: "1",
    };
    const proc = spawn(invocation.command, invocation.args, {
      cwd: worktreePath,
      env,
      stdio: ["pipe", "pipe", "pipe"],
      shell: false,
    });

    let buffer = "";
    let rawStdout = "";
    let stderr = "";
    let lastAssistantText = "";
    let lastAssistantCost = 0;
    let totalCost = 0;
    const seen = new Set<string>();
    const timeoutMs = getWorkerTimeoutMs();
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      proc.kill("SIGTERM");
      setTimeout(() => {
        if (!proc.killed) proc.kill("SIGKILL");
      }, 5000).unref?.();
    }, timeoutMs);

    const processLine = (line: string) => {
      if (!line.trim()) return;
      let event: any;
      try {
        event = JSON.parse(line);
      } catch {
        return;
      }
      if ((event.type !== "message" && event.type !== "message_end") || !event.message) return;
      const message = event.message;
      if (message.role !== "assistant" || !Array.isArray(message.content)) return;
      if (typeof message.errorMessage === "string" && message.errorMessage.trim()) {
        lastAssistantText = `Error: ${message.errorMessage.trim()}`;
      }
      const text = message.content
        .filter((part: any) => part.type === "text" && typeof part.text === "string" && part.text.trim().length > 0)
        .map((part: any) => part.text)
        .join("");
      if (text.trim()) lastAssistantText = text.trim();
      lastAssistantCost = getCostFromUsage(message.usage);
      if (event.type === "message_end") {
        const key = typeof message.id === "string" ? message.id : `${seen.size}:${lastAssistantCost}`;
        if (!seen.has(key)) {
          seen.add(key);
          totalCost += lastAssistantCost;
        }
      }
    };

    proc.stdout.on("data", (data: Buffer) => {
      const chunk = data.toString();
      rawStdout += chunk;
      buffer += chunk;
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) processLine(line);
    });
    proc.stderr.on("data", (data: Buffer) => {
      stderr += data.toString();
    });
    proc.on("close", (code, signalName) => {
      clearTimeout(timer);
      if (buffer.trim()) processLine(buffer);
      resolve({
        text: timedOut ? `${lastAssistantText}\nTimed out after ${timeoutMs} ms`.trim() : lastAssistantText,
        cost: totalCost || lastAssistantCost,
        model,
        rawStdout,
        stderr,
        exitCode: timedOut ? -1 : code,
        signal: signalName,
      });
    });
    proc.on("error", (err) => {
      clearTimeout(timer);
      reject(err);
    });
    proc.stdin.on("error", () => {});
    proc.stdin.end();

    if (signal) {
      const kill = () => proc.kill("SIGTERM");
      if (signal.aborted) kill();
      else signal.addEventListener("abort", kill, { once: true });
    }
  });
}

async function runManagedSubagent(cwd: string, branchName: string, artifactsDir: string, signal?: AbortSignal): Promise<ManagedSubagentOutcome> {
  const runId = randomRunId();
  const artifactPrefix = path.join(artifactsDir, `${runId}_${branchName.replace(/[^A-Za-z0-9_.-]/g, "_")}`);
  const inputArtifact = `${artifactPrefix}_input.md`;
  const outputArtifact = `${artifactPrefix}_output.md`;
  const metaArtifact = `${artifactPrefix}_meta.json`;
  const stateDir = `${artifactPrefix}_state`;
  let worktreePath = "";
  let result: WorkerResult | undefined;
  let error: string | undefined;
  let initialCommit: { sha: string; subject: string } | undefined;
  let commit: { sha: string; subject: string } | undefined;

  try {
    if (!safeBranchName(branchName)) throw new Error(`unsafe branch_name: ${branchName}`);
    if (!branchExists(cwd, branchName)) throw new Error(`branch does not exist: ${branchName}`);
    initialCommit = latestCommit(cwd, branchName);
    worktreePath = ensureWorktree(cwd, branchName);
    const prompt = buildWorkerPrompt(cwd, worktreePath, branchName);
    writeArtifact(inputArtifact, prompt);
    result = await runWorker(prompt, worktreePath, stateDir, signal);
    writeArtifact(outputArtifact, result.text || result.stderr || result.rawStdout);
    commit = latestCommit(cwd, branchName);
    if (commit?.sha && commit.sha !== initialCommit?.sha) {
      appendCommitMessageToFindings(worktreePath, commit.subject);
      commit = latestCommit(cwd, branchName);
    }
  } catch (err) {
    error = err instanceof Error ? err.message : String(err);
    writeArtifact(outputArtifact, error);
  } finally {
    if (worktreePath) removeWorktree(cwd, worktreePath);
  }

  commit = commit ?? latestCommit(cwd, branchName);
  const branchAdvanced = Boolean(commit?.sha && commit.sha !== initialCommit?.sha);
  const details: ManagedSubagentDetails = {
    mode: "managed",
    runId,
    branch: branchName,
    exitCode: result?.exitCode ?? null,
    signal: result?.signal ?? null,
    usage: { cost: result?.cost ?? 0 },
    model: result?.model,
    initialCommitSha: initialCommit?.sha,
    branchAdvanced,
    commitSha: commit?.sha,
    commitMessage: commit?.subject,
    artifacts: {
      inputPath: inputArtifact,
      outputPath: outputArtifact,
      metadataPath: metaArtifact,
    },
    error,
  };
  writeArtifact(metaArtifact, JSON.stringify(details, null, 2));

  const outputDiagnostic = result ? (result.text || result.stderr || result.rawStdout).trim().slice(0, 1200) : "";
  const workerLooksBroken = Boolean(result && !branchAdvanced && (result.exitCode !== 0 || result.cost === 0));
  const diagnostic = workerLooksBroken
    ? `\nWorker exited with code ${result?.exitCode ?? "unknown"}, cost ${result?.cost ?? 0}${outputDiagnostic ? `\n${outputDiagnostic}` : ""}`
    : "";
  const summary = error
    ? `managed_subagent failed for ${branchName}: ${error}`
    : `managed_subagent completed for ${branchName}${branchAdvanced && commit ? `\n${commit.sha} ${commit.subject}` : `\nNo new commit detected${diagnostic}`}`;

  return {
    branchName,
    summary,
    details,
    isError: Boolean(error || workerLooksBroken),
  };
}

async function runManagedSubagentBatch(
  cwd: string,
  branchNames: string[],
  concurrency: number,
  artifactsDir: string,
  signal?: AbortSignal,
): Promise<ManagedSubagentOutcome[]> {
  const outcomes = new Array<ManagedSubagentOutcome>(branchNames.length);
  let nextIndex = 0;
  const workers = Array.from({ length: concurrency }, async () => {
    while (nextIndex < branchNames.length) {
      const index = nextIndex++;
      outcomes[index] = await runManagedSubagent(cwd, branchNames[index], artifactsDir, signal);
    }
  });
  await Promise.all(workers);
  return outcomes;
}

export default function managedSubagentExtension(pi: ExtensionAPI): void {
  pi.registerTool({
    name: "managed_subagent",
    label: "Managed Subagent",
    description: "Run one SwarmResearch researcher in an existing branch worktree with task-eval auto-commit and hard-stop.",
    parameters: Type.Object({
      branch_name: Type.String({
        description: "Existing experiment branch name to run in .worktrees/<branch_name>",
      }),
    }),
    async execute(_toolCallId, params, signal, _onUpdate, ctx) {
      const branchName = (params as { branch_name: string }).branch_name;
      const cwd = ctx.cwd;
      const artifactsDir = getArtifactsDir(ctx, cwd);
      const outcome = await runManagedSubagent(cwd, branchName, artifactsDir, signal);
      return {
        content: [{ type: "text", text: outcome.summary }],
        details: outcome.details,
        isError: outcome.isError,
      };
    },
  });

  pi.registerTool({
    name: "managed_subagents",
    label: "Managed Subagents",
    description: "Run multiple SwarmResearch researchers in existing branch worktrees, with bounded parallelism.",
    parameters: Type.Object({
      tasks: Type.Array(
        Type.Object({
          branch_name: Type.String({
            description: "Existing experiment branch name to run in .worktrees/<branch_name>",
          }),
        }),
        {
          minItems: 1,
          description: "Researchers to launch. Each task must contain only branch_name.",
        },
      ),
      concurrency: Type.Optional(Type.Number({
        description: "Maximum researchers to run in parallel. Defaults to all tasks.",
      })),
    }),
    async execute(_toolCallId, params, signal, _onUpdate, ctx) {
      const raw = params as { tasks: Array<{ branch_name: string }>; concurrency?: number };
      const branchNames = raw.tasks.map((task) => task.branch_name);
      const duplicate = branchNames.find((branchName, index) => branchNames.indexOf(branchName) !== index);
      if (duplicate) {
        const text = `managed_subagents failed: duplicate branch_name ${duplicate}`;
        return {
          content: [{ type: "text", text }],
          details: { error: text, branchNames },
          isError: true,
        };
      }

      const concurrency = clampConcurrency(raw.concurrency, branchNames.length);
      const outcomes = await runManagedSubagentBatch(ctx.cwd, branchNames, concurrency, getArtifactsDir(ctx, ctx.cwd), signal);
      const totalCost = outcomes.reduce((sum, outcome) => sum + outcome.details.usage.cost, 0);
      const advanced = outcomes.filter((outcome) => outcome.details.branchAdvanced).length;
      const failed = outcomes.filter((outcome) => outcome.isError).length;
      const summaryLines = [
        `managed_subagents completed ${outcomes.length} researchers (concurrency ${concurrency})`,
        `advanced=${advanced} failed=${failed} cost=${totalCost}`,
        ...outcomes.map((outcome) => outcome.summary),
      ];
      return {
        content: [{ type: "text", text: summaryLines.join("\n\n") }],
        details: {
          mode: "managed-batch",
          concurrency,
          usage: { cost: totalCost },
          outcomes: outcomes.map((outcome) => outcome.details),
        },
        isError: failed > 0,
      };
    },
  });
}
