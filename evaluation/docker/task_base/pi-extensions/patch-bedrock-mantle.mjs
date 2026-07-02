/**
 * Patch @mariozechner/pi-ai's amazon-bedrock.js to route Kimi/Moonshot models
 * through bedrock-mantle (OpenAI-compatible endpoint) instead of the Converse API.
 *
 * The Converse API has known tool call parsing bugs for these models (vercel/ai#11409)
 * that cause premature end_turn and lost tool calls in multi-step agentic workflows.
 *
 * Authentication: uses AWS_BEARER_TOKEN_BEDROCK if set, otherwise falls back to
 * SigV4 signing via the standard AWS credential chain (profiles, env vars, IAM roles).
 *
 * Based on: https://github.com/shubham-root/pi-mono/commit/8fe7d1fd879c1be216186dbc4c1e82e1e68ed344
 */
import { execSync } from "child_process";
import { readFileSync, writeFileSync } from "fs";
import { join } from "path";

const npmRoot = execSync("npm root -g").toString().trim();
const bedrockFile = join(
  npmRoot,
  "@mariozechner/pi-coding-agent/node_modules/@mariozechner/pi-ai/dist/providers/amazon-bedrock.js"
);

let content = readFileSync(bedrockFile, "utf8");

// Guard: skip if already patched
if (content.includes("usesMantleRoute")) {
  console.log("amazon-bedrock.js already patched, skipping.");
  process.exit(0);
}

// 1. Add import for openai-completions after transform-messages import
const IMPORT_MARKER = 'import { transformMessages } from "./transform-messages.js";';
if (!content.includes(IMPORT_MARKER)) {
  console.error("ERROR: Could not find import marker in amazon-bedrock.js");
  process.exit(1);
}
content = content.replace(
  IMPORT_MARKER,
  `${IMPORT_MARKER}\nimport { streamOpenAICompletions, streamSimpleOpenAICompletions } from "./openai-completions.js";`
);

// 2. Inject helper functions before streamBedrock
const HELPER_FUNCTIONS = `
function usesMantleRoute(modelId) {
    const lower = modelId.toLowerCase();
    return lower.includes("kimi") || lower.includes("moonshot") || lower.includes("gpt") || lower.includes("nvidia");
}
function buildMantleModel(model, region) {
    return {
        id: model.id.replace(/-[0-9]+:[0-9]+$/, ""),
        name: model.name,
        api: "openai-completions",
        provider: "amazon-bedrock",
        baseUrl: \`https://bedrock-mantle.\${region}.api.aws/v1\`,
        reasoning: model.reasoning,
        input: model.input,
        cost: model.cost,
        contextWindow: model.contextWindow,
        maxTokens: model.maxTokens,
        compat: {
            supportsStore: false,
            supportsDeveloperRole: false,
            supportsReasoningEffort: false,
            maxTokensField: "max_tokens",
            supportsStrictMode: false,
        },
    };
}
function getMantleApiKey() {
    if (typeof process !== "undefined") {
        return process.env.AWS_BEARER_TOKEN_BEDROCK;
    }
    return undefined;
}
function getMantleRegion(options) {
    if (options?.region) return options.region;
    if (typeof process !== "undefined") {
        return process.env.AWS_REGION || process.env.AWS_DEFAULT_REGION || "us-east-1";
    }
    return "us-east-1";
}
function makeMantleErrorMsg(model, detail) {
    return {
        role: "assistant", content: [],
        api: "bedrock-converse-stream",
        provider: model.provider, model: model.id,
        usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
        stopReason: "error",
        errorMessage: detail || "bedrock-mantle request failed",
        timestamp: Date.now(),
    };
}
// SigV4 fetch interceptor for bedrock-mantle requests (used when AWS_BEARER_TOKEN_BEDROCK is not set).
// Intercepts requests to bedrock-mantle.*.api.aws, strips the OpenAI-style Bearer header,
// and replaces it with SigV4 signing via the standard AWS credential chain.
let _mantleSigV4FetchInstalled = false;
async function ensureMantleSigV4Fetch() {
    if (_mantleSigV4FetchInstalled) return;
    _mantleSigV4FetchInstalled = true;
    const [{ defaultProvider }, { SignatureV4 }, { Hash }] = await Promise.all([
        import("@aws-sdk/credential-provider-node"),
        import("@smithy/signature-v4"),
        import("@smithy/hash-node"),
    ]);
    const credProvider = defaultProvider({});
    const origFetch = globalThis.fetch;
    globalThis.fetch = async function mantleSigV4Fetch(input, init) {
        const url = typeof input === "string" ? input : (input instanceof URL ? input.href : input.url);
        if (url && url.includes("bedrock-mantle") && url.includes(".api.aws") && !process.env.AWS_BEARER_TOKEN_BEDROCK) {
            const urlObj = new URL(url);
            const hostname = urlObj.hostname;
            const region = hostname.split(".")[1];
            // Strip OpenAI-style Authorization; SigV4 will supply its own
            const hdrs = new Headers(init?.headers ?? {});
            hdrs.delete("authorization");
            hdrs.set("host", hostname);
            const requestToSign = {
                method: (init?.method || "POST").toUpperCase(),
                hostname,
                path: urlObj.pathname + urlObj.search,
                headers: Object.fromEntries(hdrs.entries()),
                body: init?.body,
            };
            const creds = await credProvider();
            const signer = new SignatureV4({
                credentials: creds,
                region,
                service: "bedrock",
                sha256: Hash.bind(null, "sha256"),
            });
            const signed = await signer.sign(requestToSign);
            return origFetch(input, { ...init, headers: signed.headers });
        }
        return origFetch(input, init);
    };
}
function streamViaMantleSimple(model, context, options) {
    const region = getMantleRegion(options);
    const mantleModel = buildMantleModel(model, region);
    const apiKey = getMantleApiKey();
    if (apiKey) {
        return streamSimpleOpenAICompletions(mantleModel, context, { ...options, apiKey });
    }
    // No bearer token: set up SigV4 fetch interceptor, then stream with a placeholder key
    const stream = new AssistantMessageEventStream();
    ensureMantleSigV4Fetch().then(() => {
        const inner = streamSimpleOpenAICompletions(mantleModel, context, { ...options, apiKey: "sigv4" });
        (async () => { try { for await (const ev of inner) { stream.push(ev); } } finally { stream.end(); } })();
    }).catch((err) => {
        stream.push({ type: "error", reason: "error", error: makeMantleErrorMsg(model, String(err)) });
        stream.end();
    });
    return stream;
}
function streamViaMantle(model, context, options) {
    const region = getMantleRegion(options);
    const mantleModel = buildMantleModel(model, region);
    const apiKey = getMantleApiKey();
    if (apiKey) {
        return streamOpenAICompletions(mantleModel, context, {
            apiKey,
            maxTokens: options?.maxTokens,
            temperature: options?.temperature,
            signal: options?.signal,
            onPayload: options?.onPayload,
        });
    }
    // No bearer token: set up SigV4 fetch interceptor, then stream with a placeholder key
    const stream = new AssistantMessageEventStream();
    ensureMantleSigV4Fetch().then(() => {
        const inner = streamOpenAICompletions(mantleModel, context, {
            apiKey: "sigv4",
            maxTokens: options?.maxTokens,
            temperature: options?.temperature,
            signal: options?.signal,
            onPayload: options?.onPayload,
        });
        (async () => { try { for await (const ev of inner) { stream.push(ev); } } finally { stream.end(); } })();
    }).catch((err) => {
        stream.push({ type: "error", reason: "error", error: makeMantleErrorMsg(model, String(err)) });
        stream.end();
    });
    return stream;
}
`;

const STREAM_BEDROCK_MARKER = "export const streamBedrock = ";
if (!content.includes(STREAM_BEDROCK_MARKER)) {
  console.error("ERROR: Could not find streamBedrock export in amazon-bedrock.js");
  process.exit(1);
}
content = content.replace(STREAM_BEDROCK_MARKER, `${HELPER_FUNCTIONS}\n${STREAM_BEDROCK_MARKER}`);

// 3. Add Moonshot check at start of streamBedrock body
const STREAM_BEDROCK_BODY = "export const streamBedrock = (model, context, options = {}) => {\n    const stream = new AssistantMessageEventStream();";
if (!content.includes(STREAM_BEDROCK_BODY)) {
  console.error("ERROR: Could not find streamBedrock body in amazon-bedrock.js");
  process.exit(1);
}
content = content.replace(
  STREAM_BEDROCK_BODY,
  "export const streamBedrock = (model, context, options = {}) => {\n    if (usesMantleRoute(model.id)) { return streamViaMantle(model, context, options); }\n    const stream = new AssistantMessageEventStream();"
);

// 4. Add Moonshot check at start of streamSimpleBedrock body
const STREAM_SIMPLE_BODY = "export const streamSimpleBedrock = (model, context, options) => {\n    const base = buildBaseOptions(model, options, undefined);";
if (!content.includes(STREAM_SIMPLE_BODY)) {
  console.error("ERROR: Could not find streamSimpleBedrock body in amazon-bedrock.js");
  process.exit(1);
}
content = content.replace(
  STREAM_SIMPLE_BODY,
  "export const streamSimpleBedrock = (model, context, options) => {\n    if (usesMantleRoute(model.id)) { return streamViaMantleSimple(model, context, options); }\n    const base = buildBaseOptions(model, options, undefined);"
);

writeFileSync(bedrockFile, content, "utf8");
console.log("Patched amazon-bedrock.js: Kimi/Moonshot models now route via bedrock-mantle (SigV4 fallback enabled).");
