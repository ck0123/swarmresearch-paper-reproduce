#!/usr/bin/env node

const fs = require("fs");
const path = require("path");

const required = process.argv.includes("--required");
const fixedCandidates = [
  "/usr/lib/node_modules/pi-subagents/utils.ts",
  "/usr/local/lib/node_modules/pi-subagents/utils.ts",
];

const fixedFunction = `export function getFinalOutput(messages: Message[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (msg.role !== "assistant" || !Array.isArray(msg.content)) continue;

    const text = msg.content
      .filter((part: any) => part.type === "text" && typeof part.text === "string" && part.text.trim().length > 0)
      .map((part: any) => part.text)
      .join("");

    if (text.trim().length > 0) return text;
  }
  return "";
}`;

function patchFile(file) {
  if (!fs.existsSync(file)) return false;

  const source = fs.readFileSync(file, "utf8");
  if (source.includes("part.text.trim().length > 0") && source.includes("return text;")) {
    return true;
  }

  const pattern = /export\s+function\s+getFinalOutput\s*\(\s*messages\s*:\s*Message\[\]\s*\)\s*:\s*string\s*\{[\s\S]*?return "";\s*\}/;
  if (!pattern.test(source)) {
    throw new Error(`Could not find getFinalOutput() in ${file}`);
  }

  fs.writeFileSync(file, source.replace(pattern, fixedFunction));
  return true;
}

function clearJitiCache() {
  const dir = "/tmp/jiti";
  if (!fs.existsSync(dir)) return;

  for (const name of fs.readdirSync(dir)) {
    if (name.startsWith("pi-subagents-utils.") && name.endsWith(".mjs")) {
      fs.rmSync(path.join(dir, name), { force: true });
    }
  }
}

function findUtilsFiles(root) {
  if (!fs.existsSync(root)) return [];
  const results = [];
  const stack = [root];

  while (stack.length > 0) {
    const current = stack.pop();
    let entries;
    try {
      entries = fs.readdirSync(current, { withFileTypes: true });
    } catch {
      continue;
    }

    for (const entry of entries) {
      const fullPath = path.join(current, entry.name);
      if (entry.isDirectory()) {
        if (entry.name === ".git" || entry.name === "npm") continue;
        stack.push(fullPath);
      } else if (
        entry.isFile() &&
        entry.name === "utils.ts" &&
        fullPath.includes("pi-subagents")
      ) {
        results.push(fullPath);
      }
    }
  }

  return results;
}

const candidates = [
  ...fixedCandidates,
  ...findUtilsFiles("/usr/lib/node_modules"),
  ...findUtilsFiles("/usr/local/lib/node_modules"),
  ...findUtilsFiles("/root"),
];

let patched = false;
for (const file of [...new Set(candidates)]) {
  try {
    patched = patchFile(file) || patched;
  } catch (err) {
    console.error(`[patch-pi-subagents] ${err.message}`);
  }
}

if (patched) {
  clearJitiCache();
  console.error("[patch-pi-subagents] getFinalOutput patch is installed");
} else {
  const message = "[patch-pi-subagents] pi-subagents utils.ts not found";
  if (required) {
    console.error(message);
    process.exit(1);
  }
  console.error(`${message}; continuing`);
}
