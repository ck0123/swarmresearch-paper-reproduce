import { isToolCallEventType, type ExtensionAPI } from "@mariozechner/pi-coding-agent";

const DEFAULT_TIMEOUT_SECONDS = 1000;

function configuredTimeoutSeconds(): number {
  const raw = process.env.PI_BASH_TIMEOUT_SECONDS;
  if (!raw) return DEFAULT_TIMEOUT_SECONDS;

  const parsed = Number.parseInt(raw, 10);
  if (!Number.isFinite(parsed) || parsed < 1) return DEFAULT_TIMEOUT_SECONDS;
  return parsed;
}

export default function bashTimeoutExtension(pi: ExtensionAPI) {
  pi.on("tool_call", (event) => {
    if (!isToolCallEventType("bash", event)) return;

    const timeout = configuredTimeoutSeconds();
    const current = event.input.timeout;
    if (typeof current !== "number" || !Number.isFinite(current) || current < 1 || current > timeout) {
      event.input.timeout = timeout;
    }
  });
}
