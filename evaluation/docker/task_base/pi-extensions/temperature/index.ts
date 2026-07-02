import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";

export default function temperatureExtension(pi: ExtensionAPI) {
  pi.registerFlag("temperature", {
    description: "Model temperature (e.g. --temperature 0.7)",
    type: "string",
  });

  pi.on("before_provider_request", (event) => {
    const raw = (pi.getFlag("temperature") as string | undefined) ?? process.env.PI_TEMPERATURE ?? "1.0";

    const temperature = parseFloat(raw);
    if (!isFinite(temperature)) return;

    return { ...(event.payload as Record<string, unknown>), temperature };
  });
}
