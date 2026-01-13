import {
  getWebInstrumentations,
  initializeFaro,
  type Faro,
} from "@grafana/faro-web-sdk";
import { TracingInstrumentation } from "@grafana/faro-web-tracing";

let faro: Faro | null = null;
let initPromise: Promise<Faro | null> | null = null;

interface FrontendConfig {
  faroUrl: string | null;
  version: string;
}

async function fetchConfig(): Promise<FrontendConfig> {
  try {
    const response = await fetch("/api/config");
    if (!response.ok) {
      throw new Error(`Config fetch failed: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.debug("Failed to fetch frontend config:", error);
    return { faroUrl: null, version: "0.0.0" };
  }
}

export async function initFaro(): Promise<Faro | null> {
  if (faro) {
    return faro;
  }

  if (initPromise) {
    return initPromise;
  }

  initPromise = (async () => {
    const config = await fetchConfig();

    if (!config.faroUrl) {
      console.debug("Faro disabled: faroUrl not configured");
      return null;
    }

    faro = initializeFaro({
      url: config.faroUrl,
      app: {
        name: "contextmine-web",
        version: config.version,
        environment: import.meta.env.MODE,
      },
      instrumentations: [
        ...getWebInstrumentations({
          captureConsole: true,
          captureConsoleDisabledLevels: [],
        }),
        new TracingInstrumentation({
          instrumentationOptions: {
            propagateTraceHeaderCorsUrls: [
              new RegExp(`${window.location.origin}/api/.*`),
              new RegExp(`${window.location.origin}/mcp/.*`),
            ],
          },
        }),
      ],
    });

    console.debug("Faro initialized", { collectorUrl: config.faroUrl });
    return faro;
  })();

  return initPromise;
}

export function getFaro(): Faro | null {
  return faro;
}
