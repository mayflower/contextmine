import {
  getWebInstrumentations,
  initializeFaro,
  type Faro,
} from "@grafana/faro-web-sdk";
import { TracingInstrumentation } from "@grafana/faro-web-tracing";

import { api, apiData } from "./api/client";

let faro: Faro | null = null;
let initPromise: Promise<Faro | null> | null = null;

async function fetchConfig() {
  try {
    return await apiData(api.GET("/api/config"));
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
      // Faro 2.x moved console level config off getWebInstrumentations; an empty
      // disabledLevels list captures all console levels (was captureConsoleDisabledLevels: []).
      consoleInstrumentation: {
        disabledLevels: [],
      },
      instrumentations: [
        ...getWebInstrumentations({
          captureConsole: true,
        }),
        new TracingInstrumentation({
          instrumentationOptions: {
            propagateTraceHeaderCorsUrls: [
              new RegExp(`${globalThis.location.origin}/api/.*`),
              new RegExp(`${globalThis.location.origin}/mcp/.*`),
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
