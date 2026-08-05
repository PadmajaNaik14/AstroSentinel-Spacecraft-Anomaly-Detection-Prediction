import { card } from "../styles/ui";

interface Props {
  predictedStepsToAnomaly: number | null;
  horizon: number;
}

/** Countdown chip for the GRU forecaster's predicted-time-to-anomaly. The
 * forecast trajectory itself is rendered as a dashed continuation of the
 * telemetry line in TelemetryChart -- this panel is the plain-language readout. */
export function ForecastPanel({ predictedStepsToAnomaly, horizon }: Props) {
  const atRisk = predictedStepsToAnomaly !== null;

  return (
    <div style={card}>
      <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 10 }}>Failure Prediction</div>
      {atRisk ? (
        <div
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 8,
            padding: "6px 12px",
            borderRadius: 8,
            background: "color-mix(in srgb, var(--status-serious) 16%, var(--surface-1))",
            color: "var(--status-serious)",
            fontWeight: 600,
            fontSize: 13,
          }}
        >
          ▲ Predicted anomaly in ~{predictedStepsToAnomaly} step
          {predictedStepsToAnomaly === 1 ? "" : "s"}
        </div>
      ) : (
        <div
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 8,
            padding: "6px 12px",
            borderRadius: 8,
            background: "color-mix(in srgb, var(--status-good) 14%, var(--surface-1))",
            color: "var(--status-good)",
            fontWeight: 600,
            fontSize: 13,
          }}
        >
          ● No anomaly predicted in the next {horizon} steps
        </div>
      )}
    </div>
  );
}
