import type { CSSProperties } from "react";
import type { TelemetryPoint } from "../types";
import { HealthPill } from "./HealthPill";

interface Props {
  point: TelemetryPoint | null;
  onClear: () => void;
}

const grid: CSSProperties = {
  display: "grid",
  gridTemplateColumns: "1fr 1fr",
  gap: 10,
  fontSize: 13,
};

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div style={{ color: "var(--text-muted)", fontSize: 11 }}>{label}</div>
      <div style={{ fontWeight: 600, color: "var(--text-primary)" }}>{value}</div>
    </div>
  );
}

/** Shows every attribute carried on a clicked telemetry point -- this is the
 * answer to "what attributes exist" and "is there an anomaly at this point,"
 * pinned (not just a hover tooltip) so it survives moving the mouse away. */
export function PointDetailPanel({ point, onClear }: Props) {
  if (!point) {
    return (
      <p style={{ color: "var(--text-secondary)", fontSize: 13 }}>
        Click any point on the chart to inspect its full detection detail.
      </p>
    );
  }

  const stillWarmingUp = point.forecast === null;

  return (
    <div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 10,
        }}
      >
        <span style={{ fontSize: 13, color: "var(--text-secondary)" }}>index {point.index}</span>
        <button
          onClick={onClear}
          style={{
            fontSize: 11,
            padding: "3px 10px",
            borderRadius: 6,
            border: "1px solid var(--border)",
            background: "var(--surface-1)",
            color: "var(--text-secondary)",
            cursor: "pointer",
          }}
        >
          Clear
        </button>
      </div>

      {stillWarmingUp && (
        <p style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 0 }}>
          This point is before the first full 100-step window -- detection hadn't started yet
          (score shown as 0).
        </p>
      )}

      <div style={grid}>
        <Field label="Telemetry value" value={point.value.toFixed(4)} />
        <Field label="Reconstruction error" value={point.reconstruction_error.toFixed(4)} />
        <Field label="Ensemble score" value={`${point.ensemble_score.toFixed(2)}x threshold`} />
        <Field label="Anomaly flag" value={point.is_anomaly ? "Yes" : "No"} />
        <Field
          label="Ground truth (labeled)"
          value={point.ground_truth_anomaly ? "Anomalous" : "Normal"}
        />
        <Field
          label="Predicted time-to-anomaly"
          value={
            point.predicted_steps_to_anomaly !== null
              ? `~${point.predicted_steps_to_anomaly} steps`
              : "None in horizon"
          }
        />
      </div>

      <div style={{ marginTop: 12 }}>
        <HealthPill state={point.health_state} />
      </div>
    </div>
  );
}
