import type { CSSProperties } from "react";
import { sectionLabel } from "../styles/ui";
import type { TelemetryPoint } from "../types";
import { HealthPill } from "./HealthPill";

interface Props {
  point: TelemetryPoint | null;
  onClear: () => void;
}

const grid: CSSProperties = {
  display: "grid",
  gridTemplateColumns: "1fr 1fr",
  gap: 12,
  fontSize: 13,
};

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div style={sectionLabel}>{label}</div>
      <div style={{ fontWeight: 600, color: "var(--text-primary)", marginTop: 2 }}>{value}</div>
    </div>
  );
}

/** Shows every attribute carried on a clicked telemetry point -- this is the
 * answer to "what attributes exist" and "is there an anomaly at this point,"
 * pinned (not just a hover tooltip) so it survives moving the mouse away. */
export function PointDetailPanel({ point, onClear }: Props) {
  if (!point) {
    return (
      <p style={{ color: "var(--text-secondary)", fontSize: 13, margin: 0 }}>
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
        <span style={sectionLabel}>Index {point.index}</span>
        <button className="btn btn-sm btn-ghost" onClick={onClear}>
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
