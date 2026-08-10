import type { CSSProperties } from "react";
import { card, sectionLabel } from "../styles/ui";

interface Props {
  hasActivePoint: boolean;
  stillWarmingUp: boolean;
  predictedStepsToAnomaly: number | null;
  forecastValues: number[] | null;
  currentValue: number | null;
  horizon: number;
  anomalyClass?: string;
  sourceIndex?: number;
  isSelected: boolean;
}

// The dataset has no failure-type labels (NASA anonymizes what each channel
// physically measures) -- so instead of inventing fake diagnoses ("battery
// failure"), we describe the *shape* of deviation this channel has
// historically shown, which is real, labeled information (labeled_anomalies.csv).
const ANOMALY_CLASS_HINT: Record<string, string> = {
  point: "a sudden, sharp deviation -- this channel's past anomalies were spike-type, not gradual",
  contextual:
    "a gradual drift out of its normal pattern -- this channel's past anomalies were contextual, not sharp spikes",
  mixed: "either a sudden spike or a gradual drift -- this channel has shown both anomaly types historically",
};

function trendDescription(values: number[], current: number): string {
  const delta = values[values.length - 1] - values[0];
  const scale = Math.max(Math.abs(current), 1e-6);
  const relative = delta / scale;
  if (Math.abs(relative) < 0.02) return "holding roughly steady";
  return relative > 0 ? "trending upward" : "trending downward";
}

const chipBase: CSSProperties = {
  display: "inline-flex",
  alignItems: "center",
  gap: 8,
  padding: "6px 12px",
  borderRadius: "var(--radius-md)",
  fontWeight: 600,
  fontSize: 13,
};

/** Countdown + plain-language readout for the GRU forecaster's
 * predicted-time-to-anomaly. Follows whichever point is active -- the
 * clicked/pinned point if there is one, otherwise the live latest point --
 * same pattern as the SHAP panel, so clicking a point on the chart actually
 * changes what this panel says instead of always showing the live forecast. */
export function ForecastPanel({
  hasActivePoint,
  stillWarmingUp,
  predictedStepsToAnomaly,
  forecastValues,
  currentValue,
  horizon,
  anomalyClass,
  sourceIndex,
  isSelected,
}: Props) {
  const atRisk = predictedStepsToAnomaly !== null;

  return (
    <div style={card}>
      <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 4 }}>Failure Prediction</div>
      {hasActivePoint && (
        <div style={{ ...sectionLabel, marginBottom: 10 }}>
          {sourceIndex !== undefined ? `Index ${sourceIndex} · ` : ""}
          {isSelected ? "selected point" : "live"}
        </div>
      )}

      {!hasActivePoint ? (
        <p style={{ color: "var(--text-secondary)", fontSize: 13, margin: 0 }}>Waiting for data...</p>
      ) : stillWarmingUp ? (
        <p style={{ color: "var(--text-muted)", fontSize: 12, margin: 0 }}>
          This point is before the first full 100-step window -- no forecast was computed yet.
        </p>
      ) : atRisk ? (
        <>
          <div
            style={{
              ...chipBase,
              background: "color-mix(in srgb, var(--status-serious) 16%, var(--surface-1))",
              color: "var(--status-serious)",
            }}
          >
            ▲ Predicted anomaly in ~{predictedStepsToAnomaly} step
            {predictedStepsToAnomaly === 1 ? "" : "s"}
          </div>
          <p style={{ fontSize: 12, color: "var(--text-secondary)", marginTop: 10, marginBottom: 0 }}>
            Likely {anomalyClass ? ANOMALY_CLASS_HINT[anomalyClass] : "an anomaly"} -- based on the GRU
            forecaster's projected next {horizon} steps re-scored through the trained detector, not a
            specific failure diagnosis (this dataset doesn't label failure types).
          </p>
        </>
      ) : (
        <>
          <div
            style={{
              ...chipBase,
              background: "color-mix(in srgb, var(--status-good) 14%, var(--surface-1))",
              color: "var(--status-good)",
            }}
          >
            ● No anomaly predicted in the next {horizon} steps
          </div>
          {forecastValues && forecastValues.length > 0 && currentValue !== null && (
            <p style={{ fontSize: 12, color: "var(--text-secondary)", marginTop: 10, marginBottom: 0 }}>
              Forecast trend: {trendDescription(forecastValues, currentValue)} ({currentValue.toFixed(3)}{" "}
              &rarr; {forecastValues[forecastValues.length - 1].toFixed(3)}).
            </p>
          )}
        </>
      )}
    </div>
  );
}
