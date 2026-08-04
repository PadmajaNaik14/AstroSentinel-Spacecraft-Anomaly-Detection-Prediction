import type { CSSProperties } from "react";
import {
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceArea,
  ReferenceDot,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { AnomalyWindow, HealthState, TelemetryPoint } from "../types";

interface Props {
  points: TelemetryPoint[];
  groundTruthWindows: AnomalyWindow[];
  forecast?: { baseIndex: number; values: number[] } | null;
  selectedIndex?: number | null;
  onPointClick?: (point: TelemetryPoint) => void;
}

interface Row {
  index: number;
  value: number | null;
  forecastValue: number | null;
  reconstructionError: number | null;
  ensembleScore: number | null;
  isAnomaly: boolean | null;
  healthState: HealthState | null;
  groundTruthAnomaly: boolean | null;
}

const tooltipStyle: CSSProperties = {
  background: "var(--surface-1)",
  border: "1px solid var(--border)",
  borderRadius: 6,
  padding: "8px 10px",
  fontSize: 12,
  color: "var(--text-primary)",
  lineHeight: 1.6,
};

function ChartTooltip({ active, payload }: { active?: boolean; payload?: { payload: Row }[] }) {
  if (!active || !payload || payload.length === 0) return null;
  const row = payload[0].payload;

  if (row.value === null) {
    return (
      <div style={tooltipStyle}>
        <div style={{ fontWeight: 700 }}>index {row.index} &middot; forecast</div>
        <div>predicted value: {row.forecastValue?.toFixed(3)}</div>
      </div>
    );
  }

  return (
    <div style={tooltipStyle}>
      <div style={{ fontWeight: 700, marginBottom: 2 }}>index {row.index}</div>
      <div>value: {row.value.toFixed(3)}</div>
      <div>reconstruction error: {row.reconstructionError?.toFixed(3)}</div>
      <div>ensemble score: {row.ensembleScore?.toFixed(2)}x threshold</div>
      <div>anomaly flag: {row.isAnomaly ? "yes" : "no"}</div>
      <div>health state: {row.healthState}</div>
      <div>ground truth: {row.groundTruthAnomaly ? "labeled anomaly" : "normal"}</div>
      <div style={{ marginTop: 4, color: "var(--text-muted)", fontStyle: "italic" }}>
        click to pin &amp; explain
      </div>
    </div>
  );
}

/** Single-hue telemetry line (categorical slot 1). Anomaly windows shaded as
 * ReferenceAreas: predicted in status-serious wash, ground truth in a
 * status-critical hairline band underneath -- distinguishable without adding
 * a second hue to the status set. Forecast is the same series continued as a
 * dashed segment, not a new color. Clicking a real (non-forecast) point pins
 * it via onPointClick -- the parent renders its full attributes and drives
 * the SHAP panel from the selection. */
export function TelemetryChart({
  points,
  groundTruthWindows,
  forecast,
  selectedIndex,
  onPointClick,
}: Props) {
  if (points.length === 0) {
    return <p style={{ color: "var(--text-secondary)" }}>No telemetry yet.</p>;
  }

  const minIndex = points[0].index;
  const maxIndex = points[points.length - 1].index;

  const rows: Row[] = points.map((p) => ({
    index: p.index,
    value: p.value,
    forecastValue: null,
    reconstructionError: p.reconstruction_error,
    ensembleScore: p.ensemble_score,
    isAnomaly: p.is_anomaly,
    healthState: p.health_state,
    groundTruthAnomaly: p.ground_truth_anomaly,
  }));

  if (forecast && forecast.values.length > 0) {
    const boundary = rows[rows.length - 1];
    if (boundary) boundary.forecastValue = boundary.value;
    forecast.values.forEach((v, i) => {
      rows.push({
        index: forecast.baseIndex + i + 1,
        value: null,
        forecastValue: v,
        reconstructionError: null,
        ensembleScore: null,
        isAnomaly: null,
        healthState: null,
        groundTruthAnomaly: null,
      });
    });
  }

  const predictedRuns: { start: number; end: number }[] = [];
  let runStart: number | null = null;
  points.forEach((p, i) => {
    if (p.is_anomaly && runStart === null) runStart = p.index;
    if (!p.is_anomaly && runStart !== null) {
      predictedRuns.push({ start: runStart, end: points[i - 1].index });
      runStart = null;
    }
  });
  if (runStart !== null) {
    predictedRuns.push({ start: runStart, end: points[points.length - 1].index });
  }

  const visibleGroundTruth = groundTruthWindows.filter(
    (w) => w.end >= minIndex && w.start <= maxIndex
  );

  const selectedRow =
    selectedIndex != null ? rows.find((r) => r.index === selectedIndex && r.value !== null) : null;

  const handleClick = (state: { activePayload?: { payload: Row }[] }) => {
    if (!onPointClick) return;
    const row = state?.activePayload?.[0]?.payload;
    if (!row || row.value === null) return; // forecast points have no detection attributes
    const original = points.find((p) => p.index === row.index);
    if (original) onPointClick(original);
  };

  return (
    <ResponsiveContainer width="100%" height={280}>
      <ComposedChart
        data={rows}
        margin={{ top: 8, right: 16, bottom: 0, left: 0 }}
        onClick={handleClick}
        style={{ cursor: onPointClick ? "pointer" : undefined }}
      >
        <CartesianGrid stroke="var(--gridline)" vertical={false} />
        <XAxis
          dataKey="index"
          type="number"
          domain={["dataMin", "dataMax"]}
          stroke="var(--baseline)"
          tick={{ fill: "var(--text-muted)", fontSize: 11 }}
        />
        <YAxis stroke="var(--baseline)" tick={{ fill: "var(--text-muted)", fontSize: 11 }} />
        <Tooltip content={<ChartTooltip />} />
        {visibleGroundTruth.map((w, i) => (
          <ReferenceArea
            key={`gt-${i}`}
            x1={Math.max(w.start, minIndex)}
            x2={Math.min(w.end, maxIndex)}
            fill="var(--status-critical)"
            fillOpacity={0.08}
            stroke="var(--status-critical)"
            strokeOpacity={0.3}
          />
        ))}
        {predictedRuns.map((r, i) => (
          <ReferenceArea
            key={`pred-${i}`}
            x1={r.start}
            x2={r.end}
            fill="var(--status-serious)"
            fillOpacity={0.18}
          />
        ))}
        <Line
          type="monotone"
          dataKey="value"
          stroke="var(--series-1)"
          strokeWidth={2}
          dot={false}
          isAnimationActive={false}
          connectNulls={false}
        />
        <Line
          type="monotone"
          dataKey="forecastValue"
          stroke="var(--series-1)"
          strokeWidth={2}
          strokeDasharray="4 3"
          dot={false}
          isAnimationActive={false}
          connectNulls
        />
        {selectedRow && selectedRow.value !== null && (
          <ReferenceDot
            x={selectedRow.index}
            y={selectedRow.value}
            r={6}
            fill="var(--series-1)"
            stroke="var(--surface-1)"
            strokeWidth={2}
          />
        )}
      </ComposedChart>
    </ResponsiveContainer>
  );
}
