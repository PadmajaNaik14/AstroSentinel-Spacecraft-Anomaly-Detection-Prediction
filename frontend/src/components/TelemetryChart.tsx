import {
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceArea,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { AnomalyWindow, TelemetryPoint } from "../types";

interface Props {
  points: TelemetryPoint[];
  groundTruthWindows: AnomalyWindow[];
  forecast?: { baseIndex: number; values: number[] } | null;
}

interface Row {
  index: number;
  value: number | null;
  forecastValue: number | null;
}

/** Single-hue telemetry line (categorical slot 1). Anomaly windows shaded as
 * ReferenceAreas: predicted in status-serious wash, ground truth in a
 * status-critical hairline band underneath -- distinguishable without adding
 * a second hue to the status set. Forecast is the same series continued as a
 * dashed segment, not a new color. */
export function TelemetryChart({ points, groundTruthWindows, forecast }: Props) {
  if (points.length === 0) {
    return <p style={{ color: "var(--text-secondary)" }}>No telemetry yet.</p>;
  }

  const minIndex = points[0].index;
  const maxIndex = points[points.length - 1].index;

  const rows: Row[] = points.map((p) => ({ index: p.index, value: p.value, forecastValue: null }));

  if (forecast && forecast.values.length > 0) {
    const boundary = rows[rows.length - 1];
    if (boundary) boundary.forecastValue = boundary.value;
    forecast.values.forEach((v, i) => {
      rows.push({ index: forecast.baseIndex + i + 1, value: null, forecastValue: v });
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

  return (
    <ResponsiveContainer width="100%" height={280}>
      <ComposedChart data={rows} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
        <CartesianGrid stroke="var(--gridline)" vertical={false} />
        <XAxis
          dataKey="index"
          type="number"
          domain={["dataMin", "dataMax"]}
          stroke="var(--baseline)"
          tick={{ fill: "var(--text-muted)", fontSize: 11 }}
        />
        <YAxis stroke="var(--baseline)" tick={{ fill: "var(--text-muted)", fontSize: 11 }} />
        <Tooltip
          contentStyle={{
            background: "var(--surface-1)",
            border: "1px solid var(--border)",
            fontSize: 12,
          }}
          labelStyle={{ color: "var(--text-secondary)" }}
        />
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
      </ComposedChart>
    </ResponsiveContainer>
  );
}
