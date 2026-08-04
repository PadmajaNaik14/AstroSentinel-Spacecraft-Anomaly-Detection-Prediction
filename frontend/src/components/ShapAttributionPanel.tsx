import {
  Bar,
  BarChart,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ExplainResponse } from "../types";

/** Diverging blue<->red polarity encoding (push-toward-anomaly vs
 * push-toward-normal), the genuine SHAP output (TreeExplainer on the
 * Isolation Forest). The reconstruction-error heatmap below is explicitly
 * NOT SHAP -- a separate, clearly labeled technique (see plan.md). */
export function ShapAttributionPanel({ explain }: { explain: ExplainResponse | undefined }) {
  if (!explain) {
    return (
      <p style={{ color: "var(--text-secondary)" }}>
        Waiting for an anomalous point to explain...
      </p>
    );
  }

  const data = explain.feature_names.map((name, i) => ({
    name,
    shap: explain.shap_values[i],
  }));

  return (
    <div>
      <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 4 }}>SHAP Feature Attribution</div>
      <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 8 }}>
        Isolation Forest, engineered rolling-error features
      </div>
      <ResponsiveContainer width="100%" height={180}>
        <BarChart data={data} layout="vertical" margin={{ left: 8, right: 16 }}>
          <XAxis
            type="number"
            stroke="var(--baseline)"
            tick={{ fill: "var(--text-muted)", fontSize: 11 }}
          />
          <YAxis
            type="category"
            dataKey="name"
            stroke="var(--baseline)"
            tick={{ fill: "var(--text-secondary)", fontSize: 11 }}
            width={110}
          />
          <ReferenceLine x={0} stroke="var(--baseline)" />
          <Tooltip
            contentStyle={{
              background: "var(--surface-1)",
              border: "1px solid var(--border)",
              fontSize: 12,
            }}
          />
          <Bar dataKey="shap" isAnimationActive={false}>
            {data.map((d, i) => (
              <Cell key={i} fill={d.shap >= 0 ? "var(--series-8)" : "var(--series-1)"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      <div style={{ marginTop: 12 }}>
        <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 4, color: "var(--text-secondary)" }}>
          Reconstruction-error heatmap (not SHAP)
        </div>
        <div style={{ display: "flex", height: 16, borderRadius: 4, overflow: "hidden" }}>
          {explain.reconstruction_error_heatmap.map((v, i) => (
            <div
              key={i}
              title={`t-${explain.reconstruction_error_heatmap.length - i}: ${v.toFixed(2)}`}
              style={{
                flex: 1,
                background: `color-mix(in srgb, var(--status-critical) ${Math.round(v * 100)}%, var(--surface-1))`,
              }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
