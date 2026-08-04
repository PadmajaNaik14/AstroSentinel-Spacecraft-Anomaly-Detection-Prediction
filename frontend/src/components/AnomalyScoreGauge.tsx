import { RadialBar, RadialBarChart, ResponsiveContainer } from "recharts";
import { HEALTH_META } from "../healthState";
import type { HealthState } from "../types";
import { HealthPill } from "./HealthPill";

interface Props {
  score: number;
  healthState: HealthState;
}

/** Score is threshold-relative (1.0 == at the ensemble threshold); the gauge
 * caps its visual fill at 3x threshold so a wild outlier doesn't blow past
 * the arc, while the hero number underneath always shows the true value. */
export function AnomalyScoreGauge({ score, healthState }: Props) {
  const capped = Math.min(score, 3);
  const data = [{ name: "score", value: capped, fill: `var(${HEALTH_META[healthState].colorVar})` }];

  return (
    <div style={{ textAlign: "center" }}>
      <div style={{ height: 140 }}>
        <ResponsiveContainer width="100%" height="100%">
          <RadialBarChart
            innerRadius="70%"
            outerRadius="100%"
            data={data}
            startAngle={225}
            endAngle={-45}
          >
            <RadialBar dataKey="value" background={{ fill: "var(--gridline)" }} cornerRadius={4} />
          </RadialBarChart>
        </ResponsiveContainer>
      </div>
      <div style={{ marginTop: -56, fontSize: 28, fontWeight: 700, color: "var(--text-primary)" }}>
        {score.toFixed(2)}x
      </div>
      <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 8 }}>
        of anomaly threshold
      </div>
      <HealthPill state={healthState} />
    </div>
  );
}
