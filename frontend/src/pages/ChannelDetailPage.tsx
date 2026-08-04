import { useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { useParams } from "react-router-dom";
import { api } from "../api/client";
import { AnomalyScoreGauge } from "../components/AnomalyScoreGauge";
import { ForecastPanel } from "../components/ForecastPanel";
import { ShapAttributionPanel } from "../components/ShapAttributionPanel";
import { TelemetryChart } from "../components/TelemetryChart";
import { useChannelStream } from "../hooks/useChannelStream";

const card: React.CSSProperties = {
  border: "1px solid var(--border)",
  borderRadius: 10,
  background: "var(--surface-1)",
  padding: 16,
};

export function ChannelDetailPage() {
  const { chanId } = useParams<{ chanId: string }>();
  const [paused, setPaused] = useState(false);

  const { data: channel } = useQuery({
    queryKey: ["channel", chanId],
    queryFn: () => api.getChannel(chanId!),
    enabled: !!chanId,
  });

  const { data: anomalies } = useQuery({
    queryKey: ["anomalies", chanId],
    queryFn: () => api.getAnomalies(chanId!),
    enabled: !!chanId,
  });

  const { points, connected } = useChannelStream(chanId ?? "", paused);
  const latest = points[points.length - 1];

  const { data: explain } = useQuery({
    queryKey: ["explain", chanId, latest?.index],
    queryFn: () => api.explain(chanId!, latest!.index),
    enabled: !!chanId && !!latest && latest.is_anomaly,
    staleTime: 5000,
  });

  const forecast = useMemo(() => {
    if (!latest?.forecast) return null;
    return { baseIndex: latest.index, values: latest.forecast };
  }, [latest]);

  if (!chanId) return null;

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <h1 style={{ fontSize: 20, margin: 0 }}>{chanId}</h1>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <span
            style={{
              fontSize: 12,
              color: connected ? "var(--status-good)" : "var(--text-muted)",
            }}
          >
            {connected ? "● live" : "○ connecting..."}
          </span>
          <button
            onClick={() => setPaused((p) => !p)}
            style={{
              fontSize: 12,
              padding: "4px 12px",
              borderRadius: 6,
              border: "1px solid var(--border)",
              background: "var(--surface-1)",
              color: "var(--text-primary)",
              cursor: "pointer",
            }}
          >
            {paused ? "Resume" : "Pause"}
          </button>
        </div>
      </div>
      {channel && (
        <p style={{ color: "var(--text-secondary)", fontSize: 13, marginTop: 4 }}>
          {channel.spacecraft} &middot; {channel.anomaly_class} &middot; eval F1{" "}
          {channel.eval_f1.toFixed(2)} (P {channel.eval_precision.toFixed(2)} / R{" "}
          {channel.eval_recall.toFixed(2)})
        </p>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 16, marginTop: 16 }}>
        <div style={card}>
          <TelemetryChart
            points={points}
            groundTruthWindows={anomalies?.ground_truth ?? []}
            forecast={forecast}
          />
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div style={card}>
            {latest ? (
              <AnomalyScoreGauge score={latest.ensemble_score} healthState={latest.health_state} />
            ) : (
              <p style={{ color: "var(--text-secondary)" }}>Warming up...</p>
            )}
          </div>
          <ForecastPanel
            predictedStepsToAnomaly={latest?.predicted_steps_to_anomaly ?? null}
            horizon={channel?.forecast_horizon ?? 15}
          />
        </div>
      </div>

      <div style={{ ...card, marginTop: 16 }}>
        <ShapAttributionPanel explain={explain} />
      </div>
    </div>
  );
}
