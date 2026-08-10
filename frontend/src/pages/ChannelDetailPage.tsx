import { useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { useParams } from "react-router-dom";
import { api } from "../api/client";
import { AnomalyScoreGauge } from "../components/AnomalyScoreGauge";
import { ForecastPanel } from "../components/ForecastPanel";
import { PointDetailPanel } from "../components/PointDetailPanel";
import { ShapAttributionPanel } from "../components/ShapAttributionPanel";
import { TelemetryChart } from "../components/TelemetryChart";
import { useChannelStream } from "../hooks/useChannelStream";
import { card, pageTitle, sectionLabel } from "../styles/ui";
import type { TelemetryPoint } from "../types";

function MetaStat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div style={sectionLabel}>{label}</div>
      <div style={{ fontSize: 13.5, fontWeight: 600, color: "var(--text-primary)", marginTop: 2 }}>
        {value}
      </div>
    </div>
  );
}

export function ChannelDetailPage() {
  const { chanId } = useParams<{ chanId: string }>();
  const [paused, setPaused] = useState(false);
  const [selectedPoint, setSelectedPoint] = useState<TelemetryPoint | null>(null);

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

  // Whatever the user clicked drives the point-specific panels; fall back to
  // the live latest point when nothing is pinned.
  const activePoint = selectedPoint ?? latest;

  // Explain whatever the user clicked; fall back to the live latest anomaly
  // when nothing is pinned, so the SHAP panel is useful even before any click.
  const explainIndex = selectedPoint?.index ?? (latest?.is_anomaly ? latest.index : undefined);
  const explainSource = selectedPoint ? "selected point" : "live anomaly";

  const { data: explain } = useQuery({
    queryKey: ["explain", chanId, explainIndex],
    queryFn: () => api.explain(chanId!, explainIndex!),
    enabled: !!chanId && explainIndex !== undefined,
    staleTime: 5000,
  });

  const forecast = useMemo(() => {
    if (!latest?.forecast) return null;
    return { baseIndex: latest.index, values: latest.forecast };
  }, [latest]);

  if (!chanId) return null;

  return (
    <div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
          flexWrap: "wrap",
          gap: 16,
        }}
      >
        <div>
          <h1 style={pageTitle}>{chanId}</h1>
          {channel && (
            <div style={{ display: "flex", gap: 28, marginTop: 12 }}>
              <MetaStat label="Spacecraft" value={channel.spacecraft} />
              <MetaStat label="Anomaly class" value={channel.anomaly_class} />
              <MetaStat label="Eval F1" value={channel.eval_f1.toFixed(2)} />
              <MetaStat
                label="Precision / Recall"
                value={`${channel.eval_precision.toFixed(2)} / ${channel.eval_recall.toFixed(2)}`}
              />
            </div>
          )}
        </div>
        <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
          <span
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 6,
              fontSize: 12,
              fontWeight: 600,
              color: connected ? "var(--status-good)" : "var(--text-muted)",
            }}
          >
            <span
              style={{
                width: 6,
                height: 6,
                borderRadius: "50%",
                background: connected ? "var(--status-good)" : "var(--text-muted)",
              }}
            />
            {connected ? "Live" : "Connecting..."}
          </span>
          <button className="btn" onClick={() => setPaused((p) => !p)}>
            {paused ? "Resume" : "Pause"}
          </button>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 16, marginTop: 20 }}>
        <div style={card}>
          <TelemetryChart
            points={points}
            groundTruthWindows={anomalies?.ground_truth ?? []}
            forecast={forecast}
            selectedIndex={selectedPoint?.index ?? null}
            onPointClick={setSelectedPoint}
          />
          <div style={{ borderTop: "1px solid var(--gridline)", marginTop: 14, paddingTop: 14 }}>
            <PointDetailPanel point={selectedPoint} onClear={() => setSelectedPoint(null)} />
          </div>
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
            hasActivePoint={!!activePoint}
            stillWarmingUp={!!activePoint && activePoint.forecast === null}
            predictedStepsToAnomaly={activePoint?.predicted_steps_to_anomaly ?? null}
            forecastValues={activePoint?.forecast ?? null}
            currentValue={activePoint?.value ?? null}
            horizon={channel?.forecast_horizon ?? 15}
            anomalyClass={channel?.anomaly_class}
            sourceIndex={activePoint?.index}
            isSelected={!!selectedPoint}
          />
        </div>
      </div>

      <div style={{ ...card, marginTop: 16 }}>
        {explainIndex !== undefined && (
          <div style={{ ...sectionLabel, marginBottom: 10 }}>
            Explaining index {explainIndex} &middot; {explainSource}
          </div>
        )}
        <ShapAttributionPanel explain={explain} />
      </div>
    </div>
  );
}
