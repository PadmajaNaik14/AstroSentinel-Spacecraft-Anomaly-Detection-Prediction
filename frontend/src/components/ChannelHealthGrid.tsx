import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { Line, LineChart, ResponsiveContainer } from "recharts";
import { api } from "../api/client";
import { HEALTH_META } from "../healthState";
import type { ChannelSummary } from "../types";
import { HealthPill } from "./HealthPill";

function ChannelCard({ channel }: { channel: ChannelSummary }) {
  const { data } = useQuery({
    queryKey: ["telemetry-tail", channel.chan_id],
    queryFn: () => api.getTelemetry(channel.chan_id),
    staleTime: 60_000,
  });

  const points = data?.points ?? [];
  const last = points[points.length - 1];
  const sparkData = points.slice(-80).map((p) => ({ score: p.ensemble_score }));

  return (
    <Link
      to={`/channels/${channel.chan_id}`}
      style={{
        display: "block",
        border: "1px solid var(--border)",
        borderRadius: 10,
        background: "var(--surface-1)",
        padding: 16,
        textDecoration: "none",
        color: "var(--text-primary)",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span style={{ fontWeight: 700, fontSize: 15 }}>{channel.chan_id}</span>
        {last ? <HealthPill state={last.health_state} /> : null}
      </div>
      <div style={{ marginTop: 4, fontSize: 12, color: "var(--text-secondary)" }}>
        {channel.spacecraft} &middot; {channel.anomaly_class} &middot; F1{" "}
        {channel.eval_f1.toFixed(2)}
      </div>
      <div style={{ height: 40, marginTop: 8 }}>
        {sparkData.length > 1 && (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={sparkData}>
              <Line
                type="monotone"
                dataKey="score"
                stroke={
                  last ? `var(${HEALTH_META[last.health_state].colorVar})` : "var(--series-1)"
                }
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
    </Link>
  );
}

export function ChannelHealthGrid() {
  const { data: channels, isLoading, error } = useQuery({
    queryKey: ["channels"],
    queryFn: api.listChannels,
  });

  if (isLoading) return <p style={{ color: "var(--text-secondary)" }}>Loading channels...</p>;
  if (error)
    return (
      <p style={{ color: "var(--status-critical)" }}>
        Failed to load channels: {(error as Error).message}
      </p>
    );

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))",
        gap: 12,
      }}
    >
      {channels!.map((c) => (
        <ChannelCard key={c.chan_id} channel={c} />
      ))}
    </div>
  );
}
