import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { Line, LineChart, ResponsiveContainer } from "recharts";
import { api } from "../api/client";
import { HEALTH_META } from "../healthState";
import type { ChannelSummary } from "../types";
import { HealthPill } from "./HealthPill";

function ChannelCard({ channel }: { channel: ChannelSummary }) {
  const { data, isLoading } = useQuery({
    queryKey: ["telemetry-tail", channel.chan_id],
    queryFn: () => api.getTelemetry(channel.chan_id),
    staleTime: 60_000,
  });

  const points = data?.points ?? [];
  const last = points[points.length - 1];
  const sparkData = points.slice(-80).map((p) => ({ score: p.ensemble_score }));

  return (
    <Link to={`/channels/${channel.chan_id}`} className="card-link">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span style={{ fontWeight: 700, fontSize: 15, letterSpacing: "-0.01em" }}>
          {channel.chan_id}
        </span>
        {last ? (
          <HealthPill state={last.health_state} />
        ) : (
          <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
            {isLoading ? "loading..." : ""}
          </span>
        )}
      </div>
      <div
        style={{
          marginTop: 6,
          fontSize: 12,
          color: "var(--text-secondary)",
          display: "flex",
          alignItems: "center",
        }}
      >
        <span>
          {channel.spacecraft} &middot; {channel.anomaly_class}
        </span>
        <span style={{ marginLeft: "auto", fontSize: 11, color: "var(--text-muted)", fontWeight: 700 }}>
          F1 {channel.eval_f1.toFixed(2)}
        </span>
      </div>
      <div style={{ height: 40, marginTop: 10 }}>
        {sparkData.length > 1 ? (
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
        ) : (
          <div
            style={{
              height: 2,
              marginTop: 19,
              background: "var(--gridline)",
              borderRadius: 2,
            }}
          />
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
        gridTemplateColumns: "repeat(auto-fill, minmax(224px, 1fr))",
        gap: 16,
      }}
    >
      {channels!.map((c) => (
        <ChannelCard key={c.chan_id} channel={c} />
      ))}
    </div>
  );
}
