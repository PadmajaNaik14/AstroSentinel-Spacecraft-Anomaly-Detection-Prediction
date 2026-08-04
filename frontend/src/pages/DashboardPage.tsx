import { ChannelHealthGrid } from "../components/ChannelHealthGrid";

export function DashboardPage() {
  return (
    <div>
      <h1 style={{ fontSize: 20, marginBottom: 4 }}>Spacecraft Health Dashboard</h1>
      <p style={{ color: "var(--text-secondary)", marginTop: 0, marginBottom: 20, fontSize: 13 }}>
        20 curated SMAP/MSL telemetry channels, replayed from held-out labeled test data.
      </p>
      <ChannelHealthGrid />
    </div>
  );
}
