import { ChannelHealthGrid } from "../components/ChannelHealthGrid";
import { pageSubtitle, pageTitle } from "../styles/ui";

export function DashboardPage() {
  return (
    <div>
      <h1 style={pageTitle}>Spacecraft Health Dashboard</h1>
      <p style={{ ...pageSubtitle, marginBottom: 22 }}>
        20 curated SMAP/MSL telemetry channels, replayed from held-out labeled test data.
      </p>
      <ChannelHealthGrid />
    </div>
  );
}
