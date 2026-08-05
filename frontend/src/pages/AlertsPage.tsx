import { AlertsTable } from "../components/AlertsTable";
import { card, pageSubtitle, pageTitle } from "../styles/ui";

export function AlertsPage() {
  return (
    <div>
      <h1 style={pageTitle}>Alerts</h1>
      <p style={{ ...pageSubtitle, marginBottom: 22 }}>
        Generated automatically when a channel's detector flags an anomaly.
      </p>
      <div style={card}>
        <AlertsTable />
      </div>
    </div>
  );
}
