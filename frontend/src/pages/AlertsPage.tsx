import { AlertsTable } from "../components/AlertsTable";

export function AlertsPage() {
  return (
    <div>
      <h1 style={{ fontSize: 20, marginBottom: 12 }}>Alerts</h1>
      <div
        style={{
          border: "1px solid var(--border)",
          borderRadius: 10,
          background: "var(--surface-1)",
          padding: 16,
        }}
      >
        <AlertsTable />
      </div>
    </div>
  );
}
