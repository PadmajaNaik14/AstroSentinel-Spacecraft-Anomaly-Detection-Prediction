import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api/client";
import { statusChip } from "../styles/ui";

const SEVERITY_COLOR_VAR: Record<string, string> = {
  warning: "--status-warning",
  critical: "--status-serious",
  failure_risk: "--status-critical",
};

export function AlertsTable({ chanId }: { chanId?: string }) {
  const queryClient = useQueryClient();
  const { data, isLoading } = useQuery({
    queryKey: ["alerts", chanId],
    queryFn: () => api.getAlerts({ chan_id: chanId, limit: 100 }),
    refetchInterval: 5000,
  });

  const ack = useMutation({
    mutationFn: (id: number) => api.acknowledgeAlert(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["alerts"] }),
  });

  if (isLoading) return <p style={{ color: "var(--text-secondary)" }}>Loading alerts...</p>;
  const items = data?.items ?? [];

  if (items.length === 0) {
    return <p style={{ color: "var(--text-secondary)" }}>No alerts yet.</p>;
  }

  return (
    <table className="data-table">
      <thead>
        <tr>
          <th>Time</th>
          <th>Channel</th>
          <th>Severity</th>
          <th>Message</th>
          <th />
        </tr>
      </thead>
      <tbody>
        {items.map((a) => (
          <tr key={a.id}>
            <td style={{ color: "var(--text-secondary)", whiteSpace: "nowrap" }}>
              {new Date(a.created_at).toLocaleString()}
            </td>
            <td style={{ fontWeight: 600 }}>{a.chan_id}</td>
            <td>
              <span style={statusChip(SEVERITY_COLOR_VAR[a.severity] ?? "--text-muted")}>
                {a.severity.replace("_", " ")}
              </span>
            </td>
            <td style={{ color: "var(--text-secondary)" }}>{a.message}</td>
            <td>
              {a.acknowledged ? (
                <span style={{ color: "var(--text-muted)", fontSize: 12 }}>Acknowledged</span>
              ) : (
                <button className="btn btn-sm" onClick={() => ack.mutate(a.id)}>
                  Acknowledge
                </button>
              )}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
