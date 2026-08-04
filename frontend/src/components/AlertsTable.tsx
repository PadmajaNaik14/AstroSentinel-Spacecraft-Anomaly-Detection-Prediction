import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api/client";

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
    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
      <thead>
        <tr style={{ textAlign: "left", color: "var(--text-muted)", fontSize: 11 }}>
          <th style={{ padding: "6px 8px" }}>Time</th>
          <th style={{ padding: "6px 8px" }}>Channel</th>
          <th style={{ padding: "6px 8px" }}>Severity</th>
          <th style={{ padding: "6px 8px" }}>Message</th>
          <th style={{ padding: "6px 8px" }} />
        </tr>
      </thead>
      <tbody>
        {items.map((a) => (
          <tr key={a.id} style={{ borderTop: "1px solid var(--gridline)" }}>
            <td style={{ padding: "6px 8px", color: "var(--text-secondary)" }}>
              {new Date(a.created_at).toLocaleString()}
            </td>
            <td style={{ padding: "6px 8px", fontWeight: 600 }}>{a.chan_id}</td>
            <td style={{ padding: "6px 8px" }}>
              <span
                style={{
                  color: `var(${SEVERITY_COLOR_VAR[a.severity] ?? "--text-secondary"})`,
                  fontWeight: 600,
                }}
              >
                {a.severity}
              </span>
            </td>
            <td style={{ padding: "6px 8px", color: "var(--text-secondary)" }}>{a.message}</td>
            <td style={{ padding: "6px 8px" }}>
              {a.acknowledged ? (
                <span style={{ color: "var(--text-muted)" }}>Acknowledged</span>
              ) : (
                <button
                  onClick={() => ack.mutate(a.id)}
                  style={{
                    fontSize: 12,
                    padding: "4px 10px",
                    borderRadius: 6,
                    border: "1px solid var(--border)",
                    background: "var(--surface-1)",
                    color: "var(--text-primary)",
                    cursor: "pointer",
                  }}
                >
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
