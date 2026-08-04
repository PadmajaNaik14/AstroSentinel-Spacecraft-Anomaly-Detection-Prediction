import type {
  AlertListResponse,
  AlertOut,
  AnomaliesResponse,
  ChannelDetail,
  ChannelSummary,
  ExplainResponse,
  ForecastResponse,
  TelemetryResponse,
} from "../types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

async function request<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText}: ${await res.text()}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  listChannels: () => request<ChannelSummary[]>("/api/channels"),

  getChannel: (chanId: string) => request<ChannelDetail>(`/api/channels/${chanId}`),

  getTelemetry: (chanId: string, start?: number, end?: number) => {
    const params = new URLSearchParams();
    if (start !== undefined) params.set("start", String(start));
    if (end !== undefined) params.set("end", String(end));
    const qs = params.toString();
    return request<TelemetryResponse>(
      `/api/channels/${chanId}/telemetry${qs ? `?${qs}` : ""}`
    );
  },

  getAnomalies: (chanId: string) =>
    request<AnomaliesResponse>(`/api/channels/${chanId}/anomalies`),

  getForecast: (chanId: string, index?: number) => {
    const qs = index !== undefined ? `?index=${index}` : "";
    return request<ForecastResponse>(`/api/channels/${chanId}/forecast${qs}`);
  },

  getAlerts: (params?: {
    chan_id?: string;
    severity?: string;
    limit?: number;
    offset?: number;
  }) => {
    const usp = new URLSearchParams();
    if (params?.chan_id) usp.set("chan_id", params.chan_id);
    if (params?.severity) usp.set("severity", params.severity);
    if (params?.limit !== undefined) usp.set("limit", String(params.limit));
    if (params?.offset !== undefined) usp.set("offset", String(params.offset));
    const qs = usp.toString();
    return request<AlertListResponse>(`/api/alerts${qs ? `?${qs}` : ""}`);
  },

  acknowledgeAlert: async (id: number): Promise<AlertOut> => {
    const res = await fetch(`${API_BASE}/api/alerts/${id}/acknowledge`, { method: "POST" });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return res.json() as Promise<AlertOut>;
  },

  explain: (chanId: string, index: number) =>
    request<ExplainResponse>(`/api/channels/${chanId}/explain?index=${index}`),
};

export function wsUrl(chanId: string): string {
  const base = API_BASE.replace(/^http/, "ws");
  return `${base}/ws/replay/${chanId}`;
}
