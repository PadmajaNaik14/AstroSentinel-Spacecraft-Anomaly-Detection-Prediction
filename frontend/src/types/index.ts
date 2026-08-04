export type HealthState = "stable" | "normal" | "warning" | "critical" | "failure_risk";

export interface ChannelSummary {
  chan_id: string;
  spacecraft: string;
  anomaly_class: string;
  num_values: number;
  eval_f1: number;
}

export interface ChannelDetail extends ChannelSummary {
  window_size: number;
  forecast_horizon: number;
  ae_threshold: number;
  eval_precision: number;
  eval_recall: number;
  forecast_mae: number | null;
  trained_at: string;
}

export interface TelemetryPoint {
  index: number;
  value: number;
  reconstruction_error: number;
  ensemble_score: number;
  is_anomaly: boolean;
  health_state: HealthState;
  ground_truth_anomaly: boolean;
  forecast: number[] | null;
  predicted_steps_to_anomaly: number | null;
}

export interface TelemetryResponse {
  chan_id: string;
  points: TelemetryPoint[];
}

export interface AnomalyWindow {
  start: number;
  end: number;
  anomaly_class: string;
}

export interface AnomalyEventOut {
  id: number;
  start_index: number;
  end_index: number | null;
  peak_score: number;
  health_state: HealthState;
}

export interface AnomaliesResponse {
  chan_id: string;
  ground_truth: AnomalyWindow[];
  predicted: AnomalyEventOut[];
}

export interface ForecastResponse {
  chan_id: string;
  base_index: number;
  forecast_values: number[];
  predicted_steps_to_anomaly: number | null;
}

export interface AlertOut {
  id: number;
  chan_id: string;
  event_id: number | null;
  severity: string;
  message: string;
  acknowledged: boolean;
  created_at: string;
}

export interface AlertListResponse {
  items: AlertOut[];
  total: number;
}

export interface ExplainResponse {
  chan_id: string;
  index: number;
  feature_names: string[];
  shap_values: number[];
  feature_values: number[];
  reconstruction_error_heatmap: number[];
}
