import type { HealthState } from "./types";

export interface HealthMeta {
  label: string;
  icon: string;
  colorVar: string;
}

/** Stable/Normal -> good, Warning -> warning, Critical -> serious,
 * Failure Risk -> critical (see plan.md's status-palette mapping). */
export const HEALTH_META: Record<HealthState, HealthMeta> = {
  stable: { label: "Stable", icon: "●", colorVar: "--status-good" },
  normal: { label: "Normal", icon: "●", colorVar: "--status-good" },
  warning: { label: "Warning", icon: "▲", colorVar: "--status-warning" },
  critical: { label: "Critical", icon: "▲", colorVar: "--status-serious" },
  failure_risk: { label: "Failure Risk", icon: "✖", colorVar: "--status-critical" },
};
