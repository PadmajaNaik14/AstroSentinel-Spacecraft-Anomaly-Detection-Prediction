import { HEALTH_META } from "../healthState";
import type { HealthState } from "../types";

export function HealthPill({ state }: { state: HealthState }) {
  const meta = HEALTH_META[state];
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 6,
        padding: "3px 10px",
        borderRadius: 999,
        fontSize: 12.5,
        fontWeight: 600,
        color: "var(--text-primary)",
        border: `1px solid var(--border)`,
        background: `color-mix(in srgb, var(${meta.colorVar}) 16%, var(--surface-1))`,
      }}
    >
      <span style={{ color: `var(${meta.colorVar})` }}>{meta.icon}</span>
      {meta.label}
    </span>
  );
}
