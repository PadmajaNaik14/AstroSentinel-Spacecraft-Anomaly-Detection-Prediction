import type { CSSProperties } from "react";

/** Shared style constants so every page/component draws from one definition
 * of "a card," "a page title," etc. instead of re-declaring slightly
 * different inline styles. Hover states live as CSS classes in tokens.css
 * (inline styles can't do :hover) -- this file covers the rest. */

export const card: CSSProperties = {
  border: "1px solid var(--border)",
  borderRadius: "var(--radius-lg)",
  background: "var(--surface-1)",
  padding: "var(--space-4)",
  boxShadow: "var(--shadow-sm)",
};

export const pageTitle: CSSProperties = {
  fontSize: 21,
  fontWeight: 700,
  letterSpacing: "-0.01em",
  margin: 0,
  color: "var(--text-primary)",
};

export const pageSubtitle: CSSProperties = {
  color: "var(--text-secondary)",
  fontSize: 13,
  marginTop: 4,
  marginBottom: 0,
};

export const sectionLabel: CSSProperties = {
  fontSize: 10.5,
  fontWeight: 700,
  letterSpacing: "0.05em",
  textTransform: "uppercase",
  color: "var(--text-muted)",
};

export function statusChip(colorVar: string): CSSProperties {
  return {
    display: "inline-flex",
    alignItems: "center",
    gap: 6,
    padding: "2px 10px",
    borderRadius: 999,
    fontSize: 11,
    fontWeight: 700,
    textTransform: "uppercase",
    letterSpacing: "0.03em",
    color: `var(${colorVar})`,
    background: `color-mix(in srgb, var(${colorVar}) 14%, var(--surface-1))`,
  };
}
