import { Link, Route, Routes } from "react-router-dom";
import { AlertsPage } from "./pages/AlertsPage";
import { ChannelDetailPage } from "./pages/ChannelDetailPage";
import { DashboardPage } from "./pages/DashboardPage";

export function App() {
  return (
    <div style={{ maxWidth: 1200, margin: "0 auto", padding: "20px 24px" }}>
      <nav style={{ display: "flex", gap: 16, alignItems: "center", marginBottom: 24 }}>
        <span style={{ fontWeight: 800, fontSize: 16 }}>AstroSentinel</span>
        <Link to="/" style={{ fontSize: 13, color: "var(--text-secondary)" }}>
          Dashboard
        </Link>
        <Link to="/alerts" style={{ fontSize: 13, color: "var(--text-secondary)" }}>
          Alerts
        </Link>
      </nav>
      <Routes>
        <Route path="/" element={<DashboardPage />} />
        <Route path="/channels/:chanId" element={<ChannelDetailPage />} />
        <Route path="/alerts" element={<AlertsPage />} />
      </Routes>
    </div>
  );
}
