import { Link, NavLink, Route, Routes } from "react-router-dom";
import { AlertsPage } from "./pages/AlertsPage";
import { ChannelDetailPage } from "./pages/ChannelDetailPage";
import { DashboardPage } from "./pages/DashboardPage";

function navLinkClass({ isActive }: { isActive: boolean }): string {
  return isActive ? "nav-link active" : "nav-link";
}

export function App() {
  return (
    <>
      <header className="app-header">
        <div className="app-header-inner">
          <Link to="/" className="brand">
            <span className="brand-mark" aria-hidden="true" />
            AstroSentinel
          </Link>
          <nav style={{ display: "flex", gap: 20 }}>
            <NavLink to="/" end className={navLinkClass}>
              Dashboard
            </NavLink>
            <NavLink to="/alerts" className={navLinkClass}>
              Alerts
            </NavLink>
          </nav>
        </div>
      </header>
      <div style={{ maxWidth: 1240, margin: "0 auto", padding: "24px" }}>
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/channels/:chanId" element={<ChannelDetailPage />} />
          <Route path="/alerts" element={<AlertsPage />} />
        </Routes>
      </div>
    </>
  );
}
