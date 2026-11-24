import { Blocks, Boxes, Cpu, Database, HardDrive, House, Menu, PackageOpen, X } from "lucide-react";
import { Suspense, useEffect, useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { RouteErrorBoundary } from "./RouteErrorBoundary";
import { RouteLoading } from "./RouteLoading";
import { ControlPlaneControl } from "./ControlPlaneControl";

const navigation = [
  ["/", "首页", House, true],
  ["/data", "数据资产", Database, false],
  ["/weights", "权重", PackageOpen, false],
  ["/projects", "项目中心", Boxes, true],
  ["/benchmarks", "评测集", Blocks, false],
  ["/storage", "存储", HardDrive, false],
  ["/execution", "计算", Cpu, false],
] as const;

export function SystemShell() {
  const location = useLocation();
  const [mobileOpen, setMobileOpen] = useState(false);

  useEffect(() => setMobileOpen(false), [location.pathname]);

  return (
    <div className="system-shell">
      <header className="system-header">
        <NavLink className="brand large system-brand" to="/">
          <span className="brand-mark"><Boxes size={22} /></span>
          <span><strong>ForgeLM</strong><small>Local lifecycle studio</small></span>
        </NavLink>
        <nav className={`system-navigation ${mobileOpen ? "system-navigation-open" : ""}`} aria-label="系统导航" id="system-navigation">
          {navigation.map(([to, label, Icon, end]) => (
            <NavLink key={to} to={to} end={end} aria-label={label} className={({ isActive }) => isActive ? "active" : ""} onClick={() => setMobileOpen(false)}>
              <Icon size={16} /><span>{label}</span>
            </NavLink>
          ))}
        </nav>
        <div className="system-header-actions">
          <ControlPlaneControl />
          <button type="button" className="icon-button system-mobile-menu" aria-label={mobileOpen ? "关闭系统导航" : "打开系统导航"} aria-expanded={mobileOpen} aria-controls="system-navigation" onClick={() => setMobileOpen((open) => !open)}>
            {mobileOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
        </div>
      </header>
      {mobileOpen && <button type="button" className="system-nav-scrim" aria-label="关闭系统导航" onClick={() => setMobileOpen(false)} />}
      <RouteErrorBoundary systemPage><Suspense fallback={<main className="system-page"><RouteLoading /></main>}><Outlet /></Suspense></RouteErrorBoundary>
    </div>
  );
}
