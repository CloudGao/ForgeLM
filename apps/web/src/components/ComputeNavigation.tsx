import { NavLink } from "react-router-dom";
import "./PluginManager.css";

export function ComputeNavigation() {
  return <nav className="compute-navigation" aria-label="计算管理"><NavLink to="/execution" end className={({ isActive }) => `button secondary${isActive ? " active" : ""}`}>执行环境</NavLink><NavLink to="/execution/plugins" className={({ isActive }) => `button secondary${isActive ? " active" : ""}`}>插件管理</NavLink></nav>;
}
