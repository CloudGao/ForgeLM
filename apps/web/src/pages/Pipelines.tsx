import { Navigate, useLocation } from "react-router-dom";

export function Pipelines() {
  const location = useLocation();
  const search = new URLSearchParams(location.search);
  if (search.has("instance") && search.has("run")) { search.set("attempt", search.get("run")!); search.delete("run"); }
  search.delete("tab"); search.delete("definition");
  return <Navigate to={{ pathname: "../training", search: search.toString() }} replace />;
}
