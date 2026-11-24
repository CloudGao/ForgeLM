import { useQuery } from "@tanstack/react-query";
import { api } from "../api";
import { RouteLoading } from "../components/RouteLoading";
import { ErrorMessage } from "../components/ui";
import { DataCenter } from "./DataCenter";

type CatalogContext = {
  catalog_project_id: string;
  workspace_uri: string;
};

export function GlobalDataCenter() {
  const context = useQuery({
    queryKey: ["global-data-context"],
    queryFn: () => api<CatalogContext>("/catalog/context"),
  });

  return (
    <main className="system-page data-system-page">
      <ErrorMessage error={context.error} />
      {context.isPending && <RouteLoading label="正在连接全局数据工作区…" />}
      {context.data && <DataCenter projectIdOverride={context.data.catalog_project_id} globalMode workspaceUri={context.data.workspace_uri} />}
    </main>
  );
}
