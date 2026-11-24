import { ComputeNavigation } from "../components/ComputeNavigation";
import { PackageSuiteManager } from "../components/PackageSuiteManager";
import { PageHeader } from "../components/ui";

export function PluginCenter() {
  return <div className="execution-page"><PageHeader eyebrow="PLUGIN SUITES" title="插件管理" description="导入套件，统一安装与更新内部组件，保留运行版本和血缘。" /><ComputeNavigation /><PackageSuiteManager /></div>;
}
