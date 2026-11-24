export type PluginVersion = {
  id: string; plugin_id: string; version: string; kind: string; entrypoint: string;
  enabled?: boolean; support_level?: string; protocol_version?: number;
  config_schema?: Record<string, any>; capabilities?: Record<string, any>;
  runtime?: Record<string, any>; source?: Record<string, any>;
};
export type PluginBinding = { plugin_id: string; plugin_version_id: string; enabled: boolean; default_config: Record<string, any> };
export type PluginWorker = { id: string; name: string; status: string; capabilities?: string[] };
export type PluginInstallation = {
  id: string; worker_id: string; plugin_id: string; state: string; task_id?: string;
  worker?: PluginWorker; current_plugin_version?: PluginVersion; target_plugin_version?: PluginVersion;
  installed_versions?: Array<Record<string, any>>; last_error?: Record<string, any>;
  last_verified_at?: string; runtime?: Record<string, any>;
};

export const pluginKindLabels: Record<string, string> = {
  quantization: "量化", release_exporter: "格式适配", release_publisher: "目标交付",
  training_engine: "训练", inference: "推理", evaluation: "评测", data_connector: "数据导入",
  tokenizer: "Tokenizer", plugin_manager: "插件管理", launcher: "运行器",
  code_connector: "代码导入", model_connector: "模型导入", data_processor: "数据处理",
};
export function pluginLabel(plugin: PluginVersion) {
  const names: Record<string, string> = {
    "plugin.llm-compressor.quantization": "LLM Compressor",
    "plugin.microlm.quantization": "MicroLM W8A16",
    "plugin.forgelm.native-exporter": "ForgeLM Native Exporter",
    "plugin.local-directory.publisher": "Local Directory Publisher",
  };
  return names[plugin.plugin_id] || plugin.capabilities?.display_name || plugin.plugin_id;
}
export function isInstallable(plugin: PluginVersion) {
  return ["quantization", "release_exporter", "release_publisher"].includes(plugin.kind) && plugin.source?.type === "builtin-catalog";
}
// Match the existing install API's numeric version ordering; never select a target implicitly.
export function compareVersions(left: string, right: string) {
  const tuple = (value: string) => value.split(".").slice(0, 3).map((part) => Number.parseInt(part, 10) || 0);
  const a = tuple(left), b = tuple(right);
  return a[0] - b[0] || a[1] - b[1] || a[2] - b[2];
}
export function schemaDefaults(schema?: Record<string, any>) {
  return Object.fromEntries(Object.entries<any>(schema?.properties || {}).filter(([, value]) => value.default !== undefined).map(([key, value]) => [key, structuredClone(value.default)]));
}
export function installedVersion(installation: PluginInstallation | undefined, versionId: string) {
  return installation?.installed_versions?.some((item) => item.plugin_version_id === versionId) || false;
}
export function installBlocker(plugin: PluginVersion, worker: PluginWorker, installation?: PluginInstallation) {
  if (!isInstallable(plugin)) return "该插件不支持独立安装";
  if (plugin.enabled === false) return "目录版本已停用";
  if (worker.status !== "online") return "Worker 离线";
  if (!worker.capabilities?.includes("builtin.plugin-manager")) return "Worker 未提供插件安装能力";
  if (["installing", "updating"].includes(installation?.state || "")) return "该插件已有安装任务";
  if (installedVersion(installation, plugin.id)) return "该精确版本已安装";
  return "";
}
