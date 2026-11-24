import { Search } from "lucide-react";
import { ReactNode, useState } from "react";
import { formatNumber } from "../api";
import { EmptyState, StatusBadge } from "./ui";

export type EvaluationDatasetRow = {
  key: string;
  kind: "version" | "builtin" | "import";
  item: any;
  versions: any[];
  source: string;
};

const sourceLabels: Record<string, string> = {
  benchmark: "内置目录", huggingface: "Hugging Face", http: "HTTP(S)", https: "HTTP(S)",
  git: "Git", upload: "本地上传", local: "本地文件", local_file: "本地文件",
  local_directory: "本地目录", manual: "手动登记",
};

export function evaluationSource(item: any): string {
  const source = item.source_config?.source_type || item.source_type || "unknown";
  return sourceLabels[source] || (source === "unknown" ? "来源未记录" : source);
}

// Only collapse a directory item by its explicit identity, never by display name.
// Project lists keep every frozen version selectable; the global list groups history.
export function evaluationDatasetRows(catalog: any[], assets: any[], candidates: any[] = [], groupVersions = true): EvaluationDatasetRow[] {
  const groups = new Map<string, any[]>();
  for (const item of assets) {
    const key = groupVersions ? JSON.stringify([item.project_id || "global", item.name]) : item.id;
    groups.set(key, [...(groups.get(key) || []), item]);
  }
  const rows: EvaluationDatasetRow[] = [...groups.entries()].map(([key, versions]) => {
    versions.sort((a, b) => Number(b.version || 0) - Number(a.version || 0));
    return { key: `version:${key}`, kind: "version", item: versions[0], versions, source: evaluationSource(versions[0]) };
  });
  const registered = new Set(assets.map((item) => item.benchmark_config?.id).filter(Boolean));
  for (const item of catalog) if (!registered.has(item.id)) {
    rows.push({ key: `builtin:${item.id}`, kind: "builtin", item, versions: [], source: "内置目录" });
  }
  const configured = new Set(assets.map((item) => item.source_config?.dataset_version_id).filter(Boolean));
  for (const item of candidates) if (!configured.has(item.current_version?.id)) {
    rows.push({ key: `import:${item.id}`, kind: "import", item, versions: [], source: evaluationSource(item) });
  }
  return rows;
}

export function EvaluationDatasetList({ rows, loading, selected, onToggle, renderActions }: {
  rows: EvaluationDatasetRow[];
  loading?: boolean;
  selected?: string[];
  onToggle?: (key: string) => void;
  renderActions?: (row: EvaluationDatasetRow) => ReactNode;
}) {
  const [search, setSearch] = useState("");
  const [source, setSource] = useState("");
  const sources = [...new Set(rows.map((row) => row.source))].sort();
  const needle = search.trim().toLowerCase();
  const visible = rows.filter((row) => (!source || row.source === source) && [row.item, ...row.versions].some((item) => [item.name, item.description, row.source, item.source_uri, item.source_config?.source_uri, item.task_id, item.benchmark_config?.task_id, item.task_type, item.license, item.primary_metric, item.scoring_config?.metric, ...(item.dimensions || []), ...(item.languages || []), ...(item.contamination_tags || [])].join(" ").toLowerCase().includes(needle)));
  return <section className="evaluation-dataset-catalog" aria-label="统一评测集列表">
    <div className="benchmark-toolbar">
      <div className="search-box"><Search size={16} /><input aria-label="搜索评测集" placeholder="搜索名称、来源、能力或指标" value={search} onChange={(event) => setSearch(event.target.value)} /></div>
      <label className="evaluation-source-filter">来源<select aria-label="筛选评测集来源" value={source} onChange={(event) => setSource(event.target.value)}><option value="">全部来源</option>{sources.map((value) => <option key={value} value={value}>{value}</option>)}</select></label>
      <span>{visible.length} / {rows.length} 个评测集</span>
    </div>
    {visible.length ? <div className="data-table-wrap"><table className="data-table evaluation-dataset-table"><thead><tr>{onToggle && <th>选择</th>}<th>评测集</th><th>来源</th><th>版本</th><th>数据状态</th><th>评分指标</th>{renderActions && <th>操作</th>}</tr></thead><tbody>{visible.map((row) => {
      const item = row.item;
      const builtin = row.source === "内置目录";
      const importing = row.kind === "import";
      const version = importing ? item.current_version : item;
      const dataReady = version?.state === "ready" && Boolean(version?.artifact_id);
      const config = item.benchmark_config || item;
      return <tr key={row.key}>
        {onToggle && <td><input aria-label={`选择 ${item.name}${row.kind === "version" ? ` V${item.version}` : ""}`} type="checkbox" checked={selected?.includes(row.key) || false} onChange={() => onToggle(row.key)} /></td>}
        <td className="evaluation-dataset-name"><strong>{item.name || item.id}</strong><span>{item.description || item.source_uri || item.benchmark_config?.task_id || item.task_type}</span><div className="benchmark-tags">{(item.dimensions || []).map((dimension: string) => <i key={dimension}>{dimension}</i>)}</div></td>
        <td>{row.source}</td>
        <td>{row.kind === "builtin" ? "待添加" : importing ? `来源 V${version?.version || 1}` : <>V{item.version}<small>{row.versions.length > 1 ? `${row.versions.length} 个版本` : "不可变版本"}</small></>}</td>
        <td>{importing ? dataReady ? "待配置评测" : <StatusBadge status={version?.state || item.status || "pending"} /> : builtin ? "运行时获取" : item.artifact_id ? `${formatNumber(item.sample_count)} 条样本` : "未记录题目快照"}</td>
        <td className="evaluation-dataset-metric">{config.primary_metric || item.scoring_config?.metric || item.scoring_config?.primary_metric || (row.kind === "version" ? "见评分配置" : "待配置")}</td>
        {renderActions && <td><div className="benchmark-row-actions">{renderActions(row)}</div></td>}
      </tr>;
    })}</tbody></table></div> : loading ? <p className="quiet-message">正在读取评测集…</p> : <EmptyState title={rows.length ? "没有匹配的评测集" : "还没有评测集"} description={rows.length ? "调整搜索或来源筛选。" : "添加内置评测集、导入在线数据或上传题目。"} />}
  </section>;
}
