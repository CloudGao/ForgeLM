import type { Json } from "../types";

export function TrainingPerformanceReport({ report }: { report?: Json }) {
  if (report?.schema_version !== "forgelm.training-performance.v1") return null;
  const rows = (report.candidates || []) as Json[];
  const selected = JSON.stringify(report.recommendation);
  const number = (value: unknown, digits = 0) => typeof value === "number" && Number.isFinite(value) ? value.toLocaleString("zh-CN", { maximumFractionDigits: digits }) : "—";
  return <section className="detail-section">
    <header><div><h3>训练性能比较</h3><p>从 Step {number(report.source_step)} 的同一完整状态比较；显存至少保留 {number(report.memory_headroom_mb / 1024, 1)} GiB。短程结果用于选择配置，后续仍需观察长期训练。</p></div></header>
    <div className="table-wrap"><table><thead><tr><th>微批量</th><th>激活重计算</th><th>算子组合</th><th>Token/s</th><th>峰值显存</th><th>Loss 差</th><th>梯度相对误差</th><th>结果</th></tr></thead>
      <tbody>{rows.map((row, index) => <tr key={index}>
        <td>{row.configuration.micro_batch_size}</td>
        <td>{row.configuration.activation_checkpointing ? `每 ${row.configuration.checkpoint_every_n_layers} 层` : "关闭"}</td>
        <td>{row.optimization_receipt ? row.optimization_receipt.backends.map((b: Json) => `${b.extension}: ${b.id} ${b.version}`).join(" · ") : "原实现"}</td>
        <td>{number(row.tokens_per_second)}</td><td>{typeof row.peak_reserved_mb === "number" ? `${number((row.peak_device_used_mb ?? row.peak_reserved_mb) / 1024, 2)} GiB` : "—"}</td>
        <td>{number(row.loss_absolute_difference, 6)}</td><td>{number(row.gradient_relative_l2, 6)}</td>
        <td>{JSON.stringify(row.configuration) === selected ? "已选用" : row.status === "passed" ? "通过" : row.status === "oom" ? "显存不足" : "未通过"}</td>
      </tr>)}</tbody></table></div>
  </section>;
}
