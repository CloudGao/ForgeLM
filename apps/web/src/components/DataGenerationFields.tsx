import { useQuery } from "@tanstack/react-query";
import { api } from "../api";
import type { Json, ProjectPipelineOptions } from "../types";
import { ErrorMessage, Field } from "./ui";

const styles = [["textbook", "教材化"], ["plain", "通俗解释"], ["causal", "因果展开"], ["comparison", "对比"], ["distinction", "易混淆辨析"]];
const learningStyles = [["textbook", "教材"], ["facts", "事实"]];
const stylesFor = (version: string) => ["cpt-generation-v7", "cpt-generation-v8"].includes(version) ? learningStyles : ["cpt-generation-v5", "cpt-generation-v6"].includes(version) ? documentStyles : styles;
const isFactProtocol = (version: string) => ["cpt-generation-v5", "cpt-generation-v6", "cpt-generation-v7", "cpt-generation-v8"].includes(version);
const documentStyles = [["retrospective", "回顾式"], ["thematic", "主题式"]];

export function DataGenerationFields({ kind, config, options, onChange }: {
  kind: string; config: Json; options?: ProjectPipelineOptions; onChange: (value: Json) => void;
}) {
  const weights = useQuery({ queryKey: ["model-weights"], queryFn: () => api<{ items: Json[] }>("/model-weights") });
  const workers = useQuery({ queryKey: ["workers"], queryFn: () => api<Json[]>("/workers") });
  const structured = kind === "fact_translation";
  const qa = ["dense-qa-v2", "dense-qa-v3", "dense-dialogue-v1", "dense-relations-v1"].includes(config.parameters?.prompt_version);
  const dialogue = config.parameters?.prompt_version === "dense-dialogue-v1";
  const relations = config.parameters?.prompt_version === "dense-relations-v1";
  const params = { ...(structured ? { engine: "vllm", prompt_version: "cpt-generation-v8", document_mode: "whole", styles: ["facts"], proper_name_policy: "source_exact", validation_retry_limit: 0 } : {}), ...(config.parameters || {}) };
  const update = (value: Json) => onChange({ parameters: { ...params, ...value, enable_thinking: false } });
  const binding = config.input_bindings?.input_dataset_version_id;
  const recommended = structured ? "Qwen/Qwen3-4B-AWQ" : kind === "translation" ? "tencent/Hy-MT2-1.8B" : "Qwen/Qwen3.5-4B";
  const availableStyles = stylesFor(params.prompt_version);
  const selectedStyles: string[] = params.styles || availableStyles.map(([key]) => key);
  return <>
    <ErrorMessage error={weights.error || workers.error} />
    <Field label="输出数据集名称"><input value={params.name || ""} onChange={e => update({ name: e.target.value })} /></Field>
    <Field label="输入数据来源"><select value={binding ? "upstream" : "dataset"} onChange={e => {
      const bindings = { ...(config.input_bindings || {}) };
      if (e.target.value === "upstream") bindings.input_dataset_version_id = { node_id: "", output_path: "dataset_version_id" };
      else delete bindings.input_dataset_version_id;
      const next = { ...params }; delete next.input_dataset_version_id;
      onChange({ input_bindings: bindings, parameters: next });
    }}><option value="dataset">直接选择数据集</option><option value="upstream">已连线的上游节点</option></select></Field>
    {binding ? <><Field label="上游节点 ID"><input value={binding.node_id || ""} onChange={e => onChange({ input_bindings: { ...config.input_bindings, input_dataset_version_id: { ...binding, node_id: e.target.value } } })} /></Field><Field label="上游输出字段"><input value={binding.output_path} onChange={e => onChange({ input_bindings: { ...config.input_bindings, input_dataset_version_id: { ...binding, output_path: e.target.value } } })} /></Field></> :
      <Field label="输入数据集版本"><select value={params.input_dataset_version_id || ""} onChange={e => update({ input_dataset_version_id: e.target.value })}><option value="">选择项目数据版本</option>{options?.data_versions.filter(v => v.asset_type === "dataset_version").map(v => <option key={v.asset_version_id} value={v.asset_version_id}>{v.label} · {v.status}</option>)}</select></Field>}
    <Field label="本地模型权重版本" hint={`推荐 ${recommended}；请先在权重中心下载到执行 Worker。`}><select value={params.model_weight_version_id || ""} onChange={e => update({ model_weight_version_id: e.target.value })}><option value="">选择已就绪本地权重</option>{weights.data?.items.filter(w => w.state === "ready").map(w => <option key={w.id} value={w.id}>{w.asset?.repository_id || w.asset?.name} · V{w.version} · {w.resolved_commit?.slice(0, 8)}{w.asset?.repository_id === recommended ? "（推荐）" : ""}</option>)}</select></Field>
    <Field label="推理引擎"><select disabled={structured || qa} value={params.engine || "transformers"} onChange={e => update({ engine: e.target.value, ...(params.chinese_output_constraint && e.target.value !== "vllm" ? { chinese_output_constraint: false } : {}), ...(params.validation_retry_limit === 0 && e.target.value !== "vllm" ? { validation_retry_limit: 1 } : {}) })}><option value="transformers">Transformers 本地批推理</option><option value="vllm">vLLM 本地批推理</option></select></Field>
    {params.engine === "vllm" && <>
      <Field label="权重量化" hint="使用权重格式会自动识别已量化的 AWQ 等模型；FP8 对未量化权重进行在线量化。"><select value={params.quantization || "none"} onChange={e => update({ quantization: e.target.value })}><option value="none">使用权重自带格式</option><option value="fp8">在线 FP8</option></select></Field>
      <Field label="相同提示词前缀缓存" hint="同一表达方式连续生成时可复用提示词前缀；不同模型的缓存收益需按真实文章测量。"><input type="checkbox" checked={params.enable_prefix_caching ?? false} onChange={e => update({ enable_prefix_caching: e.target.checked })} /></Field>
      {!structured && isFactProtocol(params.prompt_version) && <Field label="前置纯中文约束（试验）" hint="默认关闭。禁止生成拉丁字母；自然解码也提供重试纠错反馈，不能保证译名或事实正确。标准选卡仅跳过英文词形、词源、拼写及外文同义或对应译词卡，并保留原因；外文实体和学名不会直接丢弃，具体范围由选卡策略控制。"><input type="checkbox" checked={params.chinese_output_constraint ?? false} onChange={e => update({ chinese_output_constraint: e.target.checked, ...(e.target.checked && params.proper_name_policy === "source_exact" ? { proper_name_policy: "translate" } : {}) })} /></Field>}
      {!structured && isFactProtocol(params.prompt_version) && <Field label="质量不通过时" hint="单次兜底会保存拒绝原因并继续下一张卡，避免整段重新生成；自动调优使用相同策略。"><select value={params.validation_retry_limit ?? 1} onChange={e => update({ validation_retry_limit: Number(e.target.value) })}><option value={1}>纠正后重试一次</option><option value={0}>直接跳过，不重写</option></select></Field>}
    </>}
    <Field label="输出达到 token 上限" hint="vLLM 会先用完整输出预算重试一次；仍超长时默认跳过该篇并记录来源，不把截断正文写入数据集。"><select value={params.output_length_limit_policy || "skip"} onChange={e => update({ output_length_limit_policy: e.target.value })}><option value="skip">跳过并记录（默认）</option><option value="fail">终止任务</option></select></Field>
    <Field label="执行 Worker" hint="须同时可读取数据和模型权重。"><select value={params.resource_request?.worker_selector?.worker_id || ""} onChange={e => update({ resource_request: { ...(params.resource_request || {}), gpus: 1, worker_selector: { worker_id: e.target.value || undefined } } })}><option value="">自动选择可用 Worker</option>{workers.data?.filter(w => w.capabilities?.includes(`builtin.data-generation.${params.engine || "transformers"}`)).map(w => <option key={w.id} value={w.id}>{w.name}</option>)}</select></Field>
    <p>Thinking 已强制关闭。{structured ? "翻译主体、关系、客体字段，保留条数、顺序和限定；输出独立中文三元组版本，可连接知识数据生成。来源专名允许保留，结构校验失败记录后跳过，不重复生成。" : kind === "translation" ? "英文 → 中文，仅输出译文。" : qa ? "每次规划、生成和审核都读取母节点正文；通过模型审核的候选仍需人工检查。" : "每篇或每个片段按所选方式分别生成中文正文。"}</p>
    <Field label="文章处理方式"><select disabled={qa || isFactProtocol(params.prompt_version)} value={isFactProtocol(params.prompt_version) ? "whole" : params.document_mode || "chunk"} onChange={e => update({ document_mode: e.target.value })}><option value="chunk">按字符分块</option><option value="whole">整篇处理，不切块</option></select></Field>
    <Field label="原文字符上限" hint="留空不限制；超过上限的整篇跳过，不截断。"><input type="number" min={1} value={params.max_source_characters ?? ""} onChange={e => update({ max_source_characters: e.target.value === "" ? null : Number(e.target.value) })} /></Field>
    {kind === "knowledge" && <Field label="正文表达协议"><select value={params.prompt_version || "cpt-generation-v1"} onChange={e => {
      const nextVersion = e.target.value;
      if (["dense-qa-v2", "dense-qa-v3", "dense-dialogue-v1", "dense-relations-v1"].includes(nextVersion)) {
        const bulk = ["dense-dialogue-v1", "dense-relations-v1"].includes(nextVersion);
        update({ prompt_version: nextVersion, engine: "vllm", document_mode: "whole", text_field: "dense_note", styles: ["textbook"], batch_size: bulk ? 8 : "auto", qa_limit: nextVersion === "dense-dialogue-v1" ? 76000 : nextVersion === "dense-relations-v1" ? 300000 : 100, qa_primary_start: 0, qa_primary_limit: bulk ? 13023 : 50, qa_source_mode: "single", relation_per_node_limit: 50, dialogue_single_target: nextVersion === "dense-dialogue-v1" ? 38000 : 0, dialogue_multi_target: nextVersion === "dense-dialogue-v1" ? 19000 : 0, dialogue_natural_target: nextVersion === "dense-dialogue-v1" ? 19000 : 0, proper_name_policy: "translate", chinese_output_constraint: false, fact_input_selection: "standard", validation_retry_limit: 1 });
        return;
      }
      const switchingDocumentStyles = stylesFor(nextVersion) !== stylesFor(params.prompt_version);
      update({ prompt_version: nextVersion, ...(nextVersion !== "cpt-generation-v8" && params.proper_name_policy === "source_exact" ? { proper_name_policy: "translate" } : {}), ...(isFactProtocol(nextVersion) ? { document_mode: "whole" } : { ...(params.chinese_output_constraint ? { chinese_output_constraint: false } : {}), ...(params.fact_input_selection === "chinese_ready" ? { fact_input_selection: "standard" } : {}), ...(params.validation_retry_limit === 0 ? { validation_retry_limit: 1 } : {}) }), ...(switchingDocumentStyles ? { styles: stylesFor(nextVersion).map(([key]) => key) } : {}) });
    }}><option value="dense-relations-v1">母知识骨架：证据绑定关联知识</option><option value="dense-dialogue-v1">母知识骨架：证据绑定单轮与多轮对话</option><option value="dense-qa-v3">母知识骨架：证据优先 QA 试跑</option><option value="dense-qa-v2">母知识骨架：多角度 QA 试跑</option><option value="cpt-generation-v1">标准知识正文</option><option value="cpt-generation-v2">纯陈述正文，禁止 QA</option><option value="cpt-generation-v3">纯陈述正文，逐项保留事实</option><option value="cpt-generation-v4">大幅重写，保留完整事实（试验）</option><option value="cpt-generation-v5">事实卡写作：回顾式与主题式</option><option value="cpt-generation-v6">常识事实卡：简洁中文双文体</option><option value="cpt-generation-v7">事实卡：教材与事实</option><option value="cpt-generation-v8">事实卡：简洁忠实表达（验证中）</option></select></Field>}
    {isFactProtocol(params.prompt_version) && <p>输入正文列须为事实卡 JSON 数组（主体、关系、客体，可选限定），不需要原文。模型自由组织表达；数值与名称检查不代表语义正确。</p>}
    {kind === "knowledge" && isFactProtocol(params.prompt_version) && <Field label="生成前选卡" hint="标准选择保留外文实体、学名和代码标识。保守选择额外跳过不透明标签、拉丁分类学卡、source=wikidata5m 且实体缺少中文名的整卡，以及纯外文词语关联、反义或近似词比较卡。常识和事件仍可翻译；没有来源标识时不猜测来源。此选项会减少百科覆盖，跳过原因可查。"><select value={params.fact_input_selection || "standard"} onChange={e => update({ fact_input_selection: e.target.value, ...(e.target.value === "chinese_ready" && params.proper_name_policy === "source_exact" ? { proper_name_policy: "translate" } : {}) })}><option value="standard">仅跳过明确不适用的卡</option><option value="chinese_ready">保守选择：中文实体、常识和事件</option></select></Field>}
    {kind === "knowledge" && params.prompt_version === "cpt-generation-v8" && <Field label="无可靠译名的专名" hint="正文仍须中文。允许项在生成前按来源与关系角色确定；仅保留原始名称，不豁免职业、类型、普通英文词句，也不保证事实正确。"><select value={params.proper_name_policy || "translate"} onChange={e => update({ proper_name_policy: e.target.value, ...(e.target.value === "source_exact" ? { chinese_output_constraint: false, fact_input_selection: "standard" } : {}) })}><option value="translate">全部翻译为中文</option><option value="source_exact">允许保留来源原文专名</option></select></Field>}
    {kind === "knowledge" && !qa && <fieldset><legend>重写方式</legend>{availableStyles.map(([key, label]) => <label key={key} style={{ display: "block" }}><input type="checkbox" checked={selectedStyles.includes(key)} onChange={e => update({ styles: e.target.checked ? [...selectedStyles, key] : selectedStyles.filter(s => s !== key) })} />{label}</label>)}</fieldset>}
    {kind === "knowledge" && !qa && <Field label="文体生成顺序" hint="逐条交错可让中间输出包含多种文体；来源随机化需使用已打散的输入数据。"><select value={params.style_order || "style_major"} onChange={e => update({ style_order: e.target.value })}><option value="style_major">完成一种文体后再生成下一种</option><option value="record_major">每条输入交错生成各文体</option></select></Field>}
    {qa && <>{[["qa_limit", "最多生成题数", 100, 1, 300000], ["qa_primary_start", "主节点起始序号（从0开始）", 0, 0, 100000], ["qa_primary_limit", "最多处理主节点", 50, 1, 20000]].map(([key, label, fallback, min, max]) => <Field key={String(key)} label={String(label)}><input type="number" min={Number(min)} max={Number(max)} value={params[String(key)] ?? fallback} onChange={e => update({ [String(key)]: Number(e.target.value) })} /></Field>)}</>}
    {dialogue && <>{[["dialogue_single_target", "百科单轮目标", 38000], ["dialogue_multi_target", "百科多轮目标", 19000], ["dialogue_natural_target", "知识自然聊天目标", 19000]].map(([key, label, fallback]) => <Field key={String(key)} label={String(label)}><input type="number" min={0} max={300000} value={params[String(key)] ?? fallback} onChange={e => update({ [String(key)]: Number(e.target.value) })} /></Field>)}</>}
    {relations && <Field label="每个母节点最多关联知识"><input type="number" min={1} max={50} value={params.relation_per_node_limit ?? 50} onChange={e => update({ relation_per_node_limit: Number(e.target.value) })} /></Field>}
    <Field label="批处理模式"><select value={params.batch_size === undefined || params.batch_size === "auto" ? "auto" : "manual"} onChange={e => update({ batch_size: e.target.value === "auto" ? "auto" : 1 })}><option value="auto">自动实测吞吐并选择批量</option><option value="manual">手动批量</option></select></Field>
    {params.batch_size !== undefined && params.batch_size !== "auto" && <Field label="每批条数"><input type="number" min={1} max={params.max_batch_size || 128} value={params.batch_size} onChange={e => update({ batch_size: Number(e.target.value) })} /></Field>}
    <Field label="自动探测批量上限" hint={params.engine === "vllm" ? "vLLM 自动模式在最多 64 个活动序列内实测连续并发，比较含重试的合格产出；探测使用原输出上限。" : "默认探测至 128；选择吞吐最高的安全候选。长输出 OOM 时降低批量重试。"}><input type="number" min={1} max={1024} value={params.max_batch_size ?? 128} onChange={e => update({ max_batch_size: Number(e.target.value) })} /></Field>
    <details><summary>分块与生成参数</summary>
      <Field label="正文列名"><input value={params.text_field ?? "text"} onChange={e => update({ text_field: e.target.value })} /></Field>
      {[["chunk_characters", "每块最大字符", 3000, 128, 32000], ["max_input_tokens", "最大输入 tokens", 4096, 256, 32768], ["max_new_tokens", "最大输出 tokens", 4096, 64, 32768], ["probe_max_new_tokens", "探测输出 tokens", 128, 16, 512], ["temperature", "Temperature", 0.7, 0, 2], ["top_p", "Top P", kind === "translation" ? 0.6 : 0.8, 0.01, 1], ["gpu_memory_utilization", "GPU 显存使用上限", 0.9, 0.1, 0.95]].filter(([key]) => key !== "chunk_characters" || params.document_mode !== "whole").map(([key, label, fallback, min, max]) => <Field key={key} label={String(label)}><input type="number" step={Number(max) <= 2 ? 0.01 : 1} min={Number(min)} max={Number(max)} value={params[String(key)] ?? fallback} onChange={e => update({ [String(key)]: Number(e.target.value) })} /></Field>)}
    </details>
  </>;
}
