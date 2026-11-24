import type { Json } from "../types";

export function TrainingOptimizationEditor({ value, onChange, disabled = false }: { value?: Json; onChange: (value: Json | undefined) => void; disabled?: boolean }) {
  return <fieldset disabled={disabled} className="stage-section"><legend>训练算子</legend>
    <label className="check-field"><input type="checkbox" checked={value !== undefined} onChange={(e) => onChange(e.target.checked ? { partial_rope: "reference", linear_cross_entropy: "fla" } : undefined)} /><span>指定算子组合</span></label>
    <p>需要支持算子协议的 MicroLM 引擎。运行前校验实际硬件和数值；配置随本次训练冻结。更换实现请从 Checkpoint 创建后续训练。</p>
    {value && <div className="form-grid">
      <label>Partial RoPE<select aria-label="Partial RoPE" value={String(value.partial_rope || "reference")} onChange={(e) => onChange({ ...value, partial_rope: e.target.value })}><option value="reference">模型原实现</option><option value="flash_attn">FlashAttention Rotary（CUDA）</option></select></label>
      <label>交叉熵<select aria-label="交叉熵" value={String(value.linear_cross_entropy || "fla")} onChange={(e) => onChange({ ...value, linear_cross_entropy: e.target.value })}><option value="fla">FLA 融合 Linear CE（CUDA）</option><option value="torch_chunked">PyTorch 精确分块 CE</option></select></label>
    </div>}
  </fieldset>;
}
