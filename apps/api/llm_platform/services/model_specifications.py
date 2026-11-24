from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any


def _field(
    key: str,
    label: str,
    group: str,
    field_type: str,
    default: Any,
    *,
    minimum: int | float | None = None,
    maximum: int | float | None = None,
    step: int | float | None = None,
    options: list[tuple[str, str]] | None = None,
    hint: str = "",
    advanced: bool = False,
    visible_when: dict[str, Any] | None = None,
    zero_as_none: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "key": key,
        "label": label,
        "group": group,
        "type": field_type,
        "default": default,
        "hint": hint,
        "advanced": advanced,
    }
    if minimum is not None:
        result["minimum"] = minimum
    if maximum is not None:
        result["maximum"] = maximum
    if step is not None:
        result["step"] = step
    if options:
        result["options"] = [{"value": value, "label": name} for value, name in options]
    if visible_when:
        result["visible_when"] = visible_when
    if zero_as_none:
        result["zero_as_none"] = True
    return result


def _integer(
    key: str,
    label: str,
    group: str,
    default: int,
    minimum: int,
    maximum: int,
    *,
    step: int = 1,
    hint: str = "",
    advanced: bool = False,
    visible_when: dict[str, Any] | None = None,
    zero_as_none: bool = False,
) -> dict[str, Any]:
    return _field(
        key,
        label,
        group,
        "integer",
        default,
        minimum=minimum,
        maximum=maximum,
        step=step,
        hint=hint,
        advanced=advanced,
        visible_when=visible_when,
        zero_as_none=zero_as_none,
    )


def _number(
    key: str,
    label: str,
    group: str,
    default: float,
    minimum: float,
    maximum: float,
    *,
    step: float,
    hint: str = "",
    advanced: bool = False,
) -> dict[str, Any]:
    return _field(
        key,
        label,
        group,
        "number",
        default,
        minimum=minimum,
        maximum=maximum,
        step=step,
        hint=hint,
        advanced=advanced,
    )


def _boolean(
    key: str,
    label: str,
    group: str,
    default: bool,
    *,
    hint: str = "",
    advanced: bool = False,
) -> dict[str, Any]:
    return _field(
        key,
        label,
        group,
        "boolean",
        default,
        hint=hint,
        advanced=advanced,
    )


def _select(
    key: str,
    label: str,
    group: str,
    default: str,
    options: list[tuple[str, str]],
    *,
    hint: str = "",
    advanced: bool = False,
) -> dict[str, Any]:
    return _field(
        key,
        label,
        group,
        "select",
        default,
        options=options,
        hint=hint,
        advanced=advanced,
    )


ACTIVATIONS = [("silu", "SiLU / SwiGLU"), ("gelu", "GELU"), ("gelu_pytorch_tanh", "GELU Tanh")]


def _decoder_shape(
    *, vocab: int, hidden: int, intermediate: int, layers: int, heads: int, kv_heads: int
) -> list[dict[str, Any]]:
    return [
        _integer("vocab_size", "词表大小", "模型形状", vocab, 256, 1_048_576),
        _integer("hidden_size", "Hidden Size", "模型形状", hidden, 128, 65_536, step=64),
        _integer("num_hidden_layers", "Decoder 层数", "模型形状", layers, 1, 256),
        _integer(
            "intermediate_size",
            "FFN Intermediate Size",
            "FFN",
            intermediate,
            128,
            262_144,
            step=64,
        ),
        _integer("num_attention_heads", "Query Heads", "Attention", heads, 1, 1024),
        _integer(
            "num_key_value_heads",
            "KV Heads",
            "Attention",
            kv_heads,
            1,
            1024,
            hint="等于 Query Heads 为 MHA，等于 1 为 MQA，中间值为 GQA。",
        ),
    ]


def _rope_fields(
    *, context: int, theta: float, sliding_window: int | None = None
) -> list[dict[str, Any]]:
    fields = [
        _integer(
            "max_position_embeddings",
            "最大上下文长度",
            "位置编码",
            context,
            128,
            1_048_576,
            step=128,
        ),
        _select(
            "rope_type",
            "RoPE 类型",
            "位置编码",
            "default",
            [
                ("default", "标准 RoPE"),
                ("linear", "Linear Scaling"),
                ("dynamic", "Dynamic NTK"),
                ("yarn", "YaRN"),
                ("llama3", "Llama 3 Scaling"),
            ],
        ),
        _number("rope_theta", "RoPE Theta", "位置编码", theta, 1.0, 1_000_000_000.0, step=1000.0),
        _number(
            "rope_factor",
            "RoPE Scaling Factor",
            "位置编码",
            1.0,
            1.0,
            128.0,
            step=0.1,
            advanced=True,
            hint="仅非标准 RoPE 使用。",
        ),
        _integer(
            "rope_original_max_position_embeddings",
            "RoPE 原始上下文长度",
            "位置编码",
            context,
            128,
            1_048_576,
            step=128,
            advanced=True,
            visible_when={"field": "rope_type", "in": ["yarn", "llama3"]},
        ),
        _number(
            "rope_low_freq_factor",
            "Llama 3 Low Frequency Factor",
            "位置编码",
            1.0,
            0.01,
            128.0,
            step=0.1,
            advanced=True,
        ),
        _number(
            "rope_high_freq_factor",
            "Llama 3 High Frequency Factor",
            "位置编码",
            4.0,
            0.01,
            128.0,
            step=0.1,
            advanced=True,
        ),
    ]
    if sliding_window is not None:
        fields.append(
            _integer(
                "sliding_window",
                "滑动窗口",
                "Attention",
                sliding_window,
                0,
                1_048_576,
                step=128,
                zero_as_none=True,
                hint="0 表示关闭局部滑窗。",
            )
        )
    return fields


def _decoder_runtime(
    *, config_class: str, model_type: str, category: str, capability: str = "causal_lm"
) -> dict[str, Any]:
    return {
        "kind": "builtin_transformers",
        "config_class": config_class,
        "model_class": "AutoModelForCausalLM",
        "model_type": model_type,
        "task": capability,
        "runtime": "transformers",
        "capabilities": ["builtin.model-runtime.transformers"],
        "category": category,
    }


def _family(
    family_id: str,
    name: str,
    description: str,
    category: str,
    config_class: str,
    model_type: str,
    fields: list[dict[str, Any]],
    *,
    capabilities: list[str],
) -> dict[str, Any]:
    return {
        "id": family_id,
        "version": "2",
        "name": name,
        "description": description,
        "category": category,
        "source_type": "standard",
        "implementation": _decoder_runtime(
            config_class=config_class, model_type=model_type, category=category
        ),
        "capabilities": capabilities,
        "fields": fields,
        "presets": [],
        "starting_point": {
            "name": "实现默认配置",
            "description": "字段默认值来自对应 Transformers Config，实现起点不代表推荐训练规模。",
        },
    }


ARCHITECTURE_CATALOG: list[dict[str, Any]] = [
    _family(
        "llama_dense",
        "Llama",
        "RMSNorm、SwiGLU、RoPE，可配置 MHA/GQA/MQA。",
        "Dense Decoder",
        "LlamaConfig",
        "llama",
        [
            *_decoder_shape(vocab=32000, hidden=4096, intermediate=11008, layers=32, heads=32, kv_heads=32),
            _integer("head_dim", "Head Dim（0=自动）", "Attention", 0, 0, 1024, zero_as_none=True, advanced=True),
            _select("hidden_act", "FFN 激活", "FFN", "silu", ACTIVATIONS),
            *_rope_fields(context=2048, theta=10000.0),
            _number("rms_norm_eps", "RMSNorm Epsilon", "归一化", 0.000001, 1e-9, 0.1, step=1e-7),
            _boolean("attention_bias", "Attention Bias", "Attention", False, advanced=True),
            _boolean("mlp_bias", "MLP Bias", "FFN", False, advanced=True),
            _number("attention_dropout", "Attention Dropout", "正则化", 0.0, 0.0, 1.0, step=0.01, advanced=True),
            _number("initializer_range", "初始化标准差", "初始化", 0.02, 0.000001, 1.0, step=0.001, advanced=True),
            _boolean("tie_word_embeddings", "共享 Embedding / LM Head", "输出层", False),
        ],
        capabilities=["rope", "gqa", "swiglu", "rmsnorm"],
    ),
    _family(
        "mistral",
        "Mistral",
        "GQA、SwiGLU、RMSNorm，并原生支持滑动窗口 Attention。",
        "Dense Decoder",
        "MistralConfig",
        "mistral",
        [
            *_decoder_shape(vocab=32000, hidden=4096, intermediate=14336, layers=32, heads=32, kv_heads=8),
            _integer("head_dim", "Head Dim（0=自动）", "Attention", 0, 0, 1024, zero_as_none=True, advanced=True),
            _select("hidden_act", "FFN 激活", "FFN", "silu", ACTIVATIONS),
            *_rope_fields(context=131072, theta=100000000.0, sliding_window=4096),
            _number("rms_norm_eps", "RMSNorm Epsilon", "归一化", 0.000001, 1e-9, 0.1, step=1e-7),
            _number("attention_dropout", "Attention Dropout", "正则化", 0.0, 0.0, 1.0, step=0.01, advanced=True),
            _number("initializer_range", "初始化标准差", "初始化", 0.02, 0.000001, 1.0, step=0.001, advanced=True),
            _boolean("tie_word_embeddings", "共享 Embedding / LM Head", "输出层", False),
        ],
        capabilities=["rope", "gqa", "sliding_window", "swiglu", "rmsnorm"],
    ),
    _family(
        "qwen2",
        "Qwen2",
        "QKV Bias、可选滑动窗口、RoPE 与 GQA。",
        "Dense Decoder",
        "Qwen2Config",
        "qwen2",
        [
            *_decoder_shape(vocab=151936, hidden=4096, intermediate=22016, layers=32, heads=32, kv_heads=32),
            _select("hidden_act", "FFN 激活", "FFN", "silu", ACTIVATIONS),
            *_rope_fields(context=32768, theta=1000000.0, sliding_window=0),
            _boolean("use_sliding_window", "启用滑动窗口", "Attention", False),
            _integer("max_window_layers", "使用全局 Attention 的层数", "Attention", 28, 0, 256, advanced=True),
            _number("rms_norm_eps", "RMSNorm Epsilon", "归一化", 0.000001, 1e-9, 0.1, step=1e-7),
            _number("attention_dropout", "Attention Dropout", "正则化", 0.0, 0.0, 1.0, step=0.01, advanced=True),
            _number("initializer_range", "初始化标准差", "初始化", 0.02, 0.000001, 1.0, step=0.001, advanced=True),
            _boolean("tie_word_embeddings", "共享 Embedding / LM Head", "输出层", False),
        ],
        capabilities=["rope", "gqa", "qkv_bias", "sliding_window", "swiglu", "rmsnorm"],
    ),
    _family(
        "gemma2",
        "Gemma 2",
        "交替局部/全局 Attention、Logit Soft-capping、GeGLU 与共享输出权重。",
        "Dense Decoder",
        "Gemma2Config",
        "gemma2",
        [
            *_decoder_shape(vocab=256000, hidden=2304, intermediate=9216, layers=26, heads=8, kv_heads=4),
            _integer("head_dim", "Head Dim", "Attention", 256, 16, 1024),
            _select("hidden_activation", "FFN 激活", "FFN", "gelu_pytorch_tanh", ACTIVATIONS),
            *_rope_fields(context=8192, theta=10000.0, sliding_window=4096),
            _integer("query_pre_attn_scalar", "Query Pre-Attention Scalar", "Attention", 256, 1, 65536, advanced=True),
            _number("attn_logit_softcapping", "Attention Logit Soft-cap", "Attention", 50.0, 0.0, 10000.0, step=1.0, advanced=True),
            _number("final_logit_softcapping", "Final Logit Soft-cap", "输出层", 30.0, 0.0, 10000.0, step=1.0, advanced=True),
            _number("rms_norm_eps", "RMSNorm Epsilon", "归一化", 0.000001, 1e-9, 0.1, step=1e-7),
            _number("attention_dropout", "Attention Dropout", "正则化", 0.0, 0.0, 1.0, step=0.01, advanced=True),
            _number("initializer_range", "初始化标准差", "初始化", 0.02, 0.000001, 1.0, step=0.001, advanced=True),
            _boolean("tie_word_embeddings", "共享 Embedding / LM Head", "输出层", True),
        ],
        capabilities=["rope", "gqa", "sliding_window", "soft_capping", "geglu", "rmsnorm"],
    ),
    _family(
        "phi3",
        "Phi-3",
        "GQA、SwiGLU、可选滑动窗口及长上下文 RoPE 配置。",
        "Dense Decoder",
        "Phi3Config",
        "phi3",
        [
            *_decoder_shape(vocab=32064, hidden=3072, intermediate=8192, layers=32, heads=32, kv_heads=32),
            _select("hidden_act", "FFN 激活", "FFN", "silu", ACTIVATIONS),
            *_rope_fields(context=4096, theta=10000.0, sliding_window=0),
            _integer("original_max_position_embeddings", "原始训练上下文", "位置编码", 4096, 128, 1_048_576, step=128, advanced=True),
            _number("rms_norm_eps", "RMSNorm Epsilon", "归一化", 0.00001, 1e-9, 0.1, step=1e-7),
            _number("resid_pdrop", "Residual Dropout", "正则化", 0.0, 0.0, 1.0, step=0.01, advanced=True),
            _number("embd_pdrop", "Embedding Dropout", "正则化", 0.0, 0.0, 1.0, step=0.01, advanced=True),
            _number("attention_dropout", "Attention Dropout", "正则化", 0.0, 0.0, 1.0, step=0.01, advanced=True),
            _number("initializer_range", "初始化标准差", "初始化", 0.02, 0.000001, 1.0, step=0.001, advanced=True),
            _boolean("tie_word_embeddings", "共享 Embedding / LM Head", "输出层", False),
        ],
        capabilities=["rope", "gqa", "sliding_window", "swiglu", "rmsnorm"],
    ),
    _family(
        "gpt_neox",
        "GPT-NeoX",
        "LayerNorm、GELU、并行残差与 RoPE 的经典 Dense Decoder。",
        "Dense Decoder",
        "GPTNeoXConfig",
        "gpt_neox",
        [
            *_decoder_shape(vocab=50432, hidden=6144, intermediate=24576, layers=44, heads=64, kv_heads=64),
            _select("hidden_act", "FFN 激活", "FFN", "gelu", ACTIVATIONS),
            *_rope_fields(context=2048, theta=10000.0),
            _number("layer_norm_eps", "LayerNorm Epsilon", "归一化", 0.00001, 1e-9, 0.1, step=1e-7),
            _boolean("use_parallel_residual", "并行残差", "残差结构", True),
            _boolean("attention_bias", "Attention Bias", "Attention", True, advanced=True),
            _number("attention_dropout", "Attention Dropout", "正则化", 0.0, 0.0, 1.0, step=0.01, advanced=True),
            _number("hidden_dropout", "Hidden Dropout", "正则化", 0.0, 0.0, 1.0, step=0.01, advanced=True),
            _number("initializer_range", "初始化标准差", "初始化", 0.02, 0.000001, 1.0, step=0.001, advanced=True),
            _boolean("tie_word_embeddings", "共享 Embedding / LM Head", "输出层", False),
        ],
        capabilities=["rope", "mha", "gelu", "layernorm", "parallel_residual"],
    ),
    _family(
        "mixtral_moe",
        "Mixtral",
        "Top-K 稀疏专家、GQA、SwiGLU 与可选滑动窗口。",
        "Sparse MoE Decoder",
        "MixtralConfig",
        "mixtral",
        [
            *_decoder_shape(vocab=32000, hidden=4096, intermediate=14336, layers=32, heads=32, kv_heads=8),
            _integer("head_dim", "Head Dim（0=自动）", "Attention", 0, 0, 1024, zero_as_none=True, advanced=True),
            _select("hidden_act", "专家激活", "MoE", "silu", ACTIVATIONS),
            *_rope_fields(context=131072, theta=1000000.0, sliding_window=0),
            _integer("num_local_experts", "每层专家数", "MoE", 8, 2, 512),
            _integer("num_experts_per_tok", "每 Token 激活专家数", "MoE", 2, 1, 64),
            _number("router_aux_loss_coef", "Router 辅助损失系数", "MoE", 0.001, 0.0, 10.0, step=0.0001),
            _number("router_jitter_noise", "Router Jitter", "MoE", 0.0, 0.0, 10.0, step=0.001, advanced=True),
            _boolean("output_router_logits", "输出 Router Logits", "MoE", False, advanced=True),
            _number("rms_norm_eps", "RMSNorm Epsilon", "归一化", 0.00001, 1e-9, 0.1, step=1e-7),
            _number("attention_dropout", "Attention Dropout", "正则化", 0.0, 0.0, 1.0, step=0.01, advanced=True),
            _number("initializer_range", "初始化标准差", "初始化", 0.02, 0.000001, 1.0, step=0.001, advanced=True),
            _boolean("tie_word_embeddings", "共享 Embedding / LM Head", "输出层", False),
        ],
        capabilities=["rope", "gqa", "sparse_moe", "top_k_router", "swiglu", "rmsnorm"],
    ),
    _family(
        "qwen2_moe",
        "Qwen2-MoE",
        "共享专家、路由专家、可配置稀疏频率与 QKV Bias。",
        "Sparse MoE Decoder",
        "Qwen2MoeConfig",
        "qwen2_moe",
        [
            *_decoder_shape(vocab=151936, hidden=2048, intermediate=5632, layers=24, heads=16, kv_heads=16),
            _integer("moe_intermediate_size", "路由专家 Intermediate Size", "MoE", 1408, 64, 262144, step=64),
            _integer("shared_expert_intermediate_size", "共享专家 Intermediate Size", "MoE", 5632, 64, 262144, step=64),
            _integer("num_experts", "路由专家数", "MoE", 60, 1, 1024),
            _integer("num_experts_per_tok", "每 Token 激活专家数", "MoE", 4, 1, 64),
            _integer("decoder_sparse_step", "稀疏层间隔", "MoE", 1, 1, 256, hint="每 N 层使用一次 MoE。"),
            _boolean("norm_topk_prob", "归一化 Top-K 概率", "MoE", False),
            _number("router_aux_loss_coef", "Router 辅助损失系数", "MoE", 0.001, 0.0, 10.0, step=0.0001),
            _boolean("qkv_bias", "QKV Bias", "Attention", True),
            _select("hidden_act", "专家激活", "MoE", "silu", ACTIVATIONS),
            *_rope_fields(context=32768, theta=1000000.0, sliding_window=0),
            _boolean("use_sliding_window", "启用滑动窗口", "Attention", False),
            _integer("max_window_layers", "使用全局 Attention 的层数", "Attention", 28, 0, 256, advanced=True),
            _number("rms_norm_eps", "RMSNorm Epsilon", "归一化", 0.000001, 1e-9, 0.1, step=1e-7),
            _number("attention_dropout", "Attention Dropout", "正则化", 0.0, 0.0, 1.0, step=0.01, advanced=True),
            _number("initializer_range", "初始化标准差", "初始化", 0.02, 0.000001, 1.0, step=0.001, advanced=True),
            _boolean("tie_word_embeddings", "共享 Embedding / LM Head", "输出层", False),
        ],
        capabilities=["rope", "gqa", "sparse_moe", "shared_expert", "qkv_bias", "rmsnorm"],
    ),
    _family(
        "deepseek_v3",
        "DeepSeek-V3",
        "MLA、低秩 Q/KV、分组 Top-K、共享/路由专家和 MTP。",
        "MLA + Sparse MoE Decoder",
        "DeepseekV3Config",
        "deepseek_v3",
        [
            *_decoder_shape(vocab=129280, hidden=7168, intermediate=18432, layers=61, heads=128, kv_heads=128),
            _integer("q_lora_rank", "Q LoRA Rank（0=关闭）", "MLA", 1536, 0, 32768, zero_as_none=True),
            _integer("kv_lora_rank", "KV LoRA Rank", "MLA", 512, 1, 32768),
            _integer("qk_rope_head_dim", "QK RoPE Head Dim", "MLA", 64, 1, 1024),
            _integer("qk_nope_head_dim", "QK NoPE Head Dim", "MLA", 128, 1, 1024),
            _integer("v_head_dim", "Value Head Dim", "MLA", 128, 1, 1024),
            _integer("moe_intermediate_size", "路由专家 Intermediate Size", "MoE", 2048, 64, 262144, step=64),
            _integer("n_shared_experts", "共享专家数", "MoE", 1, 0, 128),
            _integer("n_routed_experts", "路由专家数", "MoE", 256, 1, 4096),
            _integer("num_experts_per_tok", "每 Token 激活专家数", "MoE", 8, 1, 128),
            _integer("first_k_dense_replace", "前置 Dense 层数", "MoE", 3, 0, 256),
            _integer("n_group", "路由专家组数", "MoE Router", 8, 1, 512),
            _integer("topk_group", "每 Token 选择专家组", "MoE Router", 4, 1, 512),
            _number("routed_scaling_factor", "路由缩放系数", "MoE Router", 2.5, 0.0001, 100.0, step=0.1),
            _boolean("norm_topk_prob", "归一化 Top-K 概率", "MoE Router", True),
            _integer("num_mtp_layers", "MTP 层数", "MTP", 1, 0, 16),
            _select("hidden_act", "FFN/专家激活", "FFN", "silu", ACTIVATIONS),
            *_rope_fields(context=4096, theta=10000.0),
            _boolean("rope_interleave", "交错 RoPE 维度", "位置编码", True, advanced=True),
            _number("rms_norm_eps", "RMSNorm Epsilon", "归一化", 0.000001, 1e-9, 0.1, step=1e-7),
            _number("attention_dropout", "Attention Dropout", "正则化", 0.0, 0.0, 1.0, step=0.01, advanced=True),
            _number("initializer_range", "初始化标准差", "初始化", 0.02, 0.000001, 1.0, step=0.001, advanced=True),
            _boolean("tie_word_embeddings", "共享 Embedding / LM Head", "输出层", False),
        ],
        capabilities=["mla", "rope", "grouped_top_k", "sparse_moe", "shared_expert", "mtp", "rmsnorm"],
    ),
    {
        "id": "custom_code",
        "version": "2",
        "name": "项目自定义模型工厂",
        "description": "绑定具有 model_factory 入口和 JSON Schema 的不可变代码版本。",
        "category": "Custom Implementation",
        "source_type": "custom_code",
        "implementation": {
            "kind": "custom_code",
            "runtime": "project_code",
            "task": "causal_lm",
            "capabilities": ["builtin.model-runtime.custom"],
            "category": "Custom Implementation",
        },
        "capabilities": ["project_defined"],
        "fields": [
            _field("architecture_name", "架构名称", "模型身份", "text", "CustomModel"),
            _integer("vocab_size", "词表大小", "模型形状", 32000, 256, 1_048_576),
            _integer("total_parameters", "声明总参数量", "参数声明", 1, 1, 100_000_000_000_000),
            _integer("active_parameters", "声明每 Token 激活参数", "参数声明", 1, 1, 100_000_000_000_000),
            _integer("trainable_parameters", "声明可训练参数", "参数声明", 1, 1, 100_000_000_000_000),
        ],
        "presets": [],
        "starting_point": {
            "name": "代码 Schema",
            "description": "选择代码版本后由其 config_schema 动态扩展字段。",
        },
    },
]


def architecture_catalog() -> list[dict[str, Any]]:
    return deepcopy(ARCHITECTURE_CATALOG)


def architecture_family(family_id: str) -> dict[str, Any] | None:
    return next((item for item in ARCHITECTURE_CATALOG if item["id"] == family_id), None)


def _issue(code: str, message: str, field: str | None = None) -> dict[str, Any]:
    return {"severity": "blocker", "code": code, "message": message, "field": field}


def _custom_schema_fields(schema: dict[str, Any]) -> list[dict[str, Any]]:
    properties = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(properties, dict):
        return []
    required = set(schema.get("required") or [])
    result: list[dict[str, Any]] = []
    reserved = {"architecture_name", "vocab_size", "total_parameters", "active_parameters", "trainable_parameters"}
    for key, raw in properties.items():
        if key in reserved or not isinstance(raw, dict):
            continue
        raw_type = raw.get("type")
        enum = raw.get("enum")
        if isinstance(enum, list) and enum:
            field_type = "select"
            options = [(str(value), str(value)) for value in enum]
        elif raw_type == "integer":
            field_type = "integer"
            options = None
        elif raw_type == "number":
            field_type = "number"
            options = None
        elif raw_type == "boolean":
            field_type = "boolean"
            options = None
        elif raw_type == "array" and (raw.get("items") or {}).get("type") == "integer":
            field_type = "integer_list"
            options = None
        elif raw_type == "object":
            field_type = "json"
            options = None
        else:
            field_type = "text"
            options = None
        default = raw.get("default")
        if default is None:
            default = [] if field_type == "integer_list" else {} if field_type == "json" else False if field_type == "boolean" else 0 if field_type in {"integer", "number"} else ""
        result.append(
            _field(
                str(key),
                str(raw.get("title") or key),
                str(raw.get("x-group") or "自定义模型参数"),
                field_type,
                default,
                minimum=raw.get("minimum"),
                maximum=raw.get("maximum"),
                step=raw.get("multipleOf"),
                options=options,
                hint=str(raw.get("description") or ("必填" if key in required else "")),
                advanced=bool(raw.get("x-advanced", False)),
            )
        )
    return result


def fields_for_family(family: dict[str, Any], code_version: dict[str, Any] | None) -> list[dict[str, Any]]:
    fields = deepcopy(family["fields"])
    if family["id"] == "custom_code" and code_version:
        fields.extend(_custom_schema_fields(code_version.get("config_schema") or {}))
    return fields


def _normalize_config(
    family: dict[str, Any], submitted: dict[str, Any], code_version: dict[str, Any] | None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    normalized: dict[str, Any] = {}
    for field in fields_for_family(family, code_version):
        key = field["key"]
        value = (submitted or {}).get(key, field.get("default"))
        field_type = field["type"]
        if field_type == "boolean":
            if not isinstance(value, bool):
                issues.append(_issue("invalid_model_field", f"{field['label']} 必须是布尔值", key))
                value = bool(field.get("default", False))
        elif field_type in {"text", "select"}:
            if not isinstance(value, str) or not value.strip():
                issues.append(_issue("invalid_model_field", f"{field['label']} 不能为空", key))
                value = str(field.get("default", ""))
            else:
                value = value.strip()
            allowed = {item["value"] for item in field.get("options", [])}
            if allowed and value not in allowed:
                issues.append(_issue("invalid_model_field", f"{field['label']} 选项无效", key))
        elif field_type in {"integer", "number"}:
            valid = isinstance(value, (int, float)) and not isinstance(value, bool)
            if not valid or (field_type == "integer" and not isinstance(value, int)):
                issues.append(_issue("invalid_model_field", f"{field['label']} 数值类型无效", key))
                value = field.get("default", 0)
            minimum = field.get("minimum")
            maximum = field.get("maximum")
            if minimum is not None and value < minimum or maximum is not None and value > maximum:
                issues.append(_issue("model_field_out_of_range", f"{field['label']} 超出允许范围", key))
        elif field_type == "integer_list":
            if not isinstance(value, list) or any(not isinstance(item, int) for item in value):
                issues.append(_issue("invalid_model_field", f"{field['label']} 必须是整数列表", key))
                value = list(field.get("default") or [])
        elif field_type == "json":
            if not isinstance(value, dict):
                issues.append(_issue("invalid_model_field", f"{field['label']} 必须是 JSON 对象", key))
                value = dict(field.get("default") or {})
        normalized[key] = value
    return normalized, issues


def _validate_architecture(family_id: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    if family_id == "custom_code":
        issues: list[dict[str, Any]] = []
        total = int(config["total_parameters"])
        active = int(config["active_parameters"])
        trainable = int(config["trainable_parameters"])
        if active > total:
            issues.append(_issue("active_parameters_exceed_total", "每 Token 激活参数不能超过总参数", "active_parameters"))
        if trainable > total:
            issues.append(_issue("trainable_parameters_exceed_total", "可训练参数不能超过总参数", "trainable_parameters"))
        return issues
    hidden = int(config["hidden_size"])
    heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config.get("head_dim") or (hidden // heads if heads else 0))
    issues: list[dict[str, Any]] = []
    if not config.get("head_dim") and hidden % heads:
        issues.append(_issue("hidden_size_not_divisible_by_heads", "未指定 Head Dim 时 Hidden Size 必须能被 Query Heads 整除", "hidden_size"))
    if heads % kv_heads:
        issues.append(_issue("heads_not_divisible_by_kv_heads", "Query Heads 必须能被 KV Heads 整除", "num_key_value_heads"))
    if head_dim <= 0:
        issues.append(_issue("invalid_head_dim", "Head Dim 必须大于 0", "head_dim"))
    if family_id == "mixtral_moe" and int(config["num_experts_per_tok"]) > int(config["num_local_experts"]):
        issues.append(_issue("top_k_exceeds_experts", "每 Token 激活专家数不能超过专家总数", "num_experts_per_tok"))
    if family_id == "qwen2_moe" and int(config["num_experts_per_tok"]) > int(config["num_experts"]):
        issues.append(_issue("top_k_exceeds_experts", "每 Token 激活专家数不能超过专家总数", "num_experts_per_tok"))
    if family_id == "deepseek_v3":
        if int(config["num_experts_per_tok"]) > int(config["n_routed_experts"]):
            issues.append(_issue("top_k_exceeds_experts", "每 Token 激活专家数不能超过路由专家总数", "num_experts_per_tok"))
        if int(config["topk_group"]) > int(config["n_group"]):
            issues.append(_issue("topk_group_exceeds_groups", "选择专家组数不能超过专家组总数", "topk_group"))
        if int(config["n_routed_experts"]) % int(config["n_group"]):
            issues.append(_issue("experts_not_divisible_by_groups", "路由专家数必须能被专家组数整除", "n_group"))
    if config.get("rope_type") != "default" and float(config.get("rope_factor") or 1.0) <= 1.0:
        issues.append(_issue("invalid_rope_factor", "启用 RoPE Scaling 时 Factor 必须大于 1", "rope_factor"))
    return issues


def _implementation_config(family_id: str, config: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(config)
    if family_id == "custom_code":
        return result
    rope_type = str(result.pop("rope_type", "default"))
    rope_theta = float(result.pop("rope_theta", 10000.0))
    rope_factor = float(result.pop("rope_factor", 1.0))
    rope_original_context = int(result.pop("rope_original_max_position_embeddings", 0))
    rope_low_freq = float(result.pop("rope_low_freq_factor", 1.0))
    rope_high_freq = float(result.pop("rope_high_freq_factor", 4.0))
    rope_parameters: dict[str, Any] = {"rope_type": rope_type, "rope_theta": rope_theta}
    if rope_type != "default":
        rope_parameters["factor"] = rope_factor
    if rope_type in {"yarn", "llama3"}:
        rope_parameters["original_max_position_embeddings"] = rope_original_context
    if rope_type == "llama3":
        rope_parameters["low_freq_factor"] = rope_low_freq
        rope_parameters["high_freq_factor"] = rope_high_freq
    result["rope_parameters"] = rope_parameters
    for field in architecture_family(family_id)["fields"]:
        if field.get("zero_as_none") and result.get(field["key"]) == 0:
            result[field["key"]] = None
    if family_id == "gpt_neox":
        result.pop("num_key_value_heads", None)
        result.pop("sliding_window", None)
    if family_id == "gemma2":
        result.pop("use_sliding_window", None)
    return result


def _dense_parameter_summary(family_id: str, config: dict[str, Any]) -> dict[str, Any]:
    vocab = int(config["vocab_size"])
    layers = int(config["num_hidden_layers"])
    hidden = int(config["hidden_size"])
    intermediate = int(config["intermediate_size"])
    heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config.get("head_dim") or hidden // heads)
    embeddings = vocab * hidden
    lm_head = 0 if config.get("tie_word_embeddings") else vocab * hidden
    attention_bias = bool(config.get("attention_bias") or family_id in {"qwen2", "qwen2_moe", "gpt_neox"})
    q_width = heads * head_dim
    kv_width = kv_heads * head_dim
    attention_per_layer = hidden * q_width + 2 * hidden * kv_width + q_width * hidden
    if attention_bias:
        attention_per_layer += q_width + 2 * kv_width
    if family_id == "gpt_neox":
        attention_per_layer += hidden
    gated = family_id != "gpt_neox"
    ffn_per_layer = (3 if gated else 2) * hidden * intermediate
    if bool(config.get("mlp_bias")) or family_id == "gpt_neox":
        ffn_per_layer += (2 * intermediate + hidden) if gated else intermediate + hidden
    norm_multiplier = 4 if family_id == "gemma2" else 2
    norm_parameters = layers * norm_multiplier * hidden + hidden
    if family_id == "gpt_neox":
        norm_parameters *= 2
    total = embeddings + lm_head + layers * (attention_per_layer + ffn_per_layer) + norm_parameters
    return {
        "total_parameters": total,
        "active_parameters_per_token": total,
        "trainable_parameters": total,
        "method": "architecture_formula_v2",
        "verified": True,
        "breakdown": {
            "embeddings": embeddings,
            "attention": layers * attention_per_layer,
            "ffn": layers * ffn_per_layer,
            "normalization": norm_parameters,
            "lm_head": lm_head,
        },
    }


def _moe_parameter_summary(family_id: str, config: dict[str, Any]) -> dict[str, Any]:
    base = _dense_parameter_summary("qwen2" if family_id == "qwen2_moe" else "mistral", config)
    layers = int(config["num_hidden_layers"])
    hidden = int(config["hidden_size"])
    dense_ffn_per_layer = 3 * hidden * int(config["intermediate_size"])
    if family_id == "mixtral_moe":
        experts = int(config["num_local_experts"])
        top_k = int(config["num_experts_per_tok"])
        expert_size = int(config["intermediate_size"])
        sparse_layers = layers
        shared_per_layer = 0
    else:
        experts = int(config["num_experts"])
        top_k = int(config["num_experts_per_tok"])
        expert_size = int(config["moe_intermediate_size"])
        sparse_layers = (layers + int(config["decoder_sparse_step"]) - 1) // int(config["decoder_sparse_step"])
        shared_per_layer = 3 * hidden * int(config["shared_expert_intermediate_size"])
    expert_per = 3 * hidden * expert_size
    router_per = hidden * experts
    dense_layers = layers - sparse_layers
    total_moe = sparse_layers * (experts * expert_per + shared_per_layer + router_per)
    active_moe = sparse_layers * (top_k * expert_per + shared_per_layer + router_per)
    dense_replaced = sparse_layers * dense_ffn_per_layer
    total = int(base["total_parameters"]) - dense_replaced + total_moe
    active = int(base["total_parameters"]) - dense_replaced + active_moe
    breakdown = dict(base["breakdown"])
    breakdown.pop("ffn", None)
    breakdown.update(
        {
            "dense_ffn": dense_layers * dense_ffn_per_layer,
            "routed_experts": sparse_layers * experts * expert_per,
            "active_routed_experts": sparse_layers * top_k * expert_per,
            "shared_experts": sparse_layers * shared_per_layer,
            "router": sparse_layers * router_per,
        }
    )
    return {
        "total_parameters": total,
        "active_parameters_per_token": active,
        "trainable_parameters": total,
        "method": "architecture_formula_v2",
        "verified": True,
        "breakdown": breakdown,
    }


def _deepseek_parameter_summary(config: dict[str, Any]) -> dict[str, Any]:
    vocab = int(config["vocab_size"])
    layers = int(config["num_hidden_layers"])
    hidden = int(config["hidden_size"])
    heads = int(config["num_attention_heads"])
    q_rank = int(config.get("q_lora_rank") or 0)
    kv_rank = int(config["kv_lora_rank"])
    qk_rope = int(config["qk_rope_head_dim"])
    qk_nope = int(config["qk_nope_head_dim"])
    v_dim = int(config["v_head_dim"])
    q_projection = hidden * heads * (qk_rope + qk_nope) if q_rank == 0 else hidden * q_rank + q_rank + q_rank * heads * (qk_rope + qk_nope)
    kv_projection = hidden * (kv_rank + qk_rope) + kv_rank + kv_rank * heads * (qk_nope + v_dim)
    output_projection = heads * v_dim * hidden
    attention = layers * (q_projection + kv_projection + output_projection)
    dense_layers = min(layers, int(config["first_k_dense_replace"]))
    moe_layers = layers - dense_layers
    dense_ffn = dense_layers * 3 * hidden * int(config["intermediate_size"])
    expert = 3 * hidden * int(config["moe_intermediate_size"])
    routed = moe_layers * int(config["n_routed_experts"]) * expert
    active_routed = moe_layers * int(config["num_experts_per_tok"]) * expert
    shared = moe_layers * int(config["n_shared_experts"]) * expert
    router = moe_layers * hidden * int(config["n_routed_experts"])
    embeddings = vocab * hidden
    lm_head = 0 if config.get("tie_word_embeddings") else vocab * hidden
    norms = layers * 2 * hidden + hidden
    mtp = int(config["num_mtp_layers"]) * (embeddings + 3 * hidden * int(config["intermediate_size"]))
    total = embeddings + lm_head + attention + dense_ffn + routed + shared + router + norms + mtp
    active = embeddings + lm_head + attention + dense_ffn + active_routed + shared + router + norms + mtp
    return {
        "total_parameters": total,
        "active_parameters_per_token": active,
        "trainable_parameters": total,
        "method": "architecture_formula_v2_estimate",
        "verified": False,
        "notes": ["MLA/MTP 统计将在 Worker 构造真实模型时再次核验。"],
        "breakdown": {
            "embeddings": embeddings,
            "attention_mla": attention,
            "dense_ffn": dense_ffn,
            "routed_experts": routed,
            "active_routed_experts": active_routed,
            "shared_experts": shared,
            "router": router,
            "normalization": norms,
            "mtp": mtp,
            "lm_head": lm_head,
        },
    }


def _parameter_summary(family_id: str, config: dict[str, Any]) -> dict[str, Any]:
    if family_id == "custom_code":
        return {
            "total_parameters": int(config["total_parameters"]),
            "active_parameters_per_token": int(config["active_parameters"]),
            "trainable_parameters": int(config["trainable_parameters"]),
            "method": "user_declared",
            "verified": False,
            "breakdown": {},
        }
    if family_id in {"mixtral_moe", "qwen2_moe"}:
        return _moe_parameter_summary(family_id, config)
    if family_id == "deepseek_v3":
        return _deepseek_parameter_summary(config)
    return _dense_parameter_summary(family_id, config)


def _initialization(value: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    init_type = str((value or {}).get("type") or "random")
    allowed = {"random", "pretrained", "model_version", "checkpoint"}
    issues: list[dict[str, Any]] = []
    if init_type not in allowed:
        issues.append(_issue("invalid_initialization", "初始化来源类型不受支持", "initialization.type"))
        init_type = "random"
    reference = str((value or {}).get("reference") or "").strip()
    if init_type != "random" and not reference:
        issues.append(_issue("missing_initialization_reference", "非随机初始化必须填写来源引用", "initialization.reference"))
    return {"type": init_type, **({"reference": reference} if reference else {})}, issues


def _python_code(family: dict[str, Any], spec_json: dict[str, Any]) -> str:
    implementation = family["implementation"]
    if family["id"] == "custom_code":
        return (
            "from llm_platform_core.model_runtime import build_frozen_model\n\n"
            "# 在训练子进程中读取 Worker 已校验并物化的模型规格与 model_factory。\n"
            "model, model_spec = build_frozen_model()\n"
        )
    config_class = implementation["config_class"]
    config_literal = json.dumps(spec_json["config"], ensure_ascii=False, indent=4)
    return (
        f"from transformers import {config_class}, AutoModelForCausalLM\n\n"
        f"config_dict = {config_literal}\n"
        f"config = {config_class}(**config_dict)\n"
        "model = AutoModelForCausalLM.from_config(config)\n\n"
        "# 在 ForgeLM 训练任务中应改用 build_frozen_model()，它会确认配置签名。\n"
    )


def resolve_model_specification(
    *,
    source_type: str,
    architecture_family_id: str,
    preset_id: str | None,
    config: dict[str, Any],
    initialization: dict[str, Any],
    code_version: dict[str, Any] | None = None,
) -> dict[str, Any]:
    family = architecture_family(architecture_family_id)
    if family is None:
        return {"status": "blocked", "issues": [_issue("unknown_architecture_family", "所选架构族不存在", "architecture_family_id")]}
    issues: list[dict[str, Any]] = []
    if preset_id:
        issues.append(_issue("model_profiles_not_supported", "近似参数规模预设已移除；请直接编辑真实架构字段", "preset_id"))
    if source_type != family["source_type"]:
        issues.append(_issue("architecture_source_mismatch", "架构来源类型与实现族不匹配", "source_type"))
    if source_type == "custom_code":
        if code_version is None:
            issues.append(_issue("missing_code_version", "自定义模型必须绑定已就绪代码版本", "code_version_id"))
        else:
            if not (code_version.get("entrypoints") or {}).get("model_factory"):
                issues.append(_issue("missing_model_factory_entrypoint", "代码版本必须声明 model_factory 入口", "code_version_id"))
            schema = code_version.get("config_schema") or {}
            if not isinstance(schema.get("properties"), dict):
                issues.append(_issue("missing_model_config_schema", "自定义模型代码版本必须提供包含 properties 的配置 Schema", "code_version_id"))
    normalized, config_issues = _normalize_config(family, config, code_version)
    issues.extend(config_issues)
    issues.extend(_validate_architecture(family["id"], normalized))
    resolved_initialization, initialization_issues = _initialization(initialization)
    issues.extend(initialization_issues)
    implementation_config = _implementation_config(family["id"], normalized)
    summary = _parameter_summary(family["id"], normalized)
    requirements = deepcopy(family["implementation"])
    requirements["contract_version"] = 2
    if code_version:
        requirements["code_version"] = code_version
        requirements["factory_entrypoint"] = (code_version.get("entrypoints") or {}).get("model_factory")
    frozen = {
        "schema_version": "forgelm.model-spec.v2",
        "source_type": source_type,
        "architecture": {
            "family_id": family["id"],
            "family_version": family["version"],
            "category": family["category"],
            "implementation": requirements,
        },
        "config": implementation_config,
        "initialization": resolved_initialization,
        "parameters": summary,
        "runtime_requirements": requirements,
    }
    signature = hashlib.sha256(
        json.dumps(frozen, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    frozen["config_signature"] = signature
    return {
        "status": "blocked" if issues else "ready",
        "issues": issues,
        "source_type": source_type,
        "architecture_family_id": family["id"],
        "architecture_version": family["version"],
        "preset_id": None,
        "user_config": normalized,
        "resolved_config": implementation_config,
        "initialization": resolved_initialization,
        "parameter_summary": summary,
        "runtime_requirements": requirements,
        "generated_json": frozen,
        "generated_code": _python_code(family, frozen),
        "config_signature": signature,
        "fields": fields_for_family(family, code_version),
    }
