from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import PluginVersion

COMMAND_PLUGINS = [
    ("builtin.command.data", "data_connector"),
    ("builtin.command.tokenizer", "tokenizer"),
    ("builtin.command.training", "training_engine"),
    ("builtin.command.evaluation", "evaluation"),
    ("builtin.command.quantization", "quantization"),
    ("builtin.command.inference", "inference"),
]


def register_builtin_plugins(session: Session) -> None:
    specs = [
        {
            "plugin_id": plugin_id,
            "kind": kind,
            "entrypoint": "llm_platform_worker.adapters.command:CommandAdapter",
            "capabilities": {
                "structured_jsonl_events": True,
                "artifact_upload": True,
                "shell": False,
            },
        }
        for plugin_id, kind in COMMAND_PLUGINS
    ]
    specs.append(
        {
            "plugin_id": "builtin.plugin-manager",
            "kind": "plugin_manager",
            "entrypoint": "llm_platform_worker.plugin_runtime:PluginManagerAdapter",
            "capabilities": {
                "installable_sources": ["builtin-catalog"],
                "isolated_venv": True,
                "bundled_worker_runtime": True,
                "manual_update": True,
                "shell": False,
            },
        }
    )
    specs.append(
        {
            "plugin_id": "plugin.llm-compressor.quantization",
            "kind": "quantization",
            "entrypoint": "plugin.llm-compressor.quantization",
            "config_schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "algorithm": {"type": "string", "enum": ["RTN", "GPTQ"], "default": "RTN"},
                    "scheme": {
                        "type": "string",
                        "enum": ["W4A16", "FP8_DYNAMIC"],
                        "default": "FP8_DYNAMIC",
                    },
                    "targets": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "minItems": 1,
                        "default": ["Linear"],
                    },
                    "ignore": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "default": ["lm_head"],
                    },
                    "max_sequence_length": {
                        "type": "integer",
                        "minimum": 16,
                        "maximum": 32768,
                    },
                    "num_calibration_samples": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 2048,
                    },
                    "block_size": {"type": "integer", "minimum": 16, "maximum": 1024},
                    "dampening_frac": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "maximum": 1,
                    },
                },
                "required": ["algorithm", "scheme", "targets"],
                "allOf": [
                    {
                        "if": {"properties": {"algorithm": {"const": "RTN"}}},
                        "then": {"properties": {"scheme": {"const": "FP8_DYNAMIC"}}},
                    },
                    {
                        "if": {"properties": {"algorithm": {"const": "GPTQ"}}},
                        "then": {"properties": {"scheme": {"const": "W4A16"}}},
                    },
                ],
            },
            "capabilities": {
                "framework": "llm-compressor",
                "framework_version": "0.10.0",
                "presets": ["RTN/FP8_DYNAMIC", "GPTQ/W4A16"],
                "input_formats": ["transformers-pretrained"],
                "output_formats": ["compressed-transformers"],
                "requires_calibration": ["GPTQ"],
                "shell": False,
            },
            "runtime": {
                "python": ">=3.11",
                "isolation": "venv",
                "runner": "llm_compressor",
                "requirements": ["llmcompressor==0.10.0"],
            },
            "source": {"type": "builtin-catalog"},
            "support_level": "experimental",
        }
    )
    specs.append(
        {
            "plugin_id": "plugin.forgelm.native-exporter",
            "kind": "release_exporter",
            "entrypoint": "plugin.forgelm.native-exporter",
            "config_schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "verify_smoke": {
                        "type": "boolean",
                        "default": True,
                        "description": "对受支持的 MicroLM 格式执行一次前向烟雾检查",
                    }
                },
            },
            "capabilities": {
                "framework": "forgelm-native",
                "framework_version": "0.1.0",
                "input_formats": [
                    "microlm-hybrid-safetensors",
                    "microlm-w8a16-safetensors-v1",
                    "compressed-transformers",
                    "transformers-pretrained",
                ],
                "output_formats": ["forgelm-native-model-directory-v1"],
                "preserves_inference_format": True,
                "supports_identity_export": True,
                "shell": False,
            },
            "runtime": {
                "python": ">=3.11",
                "isolation": "worker-process",
                "runner": "forgelm_native_export",
                "requirements": [],
            },
            "source": {"type": "builtin-catalog"},
            "support_level": "experimental",
        }
    )
    specs.append(
        {
            "plugin_id": "plugin.local-directory.publisher",
            "kind": "release_publisher",
            "entrypoint": "plugin.local-directory.publisher",
            "config_schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "target_subdirectory": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 120,
                        "default": "published-models",
                        "description": "管理员发布根目录下的安全子目录",
                    }
                },
                "required": ["target_subdirectory"],
            },
            "capabilities": {
                "target_kind": "local_directory",
                "input_formats": ["forgelm-native-model-directory-v1"],
                "credential_kinds": [],
                "returns_immutable_locator": True,
                "requires_admin_release_root": True,
                "shell": False,
            },
            "runtime": {
                "python": ">=3.11",
                "isolation": "worker-process",
                "runner": "local_directory_publish",
                "requirements": [],
            },
            "source": {"type": "builtin-catalog"},
            "support_level": "experimental",
        }
    )
    specs.append(
        {
            "plugin_id": "builtin.pretokenize",
            "kind": "data_processor",
            "entrypoint": "llm_platform_worker.adapters.pretokenize:PretokenizeAdapter",
            "capabilities": {
                "format": "forgelm-packed-tokens-v1",
                "document_boundaries": True,
                "immutable_shards": True,
                "external_artifacts": True,
                "shell": False,
            },
        }
    )
    specs.append(
        {
            "plugin_id": "builtin.tokenize-corpus",
            "kind": "data_processor",
            "entrypoint": (
                "llm_platform_worker.adapters.tokenized_corpus:TokenizedCorpusAdapter"
            ),
            "capabilities": {
                "format": "forgelm-token-corpus-v2",
                "document_index": True,
                "actual_output_accounting": True,
                "immutable_shards": True,
                "external_artifacts": True,
                "shell": False,
            },
        }
    )
    specs.append(
        {
            "plugin_id": "builtin.build-training-index",
            "kind": "data_processor",
            "entrypoint": (
                "llm_platform_worker.adapters.training_data_index:TrainingDataIndexAdapter"
            ),
            "capabilities": {
                "format": "forgelm-training-index-v1",
                "deterministic_sampling": True,
                "unique_only": True,
                "repeat_epochs": True,
                "tail_policies": ["drop", "pad"],
                "external_artifacts": True,
                "shell": False,
            },
        }
    )
    specs.extend(
        [
            {
                "plugin_id": "builtin.supervision-scan",
                "kind": "data_processor",
                "entrypoint": "llm_platform_worker.adapters.supervision:SupervisionScanAdapter",
                "capabilities": {
                    "format": "forgelm-supervision-record-index-v1",
                    "kinds": ["conversation", "preference"],
                    "non_blocking_quality_reports": True,
                    "shell": False,
                },
            },
            {
                "plugin_id": "builtin.supervision-corpus",
                "kind": "data_processor",
                "entrypoint": "llm_platform_worker.adapters.supervision:SupervisionCorpusAdapter",
                "capabilities": {
                    "formats": ["forgelm-sft-corpus-v1", "forgelm-preference-corpus-v1"],
                    "assistant_loss_spans": True,
                    "preference_pairs": True,
                    "external_artifacts": True,
                    "shell": False,
                },
            },
            {
                "plugin_id": "builtin.build-supervision-index",
                "kind": "data_processor",
                "entrypoint": (
                    "llm_platform_worker.adapters.supervision:SupervisionTrainingIndexAdapter"
                ),
                "capabilities": {
                    "formats": [
                        "forgelm-sft-training-index-v1",
                        "forgelm-preference-training-index-v1",
                    ],
                    "deterministic_sampling": True,
                    "repeat_epochs": True,
                    "external_artifacts": True,
                    "shell": False,
                },
            },
        ]
    )
    for engine in ("transformers", "vllm"):
        specs.append({"plugin_id": f"builtin.data-generation.{engine}", "kind": "data_processor",
                      "entrypoint": "llm_platform_worker.adapters.data_generation:DataGenerationAdapter",
                      "capabilities": {"batch": True, "batch_autotune": True, "enable_thinking": False, "single_worker": True, "pause": False, "resume": False}})
    specs.append(
        {
            "plugin_id": "builtin.data-conversion",
            "kind": "data_processor",
            "entrypoint": (
                "llm_platform_worker.adapters.data_conversion:DataConversionAdapter"
            ),
            "capabilities": {
                "formats": ["pdf", "docx", "epub"],
                "recipe_versions": True,
                "operator_composition": True,
                "ocr": False,
                "multimodal_assets": False,
                "artifact_upload": True,
                "shell": False,
            },
        }
    )
    specs.append(
        {
            "plugin_id": "builtin.data-juicer",
            "kind": "data_processor",
            "entrypoint": "llm_platform_worker.adapters.data_juicer:DataJuicerAdapter",
            "capabilities": {
                "framework": "Data-Juicer",
                "recipe_versions": True,
                "single_worker": True,
                "distributed": False,
                "artifact_upload": True,
                "shell": False,
            },
        }
    )
    specs.append({
        "plugin_id": "builtin.storage-maintenance", "kind": "data_processor",
        "entrypoint": "llm_platform_worker.adapters.storage_maintenance:StorageMaintenanceAdapter",
        "capabilities": {"inventory": True, "exact_cache_cleanup": True,
                         "single_worker": True, "shell": False},
    })
    specs.append(
        {
            "plugin_id": "builtin.hf-tokenizer",
            "kind": "tokenizer",
            "entrypoint": "llm_platform_worker.adapters.hf_tokenizer:HfTokenizerAdapter",
            "capabilities": {
                "framework": "huggingface-tokenizers",
                "algorithms": ["BPE", "Unigram", "WordPiece"],
                "deterministic_mixture_sampling": True,
                "artifact_upload": True,
                "shell": False,
            },
        }
    )
    specs.append(
        {
            "plugin_id": "builtin.disk-bpe-tokenizer",
            "kind": "tokenizer",
            "entrypoint": (
                "llm_platform_worker.adapters.disk_bpe_tokenizer:DiskBpeTokenizerAdapter"
            ),
            "capabilities": {
                "framework": "disk-backed-huggingface-bpe",
                "algorithms": ["BPE"],
                "external_frequency_aggregation": True,
                "bounded_merge_working_set": True,
                "deterministic_mixture_sampling": True,
                "artifact_upload": True,
                "shell": False,
            },
        }
    )
    specs.append(
        {
            "plugin_id": "builtin.transformers-inference",
            "kind": "inference",
            "entrypoint": (
                "llm_platform_worker.adapters.transformers_inference:TransformersInferenceAdapter"
            ),
            "capabilities": {
                "framework": "transformers",
                "chat_template": True,
                "sampling": True,
                "streaming": False,
                "shell": False,
            },
        }
    )
    specs.append(
        {
            "plugin_id": "builtin.code-import",
            "kind": "code_connector",
            "entrypoint": "llm_platform_worker.adapters.code_import:CodeImportAdapter",
            "capabilities": {
                "sources": ["git", "local", "archive"],
                "immutable_snapshot": True,
                "artifact_upload": True,
                "shell": False,
            },
        }
    )
    specs.append(
        {
            "plugin_id": "builtin.data-import",
            "kind": "data_connector",
            "entrypoint": "llm_platform_worker.adapters.data_import:DataImportAdapter",
            "capabilities": {
                "sources": ["http", "https", "huggingface", "git", "local"],
                "source_preflight": True,
                "sample_preview": True,
                "quality_analysis": True,
                "artifact_upload": True,
                "shell": False,
            },
        }
    )
    specs.append(
        {
            "plugin_id": "builtin.model-weight",
            "kind": "model_connector",
            "entrypoint": "llm_platform_worker.adapters.model_weight:ModelWeightAdapter",
            "capabilities": {
                "sources": ["huggingface"],
                "immutable_snapshot": True,
                "worker_local_storage": True,
                "eviction": True,
                "shell": False,
            },
        }
    )
    specs.append(
        {
            "plugin_id": "builtin.dataset-viewer",
            "kind": "data_connector",
            "entrypoint": "llm_platform_worker.adapters.dataset_viewer:DatasetViewerAdapter",
            "capabilities": {
                "structured_table": True,
                "read_only_sql": True,
                "pagination": True,
                "formats": ["parquet", "json", "jsonl", "csv", "tsv"],
                "shell": False,
            },
        }
    )
    specs.append({
        "plugin_id": "builtin.evaluation-viewer",
        "kind": "data_connector",
        "entrypoint": "llm_platform_worker.adapters.evaluation_viewer:EvaluationViewerAdapter",
        "capabilities": {"structured_table": True, "read_only_sql": True, "pagination": True, "search": True, "artifact_upload": True, "shell": False},
    })
    specs.append(
        {
            "plugin_id": "builtin.dataset-labeler",
            "kind": "data_connector",
            "entrypoint": "llm_platform_worker.adapters.dataset_labeler:DatasetLabelerAdapter",
            "capabilities": {
                "stable_record_selection": True,
                "query_scope": True,
                "relational_parquet_snapshots": True,
                "single_and_multiple_groups": True,
                "artifact_upload": True,
                "shell": False,
            },
        }
    )
    specs.append(
        {
            "plugin_id": "builtin.data-mixture",
            "kind": "data_connector",
            "entrypoint": "llm_platform_worker.adapters.data_mixture:DataMixtureAdapter",
            "capabilities": {
                "stable_record_selection": True,
                "deterministic_weighted_sampling": True,
                "artifact_upload": True,
                "shell": False,
            },
        }
    )
    specs.append(
        {
            "plugin_id": "builtin.lm-eval",
            "kind": "evaluation",
            "entrypoint": "llm_platform_worker.adapters.lm_eval:LmEvalAdapter",
            "capabilities": {
                "framework": "lm-evaluation-harness",
                "standard_benchmarks": True,
                "sample_results": True,
                "shell": False,
            },
        }
    )
    specs.append(
        {
            "plugin_id": "builtin.local-process",
            "kind": "launcher",
            "entrypoint": "llm_platform_worker.launchers.local_process:LocalProcessLauncher",
            "capabilities": {"topology": ["single_process"], "safe_stop": True},
        }
    )
    for spec in specs:
        exists = session.scalar(
            select(PluginVersion).where(
                PluginVersion.plugin_id == spec["plugin_id"],
                PluginVersion.version == "0.1.0",
            )
        )
        if exists is None:
            values = {
                "config_schema": {"type": "object", "additionalProperties": True},
                "runtime": {"python": ">=3.11", "isolation": "process"},
                "source": {"type": "builtin"},
                "support_level": "stable",
                **spec,
            }
            session.add(
                PluginVersion(
                    version="0.1.0",
                    protocol_version=1,
                    **values,
                )
            )
    session.commit()
    from .plugin_suites import register_builtin_suites

    register_builtin_suites(session)
