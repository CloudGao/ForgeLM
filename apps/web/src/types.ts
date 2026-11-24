export type Json = Record<string, any>;

export type LifecycleNodeStatus =
  | "not_started"
  | "waiting"
  | "in_progress"
  | "paused"
  | "ready"
  | "blocked"
  | "failed"
  | "stopped";

export interface LifecycleVersionRef {
  kind: string;
  id: string;
  label: string;
  version?: number | string;
  status?: string;
  metadata?: Json;
}

export interface LifecycleNode {
  key: string;
  label: string;
  status: LifecycleNodeStatus;
  raw_status: string;
  version?: LifecycleVersionRef | null;
  versions: LifecycleVersionRef[];
  detail: string;
  updated_at?: string | null;
  progress?: number | null;
  metrics?: Json;
  target: string;
  is_current?: boolean;
}

export interface ProjectPipelineRevision {
  id: string;
  pipeline_id: string;
  revision: number;
  data_refs: Array<{
    asset_type: string;
    asset_version_id: string;
    purpose: string;
  }>;
  tokenizer_version_id?: string | null;
  code_version_id?: string | null;
  plan_revision_id?: string | null;
  training_run_id?: string | null;
  model_version_id?: string | null;
  evaluation_run_id?: string | null;
  quantization_run_id?: string | null;
  release_id?: string | null;
  status?: "draft" | "published";
  nodes?: ProjectPipelineDagNode[];
  edges?: ProjectPipelineDagEdge[];
  variables?: Json;
  viewport?: Json;
  validation_report?: PipelineValidationReport;
  content_hash?: string | null;
  published_at?: string | null;
  executable_graph?: boolean;
  mapping_snapshot: Json;
  locked: boolean;
}

export interface ProjectPipeline {
  id: string;
  project_id: string;
  name: string;
  description: string;
  objective: "pretraining" | "post_training" | "evaluation" | "release";
  archived: boolean;
  status: LifecycleNodeStatus;
  current_node: string;
  progress?: number | null;
  revision: ProjectPipelineRevision;
  nodes: LifecycleNode[];
  created_at: string;
  updated_at: string;
}

export interface ProjectPipelineOption {
  id: string;
  label: string;
  status: string;
  [key: string]: any;
}

export interface ProjectPipelineOptions {
  data_versions: Array<{
    asset_type: string;
    asset_version_id: string;
    label: string;
    status: string;
  }>;
  tokenizer_versions: ProjectPipelineOption[];
  code_versions: ProjectPipelineOption[];
  plan_revisions: ProjectPipelineOption[];
  training_runs: ProjectPipelineOption[];
  model_versions: ProjectPipelineOption[];
  evaluation_runs: ProjectPipelineOption[];
  quantization_runs: ProjectPipelineOption[];
  releases: ProjectPipelineOption[];
}

export type PipelineNodeKind =
  | "fact_translation"
  | "translation"
  | "knowledge"
  | "selection"
  | "preparation"
  | "data"
  | "transform"
  | "clean"
  | "mixture"
  | "tokenizer"
  | "training"
  | "evaluation"
  | "check"
  | "custom"
  | "branch"
  | "join"
  | "quantization"
  | "release";

export interface ProjectPipelineDagNode {
  id: string;
  type: PipelineNodeKind;
  position: { x: number; y: number };
  label: string;
  description?: string;
  config: Json;
}

export interface ProjectPipelineDagEdge {
  id: string;
  source: string;
  target: string;
  source_handle?: string | null;
  target_handle?: string | null;
  label?: string;
}

export interface PipelineValidationIssue {
  severity: "error" | "warning";
  code: string;
  message: string;
  node_id?: string;
  edge_id?: string;
}

export interface PipelineValidationReport {
  valid: boolean;
  errors: PipelineValidationIssue[];
  warnings: PipelineValidationIssue[];
  node_count: number;
  edge_count: number;
  revision_id?: string;
}

export interface PipelineNodeCatalogItem {
  native_operations?: Array<{ id: string; label: string; input_schema: Json; output_key: string }>;
  type: PipelineNodeKind;
  label: string;
  category: string;
  description: string;
  color: string;
  inputs: Array<{ id: string; type: string; label: string; multiple?: boolean }>;
  outputs: Array<{ id: string; type: string; label: string; multiple?: boolean }>;
}

export interface ProjectPipelineDefinition extends Omit<ProjectPipeline, "revision" | "nodes" | "status" | "current_node"> {
  current_revision_id?: string | null;
  draft_revision_id?: string | null;
  state: "draft" | "published" | "empty";
  definition_kind?: "native" | "legacy_compatibility";
  node_count: number;
  edge_count: number;
  current_revision?: ProjectPipelineRevision | null;
  draft_revision?: ProjectPipelineRevision | null;
  revisions: ProjectPipelineRevision[];
  instance_count: number;
  latest_instance?: ProjectPipelineInstance | null;
  latest_run?: ProjectPipelineRun | null;
}

export interface ProjectPipelineInstance {
  id: string;
  project_id: string;
  pipeline_id: string | null;
  pipeline_name: string;
  definition_revision_id: string | null;
  definition_revision: ProjectPipelineRevision | null;
  name: string;
  description: string;
  status: "draft" | "ready" | "archived";
  instance_kind?: "native" | "imported";
  execution_owner?: "pipeline_scheduler" | "domain_task";
  source_refs?: { training_run_id?: string | null; task_id?: string | null };
  can_run?: boolean;
  can_restore_execution?: boolean;
  metrics_summary?: { step?: number; tokens?: number; loss?: number | null };
  output_summary?: { checkpoints?: number; models?: number };
  nodes: ProjectPipelineDagNode[];
  edges: ProjectPipelineDagEdge[];
  variables: Json;
  viewport: Json;
  validation_report: PipelineValidationReport;
  content_hash?: string | null;
  activated_at?: string | null;
  locked: boolean;
  node_count: number;
  edge_count: number;
  run_count: number;
  latest_run?: ProjectPipelineRun | null;
  created_at: string;
  updated_at: string;
}

export interface LineageAssetRelation {
  id: string;
  relation_kind: "consumed" | "produced";
  asset_type: string;
  asset_id: string;
  role: string;
  label: string;
  version?: string | null;
  status: string;
  metadata: Json;
}

export interface LineageExecution {
  id: string;
  project_id: string;
  pipeline_run_id: string;
  node_run_id: string;
  task_id?: string | null;
  node_id: string;
  node_type: PipelineNodeKind;
  node_label: string;
  status: "succeeded" | "failed" | "cancelled" | "skipped";
  execution_owner: "pipeline_scheduler" | "domain_task" | string;
  execution_identity: Json;
  content_hash: string;
  consumed_assets: LineageAssetRelation[];
  produced_assets: LineageAssetRelation[];
  started_at?: string | null;
  ended_at?: string | null;
  created_at: string;
}

export interface PipelineLineage {
  pipeline_run_id: string;
  project_id: string;
  stored: boolean;
  execution_count: number;
  relation_count: number;
  executions: LineageExecution[];
}

export interface ProjectPipelineNodeRun {
  id: string;
  pipeline_run_id: string;
  node_id: string;
  node_type: PipelineNodeKind;
  label: string;
  status: "waiting" | "queued" | "running" | "paused" | "succeeded" | "failed" | "cancelled" | "skipped";
  config_snapshot: Json;
  input_snapshot: Json;
  output_snapshot: Json;
  task_id?: string | null;
  branch_decision?: string | null;
  error: Json;
  lineage?: LineageExecution | null;
  task?: Task | null;
  started_at?: string | null;
  ended_at?: string | null;
}

export interface TrainingLineageNode {
  id: string;
  kind: "asset" | "process";
  label: string;
  asset_type?: string;
  asset_id?: string;
  version?: string | null;
  status: string;
  depth: number;
  context_only: boolean;
  read_only: true;
  position?: { x: number; y: number };
  source: Record<string, string | null>;
  note: string;
}

export interface TrainingUpstreamLineage {
  attempt_id: string;
  nodes: TrainingLineageNode[];
  edges: { id: string; source: string; target: string; label: string; context_only: boolean }[];
  warnings: string[];
  truncated: boolean;
  max_depth: number;
  max_nodes: number;
}

export interface TrainingExecutionGraph {
  graph: { nodes: ProjectPipelineDagNode[]; edges: ProjectPipelineDagEdge[] };
  node_runs: ProjectPipelineNodeRun[];
  artifacts: TrainingLineageNode[];
  bindings: { id: string; source: string; target: string; label: string; context_only: boolean }[];
  warnings: string[];
  execution_node_ids: string[];
  center_node_ids?: string[];
  expanded_step_count: number;
  truncated: boolean;
}

export interface ProjectPipelineRun {
  id: string;
  project_id: string;
  pipeline_id: string | null;
  pipeline_name: string;
  instance_id: string;
  instance_name: string;
  instance?: ProjectPipelineInstance | null;
  revision_id: string | null;
  revision: ProjectPipelineRevision | null;
  status: "queued" | "running" | "paused" | "cancelling" | "succeeded" | "failed" | "cancelled";
  run_kind?: "native" | "imported";
  execution_owner?: "pipeline_scheduler" | "domain_task";
  source_refs?: { training_run_id?: string | null; task_id?: string | null };
  trigger_kind: string;
  input_snapshot: Json;
  context_snapshot: Json;
  graph_snapshot: {
    nodes: ProjectPipelineDagNode[];
    edges: ProjectPipelineDagEdge[];
    variables: Json;
    viewport: Json;
  };
  node_summary: { total?: number; completed?: number; progress?: number; counts?: Record<string, number> };
  error: Json;
  lineage?: PipelineLineage;
  node_runs: ProjectPipelineNodeRun[];
  created_at: string;
  started_at?: string | null;
  ended_at?: string | null;
}

export interface Project {
  id: string;
  name: string;
  target_model_name: string;
  workspace_uri: string;
  project_type: string;
  target_parameters: string;
  primary_languages: string[];
  target_capabilities: string[];
  lifecycle_stage: string;
  archived: boolean;
  current_model?: Json | null;
  recent_training_status: string;
  latest_evaluation?: Json | null;
  data_progress: { dataset_count: number; ready_versions: number; percent: number };
  created_at: string;
  updated_at: string;
}

export interface Task {
  id: string;
  project_id?: string;
  kind: string;
  title: string;
  adapter_id?: string;
  launcher_id?: string;
  observed_state: string;
  desired_state: string;
  progress: number;
  current_phase?: string;
  payload?: Json;
  result?: Json;
  error?: Json;
  created_at: string;
  started_at?: string | null;
  ended_at?: string | null;
  timing_estimate?: TaskTimingEstimate;
  attempts?: TaskAttempt[];
  events?: TaskEvent[];
}

export interface TaskTimingEstimate {
  status: "waiting" | "estimating" | "available" | "paused" | "unavailable" | "complete";
  actual_duration_seconds: number;
  estimated_total_seconds?: number | null;
  remaining_seconds?: number | null;
  estimated_completion_at?: string | null;
  sample_count: number;
}

export interface TaskAttempt {
  id: string;
  attempt_number: number;
  worker_id?: string | null;
  status: string;
  runtime_handle?: Json;
  environment_snapshot?: Json;
  resource_summary?: Json;
  runtime_metrics_artifact_id?: string | null;
  log_summary?: Json;
  started_at?: string | null;
  ended_at?: string | null;
  exit_code?: number | null;
  error?: Json;
}

export interface TaskEvent {
  id: number;
  task_id?: string;
  attempt_id?: string | null;
  sequence: number;
  event_type: string;
  payload: Json;
  created_at: string;
}

export interface TaskEventPage {
  items: TaskEvent[];
  has_more: boolean;
}

export interface TrainingRun {
  id: string;
  project_id: string;
  plan_revision_id: string;
  task_id: string;
  status: string;
  current_stage?: string | null;
  current_step: number;
  current_tokens: number;
  progress_counters?: Json;
  latest_metrics: Json;
  config_snapshot: Json;
  created_at: string;
  started_at?: string | null;
  ended_at?: string | null;
}

export interface Checkpoint {
  id: string;
  project_id: string;
  training_run_id: string;
  step: number;
  token_count: number;
  artifact_id?: string | null;
  metrics: Json;
  training_position?: Json;
  status: string;
  created_at: string;
}

export interface ModelVersion {
  id: string;
  project_id: string;
  name: string;
  version: number;
  kind: string;
  parent_model_version_id?: string | null;
  source_checkpoint_id?: string | null;
  artifact_id?: string | null;
  tokenizer_version_id?: string | null;
  chat_template_version_id?: string | null;
  objective_kind?: string | null;
  parameter_count: number;
  status: string;
  is_best: boolean;
  is_candidate: boolean;
  is_official: boolean;
  metadata_json: Json;
  inference_source?: Json;
  artifact?: Json | null;
  source_checkpoint?: Json | null;
  source_training_run?: Json | null;
  tokenizer?: Json | null;
  parent_model?: Json | null;
  child_model_count?: number;
  created_at: string;
  updated_at: string;
}

export interface ModelRegistry {
  models: ModelVersion[];
  checkpoints: Checkpoint[];
  lineage: Json;
}

export interface ModelArchitectureField {
  key: string;
  label: string;
  group: string;
  type: "integer" | "number" | "boolean" | "text" | "select" | "json" | "integer_list";
  default: any;
  minimum?: number;
  maximum?: number;
  step?: number;
  hint?: string;
  advanced?: boolean;
  visible_when?: { field: string; equals?: any; in?: any[] };
  zero_as_none?: boolean;
  options?: Array<{ value: string; label: string }>;
}

export interface ModelArchitecturePreset {
  id: string;
  name: string;
  description: string;
  config: Json;
}

export interface ModelArchitecture {
  id: string;
  version: string;
  name: string;
  description: string;
  category?: string;
  source_type: "standard" | "custom_code";
  implementation: any;
  capabilities?: string[];
  starting_point?: { name: string; description: string };
  fields: ModelArchitectureField[];
  presets: ModelArchitecturePreset[];
}

export interface ModelSpecificationRevision {
  id: string;
  specification_id: string;
  revision: number;
  source_type: "standard" | "custom_code";
  architecture_family_id: string;
  architecture_version: string;
  preset_id?: string | null;
  code_version_id?: string | null;
  user_config: Json;
  resolved_config: Json;
  initialization: Json;
  parameter_summary: Json;
  runtime_requirements: Json;
  generated_json: Json;
  generated_code: string;
  config_signature: string;
  locked: boolean;
  created_at: string;
}

export interface ModelSpecification {
  id: string;
  project_id: string;
  name: string;
  description: string;
  status: string;
  current_revision_id: string;
  current_revision: ModelSpecificationRevision;
  revisions: ModelSpecificationRevision[];
  created_at: string;
  updated_at: string;
}

export interface PlanRevision {
  id: string;
  revision: number;
  stages: Json[];
  input_model_ref: Json;
  model_specification_revision_id?: string | null;
  tokenizer_version_id?: string | null;
  validation_set_id?: string | null;
  validation_mixture_version_id?: string | null;
  code_version_id?: string | null;
  engine_plugin_id: string;
  launcher_plugin_id: string;
  resource_request: Json;
  effective_config: Json;
  locked: boolean;
  created_at: string;
}

export interface TrainingValidationSet {
  id: string;
  project_id: string;
  name: string;
  version: number;
  source_type: string;
  format: "jsonl" | "txt";
  filename: string;
  document_count: number;
  estimated_tokens: number;
  size_bytes: number;
  status: string;
  artifact_id: string;
  created_at: string;
}

export interface Worker {
  id: string;
  name: string;
  hostname: string;
  status: string;
  labels?: Json;
  capabilities?: string[];
  resources?: Json;
}

export interface StorageInventoryItem {
  id: string;
  project_id?: string | null;
  item_type: "category" | "cleanup_candidate";
  category: string;
  path: string;
  size_bytes: number;
  file_count: number;
  reclaimable: boolean;
  reason: string;
  metadata_json?: Json;
  task_id?: string | null;
  attempt_id?: string | null;
}

export interface StorageScan {
  id: string;
  status: string;
  summary: Json;
  error: Json;
  created_at: string;
  started_at?: string | null;
  ended_at?: string | null;
  items?: StorageInventoryItem[];
}

export interface StorageCleanupPlan {
  id: string;
  scan_id: string;
  status: string;
  confirmation_phrase: string;
  summary: Json;
  error: Json;
  expires_at: string;
  created_at: string;
  executed_at?: string | null;
  items: Array<StorageInventoryItem & { status: string; actual_reclaimed_bytes: number; error: Json }>;
}

export interface StorageOverview {
  cleanup_selection_supported?: boolean;
  current_scan?: StorageScan | null;
  latest_successful_scan?: StorageScan | null;
  cleanup_plans: StorageCleanupPlan[];
}

export interface CodeVersion {
  id: string;
  source_id: string;
  version: number;
  state: string;
  artifact_id?: string | null;
  git_commit?: string | null;
  source_snapshot: Json;
  entrypoints: Record<string, string>;
  config_schema: Json;
  dependency_spec: string;
  training_contract?: Json;
  artifact?: Json | null;
  created_at: string;
}

export interface CodeSource {
  id: string;
  project_id: string;
  name: string;
  source_type: string;
  source_uri: string;
  source_config: Json;
  status: string;
  current_version_id?: string | null;
  current_version?: CodeVersion | null;
  versions: CodeVersion[];
  created_at: string;
  updated_at: string;
}

export interface TrainingPlan {
  id: string;
  project_id: string;
  name: string;
  description: string;
  status: string;
  current_revision_id?: string | null;
  current_revision?: PlanRevision | null;
  revisions: PlanRevision[];
  created_at: string;
  updated_at: string;
}

export interface Dataset {
  id: string;
  project_id: string;
  name: string;
  source_type: string;
  modality?: "text" | "image" | "audio" | "video" | "multimodal" | "other";
  source_uri?: string;
  source_config?: Json;
  license: string;
  language: string;
  tags: string[];
  status: string;
  current_version?: Json;
  versions: Json[];
}

export interface DatasetSample {
  sample_key: string;
  source_file: string;
  raw_text: string;
  cleaned_text?: string;
  language?: string;
  length?: number;
  flags?: string[];
  rejection_reason?: string | null;
}

export interface DatasetViewerColumn {
  name: string;
  type: string;
}

export interface DatasetViewerResult {
  base_sql?: string;
  search?: string;
  source_task_id?: string;
  snapshot_metadata?: { task_id: string; framework_version: string; sha256: string };
  engine: string;
  sql: string;
  configuration: string;
  configurations: string[];
  split: string;
  splits: string[];
  format: string;
  files: string[];
  columns: DatasetViewerColumn[];
  rows: any[][];
  row_identities: Array<{ record_id: string; record_hash: string }>;
  labelable: boolean;
  labelability_reason: string;
  offset: number;
  limit: number;
  total_rows: number;
  has_more: boolean;
  truncated_cells: number;
  result_limited: boolean;
  elapsed_ms: number;
}

export interface LabelDefinition {
  id: string;
  group_id: string;
  code: string;
  name: string;
  description: string;
  color: string;
  sort_order: number;
}

export interface LabelGroup {
  id: string;
  scheme_version_id: string;
  code: string;
  name: string;
  description: string;
  selection_mode: "single" | "multiple";
  required: boolean;
  sort_order: number;
  labels: LabelDefinition[];
}

export interface LabelSchemeVersion {
  id: string;
  scheme_id: string;
  version: number;
  description: string;
  created_by: string;
  immutable: boolean;
  groups: LabelGroup[];
}

export interface LabelScheme {
  id: string;
  code: string;
  name: string;
  modality: string;
  description: string;
  status: "active" | "archived";
  current_version_id: string;
  current_version?: LabelSchemeVersion;
  versions?: LabelSchemeVersion[];
  version_count: number;
}

export interface DatasetPreflight {
  source_type: string;
  source_uri: string;
  accessible: boolean;
  files: Array<{ path: string; size_bytes?: number | null }>;
  file_count: number;
  total_size_bytes?: number | null;
  disk_free_bytes: number;
  disk_space_sufficient: boolean;
  warnings: string[];
  error?: string | null;
}

export interface Mixture {
  id: string;
  name: string;
  description: string;
  current_version?: Json;
  versions: Json[];
}

export interface ChatTemplateVersion {
  id: string;
  template_id: string;
  version: number;
  template_body: string;
  role_schema: Json;
  required_special_tokens: string[];
  assistant_mask_mode: string;
  renderer_version: string;
  validation_report: Json;
  content_hash: string;
  immutable: boolean;
  created_at: string;
}

export interface ChatTemplate {
  id: string;
  project_id: string;
  name: string;
  description: string;
  current_version_id: string;
  versions: ChatTemplateVersion[];
}

export interface SupervisionDatasetVersion {
  id: string;
  version: number;
  dataset_version_id: string;
  dataset_name?: string;
  dataset_version?: number;
  kind: "conversation" | "preference";
  mapping_config: Json;
  filter_config: Json;
  status: string;
  accepted_record_count: number;
  rejected_record_count: number;
  validation_report: Json;
  sample_preview: Json[];
  task_id?: string | null;
  identity_hash: string;
}

export interface SupervisionDataset {
  id: string;
  project_id: string;
  name: string;
  description: string;
  current_version_id: string;
  versions: SupervisionDatasetVersion[];
}

export interface SupervisionMixtureEntry {
  supervision_dataset_version_id: string;
  weight: number;
  normalized_weight?: number;
  identity_hash?: string;
  accepted_record_count?: number;
  supervision_name?: string | null;
  supervision_version?: number | null;
  dataset_name?: string | null;
  dataset_version?: number | null;
}

export interface SupervisionMixtureVersion {
  id: string;
  version: number;
  kind: "conversation" | "preference";
  entries: SupervisionMixtureEntry[];
  seed: number;
  status: string;
  identity_hash: string;
  created_at?: string;
}

export interface SupervisionMixture {
  id: string;
  project_id: string;
  name: string;
  description: string;
  current_version_id: string;
  versions: SupervisionMixtureVersion[];
  created_at?: string;
  updated_at?: string;
}

export interface DataProcessingRecipe {
  id: string;
  project_id: string;
  name: string;
  description: string;
  current_version_id?: string | null;
  current_version?: Json | null;
  versions: Json[];
  created_at: string;
  updated_at: string;
}

export interface DataConversionRecipe {
  id: string;
  project_id: string;
  name: string;
  description: string;
  current_version_id?: string | null;
  current_version?: Json | null;
  versions: Json[];
  created_at: string;
  updated_at: string;
}

export interface DataConversionRun {
  id: string;
  project_id: string;
  recipe_version_id: string;
  input_dataset_version_id: string;
  output_dataset_version_id?: string | null;
  status: string;
  dataset_name?: string | null;
  recipe_name?: string | null;
  recipe_version?: Json | null;
  input_version?: Json | null;
  output_version?: Json | null;
  task?: Task | null;
  summary: Json;
  engine_version?: string | null;
  failure_reason?: string | null;
  created_at: string;
  updated_at: string;
}

export interface DataConversionOperatorCatalog {
  engine_adapter_id: string;
  engine_version: string;
  operators: Array<{
    name: string;
    label: string;
    category: string;
    description: string;
    parameters: Record<string, Json>;
  }>;
}

export interface DataProcessingRun {
  id: string;
  project_id: string;
  recipe_version_id: string;
  input_dataset_version_id: string;
  output_dataset_version_id?: string | null;
  output_tier: "silver" | "gold";
  status: string;
  dataset_name?: string | null;
  recipe_name?: string | null;
  recipe_version?: Json | null;
  input_version?: Json | null;
  output_version?: Json | null;
  task?: Task | null;
  summary: Json;
  engine_version?: string | null;
  failure_reason?: string | null;
  created_at: string;
  updated_at: string;
}
