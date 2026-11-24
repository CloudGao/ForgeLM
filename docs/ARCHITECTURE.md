# ForgeLM 技术架构

## 模块

| 路径 | 职责 |
|---|---|
| `apps/web` | React/TypeScript 界面，Vite 构建 |
| `apps/api/llm_platform` | FastAPI 控制面、领域服务、数据库模型 |
| `apps/api/alembic` | 数据库迁移 |
| `apps/worker/llm_platform_worker` | 独立 Agent、适配器、进程执行与事件回传 |
| `packages/llm_platform_core` | 协议、状态、校验与共享数据逻辑 |
| `packages/forgelm_plugin_sdk` | 套件开发、打包、任务和服务协议 |
| `examples/plugin-suites/text-tools` | 无模型依赖的插件示例 |
| `scripts` | 安装、启动、Worker 引导与既有迁移入口 |
| `tests` | 平台单元、契约、集成与进程级端到端验证 |

公开源码不包含 `model_sources`、MicroLM 套件、模型权重或私人工作流配置。

## 控制与执行

浏览器通过 REST/SSE 访问 FastAPI；SQLite 保存元数据，制品库保存受管理文件，外部制品位置记录保存 Worker 可访问的来源。Supervisor 持有控制面进程并提供单实例启动保护。

任务使用 Task/TaskAttempt、租约、期望状态与观察状态协议。Worker 注册能力、领取任务、回传事件并提交完成结果。模型代码、参数、数据和产物通过领域对象关联，而非直接操作业务数据库。

训练实例使用现有 DAG 和 Node Execution；新模型通过项目代码导入与通用命令适配器，或现有套件节点接入。不为模型另建调度器。

## 模型源码与外部运行时

主 wheel 只包含 API、Worker、core 与 SDK。模型仓库独立维护，通过现有导入入口冻结为 CodeVersion；平台不会将临近磁盘目录中的私有模型代码自动补进 Python 导入路径。

基础 Worker 的 release 标识由平台 wheel、引导脚本、监督脚本与部署选项决定，不读取模型仓库的 requirements。远程引导配置为 `base`、`data`、`inference`；WSL 基础安装不携带特定模型实现。

历史 MicroLM 格式的推理、评测及导出兼容桥仍保留，运行时必须由管理员另行提供；它们不包含模型结构、优化器或 Trainer 实现。公共目录不登记内置 MicroLM 套件及量化实现。训练性能协议中保留的引擎身份检查不是模型实现。

Python 训练启动桥保留可选 FLA 依赖探测缓存；与私有模型训练循环耦合的 SFT 自动批量融合实现不随平台发布。外部引擎自行负责其数值语义、状态恢复和性能优化。

## 套件边界

API 静态检查包清单、Schema、依赖锁、哈希与组件元数据，不在控制面导入套件代码。Worker 在独立环境安装、校验并调用固定 entry point；任务冻结包版本、输入与配置。

套件服务通过 Worker 的回环端口和本次令牌提供请求、流事件与取消。依赖隔离和进程树回收不构成安全沙箱，安装前必须信任来源。

## 部署与存储

默认监听 `127.0.0.1:8000`，开发 Web 服务监听回环地址。远程 Linux Worker 使用 SSH 部署及 Agent 反向隧道；此通道不等于开放普通管理 API 的鉴权。

默认控制面状态位于 `.llm-platform`；Worker 项目工作区、运行目录和外部数据有各自归属。新安装和测试不得复用私人生产数据库或数据目录。

SQLite 只支持当前单控制面部署模型。没有内置用户权限、强沙箱、S3 制品后端或通用分布式 Launcher。公网或不可信共享网络不能直接访问管理 API。

生产 Web 静态文件由控制面提供。源码安装须执行前端构建；Python wheel 单独安装不等于完整 Web 应用部署。
