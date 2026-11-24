# 平台源码发布检查

检查日期：2026-11-24。此记录区分迁移前已完成验证与新仓库的持续集成状态。最新结果以 ForgeLM 的 Actions 页面为准。

## 发布边界

包含控制面、Web、Worker、core、SDK、数据库迁移、通用脚本、text-tools 示例及平台测试。采用 Apache-2.0，第三方依赖保持各自条款。

当前发布树不包含模型结构或训练实现源码、MicroLM 套件、模型权重、训练数据、Checkpoint、生成数据、私人工作流、实验报告或运行数据库。旧模型格式兼容桥只调用外部运行时，不携带模型实现。

ForgeLM 使用独立初始提交，仅导入审查后的平台文件，不导入旧混合开发仓库的历史、分支或本地检查点。原仓库保留作存档。本地忽略的模型与运行数据不属于发布内容。

## 已执行验证

| 检查 | 实际结果 |
|---|---|
| 独立 Python 环境 | Windows / Python 3.11，新 venv 安装基础与 dev 依赖成功，未安装 Torch、Transformers 或 MicroLM |
| 依赖一致性 | `pip check` 通过；版本记录于 `constraints.txt` 与 `THIRD_PARTY_PYTHON.json` |
| 后端全量测试 | 1,019 passed、10 skipped、2 warnings |
| 真实进程端到端 | API 与 Worker 进程测试通过，包含数据导入、外部命令、终态成功、产物登记与项目关联检查 |
| 数据库 | 测试使用独立临时数据库，执行真实 Alembic 迁移，逐测试隔离状态 |
| 前端依赖 | `pnpm install --frozen-lockfile` 通过 |
| 前端测试 | 40 个文件，195 项通过 |
| 前端构建 | TypeScript 与 Vite 生产构建通过，存在大 chunk 提示 |
| Python 包 | 平台 wheel 与独立 SDK wheel 构建成功，包含 LICENSE；平台包包含 NOTICE |
| 安装包边界 | wheel 成员无模型实现、模型套件、私有 SFT 优化、GGUF 或 SafeTensors 权重 |
| wheel 独立运行 | 从 wheel 非 editable 安装，以 Python 隔离模式导入 API/Worker/SDK，临时数据库启动及 health/catalog 检查通过；模型运行时确实不存在 |
| 安装脚本 | 所有随附 Bash 脚本语法检查及 PowerShell 解析通过 |
| 发布文件扫描 | 已检查候选文件范围、常见 Token/私钥格式、模型/数据库/归档等不应分发的文件类型，未发现匹配项 |
| 第三方许可 | 保存 221 个 Web 安装包的许可声明及找到的原始许可文本；基础 Python 验证环境清单单独记录 |

10 个跳过项包括 6 项可选模型/Tokenizer 运行时、Windows 下的 Unix 进程组测试和无权限创建符号链接的测试，以及 2 个已有替代实现的旧下载测试。跳过不代表这些外部运行时已经验证。

两个警告分别是 FastAPI/Starlette 的 httpx 测试客户端弃用提示，以及缺少 Manifest 的负例测试中的 SQLAlchemy 空主键查询提示。它们没有造成测试失败。

## 未验证或未执行

- 迁移前 CI 已通过：Windows / Ubuntu × Python 3.11 / 3.12 的测试与 wheel 构建，以及 Web 和 Shell 检查。迁移时实现代码保持一致；新仓库运行结果见 [Actions](https://github.com/CloudGao/ForgeLM/actions)。
- CI 不等同于真实 Linux 主机安装、GPU 训练或远程 SSH 部署验收。
- 没有进行真实 GPU 训练、远程主机部署或全部可选推理/评测环境验收。
- 凭据检查是静态模式与文件范围检查，不是渗透测试、完整秘密检测或全部依赖漏洞审计。
- Python/Web 清单不涵盖未来另行安装的模型、GPU 驱动、外部套件和离线 wheel 的所有分发义务。

平台后续在 ForgeLM 仓库开发；本地模型文件及训练状态保留，模型源码单独管理。

## 原工作目录落地检查

已应用到原仓库 `codex/platform-source-cleanup` 分支。本地模型与配置保持原路径但不受 Git 跟踪，原协作规则保留。清理前 1,042 个文件的备份及保留的 151 个本地文件均通过 SHA-256 校验。本轮在原工作目录执行相关后端测试：30 passed、1 skipped；前端测试：40 个文件、195 passed；TypeScript/Vite 构建通过，仍有大 chunk 提示。上表全量后端结果来自此前隔离候选验证，本轮未重复执行全量后端测试。
