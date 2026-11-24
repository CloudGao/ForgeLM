# 贡献说明

修改前阅读 `docs/DESIGN.md`、`docs/ARCHITECTURE.md` 与 `AGENTS.md`。保持平台代码与外部模型代码分离；不要提交模型权重、训练语料、个人工作流、凭据或运行产物。

使用新虚拟环境安装 `python -m pip install -c constraints.txt -e ".[dev]"`，Web 使用 `pnpm install --frozen-lockfile`。提交前运行相关 pytest/Vitest 测试及 Web 构建，说明未验证的操作系统、硬件与可选运行时。不得把缺少模型运行时的跳过项写成模型训练验证通过。

复用现有领域 API、工作流实例、任务与制品血缘。重大架构变化先写方案讨论；当前状态文档只描述已经实现并验证的行为。

依赖更新需要检查许可证，更新适用的锁/约束文件、第三方清单和 Web 构建随附的许可文本。Python 依赖清单记录的是特定验证环境；发布容器、离线 wheel 或 GPU 套件时须核对实际产物。

你提交的贡献须有权按项目 Apache-2.0 许可提供。保留引用或改编代码的来源、版权和许可声明。安全问题按 `SECURITY.md` 报告。
