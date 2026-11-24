# 套件开发

从 `examples/plugin-suites/text-tools` 查看完整源码示例，或使用 SDK 的 `init` 命令生成新套件。生成器默认许可证为 `UNLICENSED`，开发者必须自行确定新套件许可；仓库随附的 text-tools 示例已设为 Apache-2.0。

套件由 `suite.json`、节点定义、配置 Schema、实现 wheel、依赖锁、README 和 LICENSE 组成。组件通过 Python distribution 的 `forgelm.components.v1` entry point 声明；不要把执行代码放进控制面。

## 构建

1. 修改套件 ID、发行包名、组件版本与目标环境。
2. 在匹配的操作系统、CPU 架构和 Python 版本上运行 SDK `lock`。目标名称与 `suite.json` 中的 `targets[].id` 一致。
3. 使用 `--sdk-source packages/forgelm_plugin_sdk` 从本仓库构建 SDK，不要求它已经发布到 PyPI。
4. 运行 `pack` 和契约测试，检查最终 `.flsuite` 的内容、哈希、依赖与许可证。
5. 通过平台插件管理入口导入包，信任、安装并检查 Worker 的实际安装回执，再在项目中使用节点。

`lock` 会调用 pip 构建与解析依赖，需要可用的下载网络；新版本使用标准构建隔离，不依赖宿主预装 setuptools/wheel。运行时安装使用包中已锁定的 wheel 和哈希；构建阶段与运行阶段不可混为一谈。

## 任务契约

组件通过 Request/Context 接收冻结输入和配置，使用 Context 定位输入、输出目录，检查取消状态并发送事件。结果声明输出制品，由 Worker 校验和登记。服务组件使用 SDK 的服务协议，认证、取消和进程生命周期有独立约束。

训练组件还必须明确声明自身的状态、暂停、恢复与 Checkpoint 契约。平台不提供模型 Trainer；模型结构和训练循环应留在独立模型仓库/套件中。

不要覆盖已经发布的包版本。源码或依赖变化应产生新版本和新的锁/哈希。虚拟环境隔离不能限制恶意代码权限，只能安装可信套件。
