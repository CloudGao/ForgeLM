# 文本处理套件示例

本目录是可修改、构建和导入的二开样例，包含一个 normalize 任务组件。

在仓库根目录、Python 3.12 Windows x86_64 环境执行：

```sh
python -m forgelm_plugin_sdk.cli lock --root examples/plugin-suites/text-tools --target windows-x86_64-cp312-cpu --sdk-source packages/forgelm_plugin_sdk
python -m forgelm_plugin_sdk.cli pack --root examples/plugin-suites/text-tools --output dist/suites
python -m forgelm_plugin_sdk.cli test dist/suites/example.text-tools-1.0.0.flsuite --contract --fixtures examples/plugin-suites/text-tools/tests/contract.json
```

其他平台请先修改 target 为实际环境并在对应机器构建。导入 .flsuite 后整套安装，选择业务项目，上传 source.txt 或选择 text/plain 制品，设置 lowercase 并运行。也可在画布“套件节点”中使用，result 输出能接到另一个 normalize 的 source 输入。

修改实现后更新组件 wheel 版本与套件版本，重新锁依赖、打包、运行契约测试；不要覆盖已发布版本。本示例使用 Apache-2.0；新建自己的套件时需明确它的许可证和第三方声明。

完整说明见[开发指南](../../../docs/developer/plugin-suite-development.md)。
