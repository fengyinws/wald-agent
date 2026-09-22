# 配置说明

```text
config/
├── README.md      # 配置说明，可以提交
├── .env.example   # 空密钥模板，可以提交
└── .env           # 本地实际配置，Git 忽略，不提交
```

在项目根目录执行：

```bash
uv sync --locked
uv run wald init
```

`wald init` 根据模板创建 `config/.env`，自动生成 `WALD_API_KEY`，并将密钥文件权限设为 `0600`。已有 `.env` 不会被覆盖。

编辑 `config/.env`：

| 配置项 | 用途 |
| --- | --- |
| `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL` | 大模型 API 密钥、地址、模型名称 |
| `TYPESAFE_API_KEY`、`JEV_MODEL` | Jev API 密钥和模型 |
| `VISION_API_KEY`、`VISION_BASE_URL`、`VISION_MODEL` | 视觉模型；密钥和地址默认共用 LLM 配置 |
| `WALD_API_KEY` | 访问本项目 HTTP API 的密钥，由初始化命令生成 |
| `ENABLED_PROVIDERS` | 文本服务默认 `["llm"]`；使用 HTTP Jev/图像接口时设置 `["llm","jev"]` |

只使用文本决策时先填写 LLM 配置。对比测试还需要 Jev 密钥；图像识别需要可用的视觉模型。

```bash
uv run wald doctor          # 检查本地配置，不调用模型
uv run wald doctor --live   # 实际调用已启用模型，产生 API 用量
uv run wald serve
```

程序默认读取当前工作目录下的 `config/.env`，环境变量优先；Docker Compose 从同一路径注入配置。修改配置后重启服务。

真实密钥只写入 `.env`，不要填写到 `.env.example` 或本说明中。`.env`、`.env.local`、`.env.backup` 等文件被忽略，`.env.example` 可以提交。完整参数见模板及 [运维说明](../docs/operations.md)。
