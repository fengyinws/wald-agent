# 部署与运维

## 支持的运行方式

服务使用一个 Python 进程和本地 SQLite WAL 数据库。并发调用上游 API，通过异步队列限制同时执行的数量；数据库操作在线程中执行，不阻塞事件循环。数据库文件锁阻止第二个进程打开同一服务数据库，包括错误配置的多个 uvicorn worker。

默认运行命令是 `wald serve`。Docker Compose 提供相同的单实例部署。不要通过增加 worker 或共享 NFS 卷扩容；多实例需要先实现共享数据库、分布式幂等和共享限流。此版本没有持久化后台任务调度器：HTTP 连接上的同步请求执行完成后返回，服务意外终止的未完成记录会在下次启动时标为失败。

## 从零部署

```bash
uv sync --locked
uv run wald init
# 编辑 config/.env，填入 LLM_API_KEY；使用 Jev 时设置 TYPESAFE_API_KEY 并启用 jev。
uv run wald doctor --live
uv run wald serve
```

已有配置时不要重新执行 init；修改 `config/.env` 后重启服务。初始化生成的服务密钥与大模型供应商密钥互相独立。生产模式缺少服务密钥、密钥过短或已启用供应商缺少凭据时，启动直接失败。

`config/` 可以提交，包含公开的 `config/.env.example` 模板与 `config/README.md` 配置说明。真实密钥放在 `config/.env`；`.env`、`.env.local`、`.env.backup` 等文件被 Git 忽略，也从 Docker 构建上下文和发布包中排除。不要将真实密钥填入模板或示例代码。Compose 从 `config/.env` 读取配置后注入容器环境，镜像中只包含公开模板。旧版根目录 `.env` 需迁移到 `config/.env`，程序不会同时读取两份配置。

```bash
docker compose up --build -d
docker compose ps
docker compose logs --tail 100 wald
curl --fail http://127.0.0.1:8000/readyz
```

Compose 将服务端口绑定到 127.0.0.1；对外提供服务时在前面配置 HTTPS 反向代理。代理的请求体限制应至少与 MAX_BODY_BYTES 对齐，读超时应大于 REQUEST_TIMEOUT_SECONDS。密钥放在环境变量或运行环境的密钥管理中，不要写入镜像。

Dockerfile 使用 uv.lock 安装依赖、非 root 运行并包含就绪检查。Compose 的数据库使用命名卷，普通容器重建不会丢失记录；`docker compose down -v` 会删除数据卷。容器停止宽限期默认 170 秒；增加服务总超时时，也要增加此宽限期。

## 客户端与权限

默认 `WALD_API_KEY` 对应客户端 `default`。多个业务调用者可在 `config/.env` 中设置：

```dotenv
WALD_API_KEYS={"support":"填写至少32字符的随机密钥","quality":"填写另一个独立随机密钥"}
```

上面是格式示意，替换为真实随机密钥后使用。命名客户端的记录、幂等键和限流各自独立；不能查询或复核其他客户端记录。`default` 和 `anonymous` 为保留名称。服务上限为 64 个命名客户端。所有业务接口、监控和开发文档要求 Bearer 服务密钥，只有 `/healthz` 与 `/readyz` 公开。

密钥配置在启动时读取。轮换时保留原客户端名称、替换其密钥并重启，这样可继续读取历史记录；改变名称会产生新的记录命名空间。复核者的 `reviewer` 是业务审计字段，调用者的实际访问身份由服务 API key 确定，并没有独立的人类登录系统或管理员跨客户端查询接口。

开发模式允许未设置服务密钥；此时所有调用属于 `anonymous`。生产配置禁止这种运行方式。开发 Swagger 页面受中间件鉴权保护，访问它时需由浏览器或反向代理携带服务 Bearer header；命令行可直接下载 `/openapi.json`。

## 资源与超时配置

| 环境变量 | 默认 | 含义 |
| --- | --- | --- |
| `MAX_CONCURRENT_REQUESTS` | 8 | 同时执行的决策/图像管线数；一张图片顺序使用视觉和 Jev |
| `MAX_QUEUED_REQUESTS` | 32 | 额外允许等待的决策数；满时返回 503 |
| `QUEUE_TIMEOUT_SECONDS` | 5 | 等待执行名额的上限 |
| `RATE_LIMIT_PER_MINUTE` | 120 | 每客户端令牌桶容量及每分钟补充量，允许相应突发 |
| `REQUEST_TIMEOUT_SECONDS` | 150 | 单条业务请求含排队/上游阶段的执行时限；批量含全部子项 |
| `API_TIMEOUT_SECONDS` | 60 | 一次供应商逻辑调用的总时限，含重试、等待和响应读取 |
| `API_CONNECT_TIMEOUT_SECONDS` | 5 | 建连及获取连接池名额的时限 |
| `API_MAX_RETRIES` | 2 | 首次调用失败后最多重试次数 |
| `API_RETRY_BASE_SECONDS` / `API_RETRY_MAX_SECONDS` | 0.5 / 10 | 无 Retry-After 时的指数退避和随机抖动上限 |
| `CIRCUIT_FAILURE_THRESHOLD` | 5 | 连续失败的逻辑调用数，达到后打开熔断器 |
| `CIRCUIT_COOLDOWN_SECONDS` | 30 | 熔断冷却时间，之后仅允许一个半开探测 |
| `MAX_BODY_BYTES` | 16 MiB | 包括 chunked 输入的整个 HTTP 请求体上限 |
| `BODY_TIMEOUT_SECONDS` | 15 | 收完整个请求体的总时限 |
| `MAX_STATE_BYTES` | 256000 | 每个文本决策序列化后（含问题）的字节上限 |
| `API_MAX_RESPONSE_BYTES` | 2 MiB | 单次上游响应体上限 |
| `RETENTION_DAYS` | 30 | 终态记录保留期，同时也是幂等记录保留期 |
| `AUDIT_STORE_INPUTS` | true | 是否保留业务 state 和输入视觉事实 |

令牌桶对文本/图像决策及记录查询生效；批量请求每个子项各消耗一个令牌。复用幂等结果也占用限流额度。复核提交依赖鉴权、请求体限制及数据库并发控制，没有额外消耗令牌。限流、队列和熔断状态在进程重启时重置，业务记录不会重置。

文本每份请求最多 64 个问题，Choice 最多 255 个候选项，Score 支持 2–10 级，批量最多 32 项。模型上下文和服务商 Structured Outputs 限额也必须满足，2048 输出 token 的默认值不适合一次输出大量候选概率；需要同时调整 `LLM_MAX_OUTPUT_TOKENS` 和供应商模型能力。

`REQUEST_TIMEOUT_SECONDS` 从记录建立后开始，涵盖排队与计算；请求体读取、身份校验、SQLite 提交和最终 HTTP 发送会额外花费时间。SDK 默认总超时 155 秒，可根据代理和服务配置增加。SDK 响应体默认上限 16 MiB，可通过 `max_response_bytes` 调整。

## 重试、幂等和失败处理

自动重试只用于建连失败/连接池超时，以及 HTTP 408/429/500/502/503/504/529。带有已知余额耗尽错误码的 429、鉴权失败、非法请求、输出 JSON/概率不合法不会重试。读超时或写入后网络失败可能已被供应商接收和计费，因此不会自动重放。服务尊重 Retry-After（秒或 HTTP 日期）；等待会超出总时限时直接返回失败，不提前重试。

供应商返回暂时性 HTTP 错误的重试仍可能产生多次计费；计费敏感场景设置 `API_MAX_RETRIES=0`。响应的 `attempts` 记录实际调用尝试数。LLM、Jev、视觉客户端各有独立熔断器，半开探测失败或取消后重新进入冷却。

客户端应为业务事件提供稳定的 `Idempotency-Key`（1–128 个 ASCII 字母、数字或 `._:-`）。作用域是客户端身份，同一客户端内跨接口共享。幂等比对包括请求、提供商、模型和相关配置：

1. 首次接收：先持久化 processing，再调用上游。
2. 相同键正在执行：409，调用方稍后查询或用同一键重试。
3. 相同键已完成：返回持久化结果，不再次调用模型。
4. 相同键但输入/模型配置不同：409，使用新的业务事件键。
5. 相同键已失败：重放失败，不自动重做；确认业务状态后用新键显式重新执行。

意外停机后，未完成记录标记 `request_interrupted`。数据库无法判断供应商在断开前是否已计费，所以这不是跨网络的“恰好执行一次”保证。超过保留期清理后，同一键可以再次执行；业务需要更长去重周期时增加保留期或保留自己的事件表。

批量 HTTP 200 表示已收集各项结果，**不表示所有项成功**，必须逐个检查 `result`/`error`。子项有自己的记录 ID，复核队列只列子项，不重复列父批次。父批次超时/取消会取消并等待剩余子项；已完成子项仍保存。批量失败项不重新执行，重试应只选择确认要重做的项并使用新业务键。

错误响应包含 `error.code`、`error.message`、`request_id`，已建立记录的失败还包含 `decision_id`。SDK 的 `RemoteServiceError` 保留相同信息；网络错误为 `ProviderError`，超时为 `ProviderTimeout`。不将原始上游响应正文回传客户端。

| HTTP | 场景 |
| --- | --- |
| 401 | 服务密钥缺失/错误 |
| 404 | 记录不存在、已过保留期或不属于该客户端 |
| 408 / 413 | 请求体读取超时 / 过大 |
| 409 | 幂等冲突、记录仍在执行、复核已提交或版本冲突 |
| 422 | 输入/复核标签不合法 |
| 429 | 本服务客户端限流，附 Retry-After |
| 502 | 上游调用失败、拒答、截断或响应校验失败 |
| 503 | 队列满、熔断、数据库不可用、配置缺失或请求中断 |
| 504 | 上游或整个决策执行超时 |

上游的 429 由服务映射为 502，以便与本服务限流区分；可用的 Retry-After 仍会传递。

## 记录、复核与数据导出

每次调用保存模型结果、时间、问题、客户端身份以及可选输入。复核要求一次提交所有原始问题的标签：Choice 候选 key、Boolean 真正的 JSON 布尔值、Score 合法整数等级。复核不覆盖模型原始结果，使用 revision 比对原子写入；已复核记录不能再次提交。

`AUDIT_STORE_INPUTS=false` 仅删除原始 state/facts，仍保留问题、模型输出与复核记录。视觉流程的输出包含视觉事实，因此仍会存储这些事实；这不是无数据存储模式。图片本身不入库。若需要业务审计或根据原始证据复核，应在业务系统保留相应资料。

```bash
uv run wald export-feedback --url http://127.0.0.1:8000 \
  --output reports/reviewed.jsonl
# 命名客户端增加 --client support
```

导出通过 HTTP 分页读取当前客户端已复核的文本决策，只包含保留了 state 的记录；图片、批次父项和缺失输入记录不导出。每页增量写临时文件，全部完成后原子替换输出；格式可以直接传给 `wald-compare --dataset`。导出不是数据库快照，过程中被保留期清理的游标会导致明确失败。大批量导出应选择低负载时段，并确保调用者的查询限额足够。

## 监控与恢复

- `/healthz`：进程存活。
- `/readyz`：启用供应商已配置密钥、存储可读、服务接受请求；不验证远端凭据是否有效。
- `wald doctor --live`：实际调用已启用供应商，验证协议和账号权限；不测试视觉模型。
- `/metrics`：`wald_http_requests_total`、`wald_http_duration_seconds`、`wald_upstream_attempts_total`、`wald_decisions_total`。Prometheus 抓取需带服务 Bearer token；使用 `/metrics` 不消耗业务查询额度。
- 日志：JSON 格式的事件、请求 ID、规范化路由、状态、耗时与错误类型，不记录请求正文、图片或 Authorization。请求 ID 可由调用方提供合法 `X-Request-ID`，也会传给上游；请求 ID 不充当身份凭证。

关注 429/503/504 数量、HTTP 延迟分位数、上游重试数、转人工比例以及磁盘容量。低错误率不能证明概率校准，应周期性用真实人工标签评估 Brier/ECE/准确率。

数据库自动每分钟清理超过保留期的终态记录，启动也会清理；删除的空间供 SQLite 复用，不保证文件立即缩小。日志及报告文件由部署环境另行轮转和保留。

```bash
# 本地服务在线备份：拒绝覆盖现有备份文件。
uv run wald backup --output reports/wald-backup.sqlite3
# 容器内数据库在线备份，然后导出（本地目标自行命名）。
docker compose exec wald wald backup --output /app/data/backup.sqlite3
docker compose cp wald:/app/data/backup.sqlite3 ./reports/wald-backup.sqlite3
```

使用 SQLite backup API 创建一致快照，不要在服务运行时只复制主 sqlite3 文件而遗漏 WAL。恢复时停止服务，保留当前数据库和 WAL/SHM 的独立备份，将备份恢复到配置路径，再启动服务；不得将旧 WAL/SHM 混入新恢复的数据库。容器卷内文件需归 UID 10001 所有。恢复完成后检查 `/readyz` 并抽查记录。

数据库版本由 `PRAGMA user_version` 管理，当前为 1；程序拒绝打开未知版本。升级前备份，再安装锁定依赖，运行测试并重启。尚没有跨版本迁移需求时不提供空的迁移脚本；后续变更 schema 必须同时实现迁移和回滚方案。
