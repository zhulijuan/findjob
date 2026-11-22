# findjob-llm-harness

配置驱动的 LLM 任务执行框架（Python）。所有模型参数、Prompt 模板、任务定义均外置于配置文件，代码零业务参数——新增任务只改 YAML，不改代码。

> 当前为 W1 阶段产出：最小问答 demo + 可复用任务框架。后续按 [docs/harness-plan.md](docs/harness-plan.md) 路线图迭代（RAG 知识库 → Agent → 评测体系）。

## 功能特性

- **配置驱动**：任务、Prompt 模板、模型参数全部在 `config/config.yaml`，pydantic 严格校验（配错早报错）
- **多 Provider**：兼容 OpenAI 格式的 API 均可接入（DeepSeek、Qwen/DashScope 等），改配置即可切换
- **Prompt 模板**：jinja2 渲染，缺失变量立即报错（StrictUndefined）
- **任务级覆盖**：单个任务可覆盖模型、温度、max_tokens、thinking 模式
- **流式输出**：`--stream` 增量打印
- **用量记录**：每次调用自动落盘 token 用量 / 耗时 / 任务名（`logs/usage.jsonl`），跑批前可估算成本
- **环境变量覆盖**：`DEEPSEEK_MODEL`、`DEEPSEEK_BASE_URL` 等环境变量优先于配置文件，便于临时切换

## 目录结构

```
findjob/
├── config/config.yaml      # 全部可配置项：provider、模型参数、任务与 Prompt 模板
├── harness/
│   ├── config.py           # 配置加载（YAML + 环境变量覆盖）+ pydantic 校验
│   ├── llm.py              # LLM 客户端（含用量日志，普通/流式均记录）
│   └── runner.py           # 任务运行器（模板渲染 + 消息构建）
├── main.py                 # 命令行入口
├── docs/harness-plan.md    # 能力路线图与工程约束
└── .env                    # 密钥（不入库，见 .env.example）
```

## 快速开始

```bash
# 1. 创建虚拟环境并安装依赖
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env，填入你的 DEEPSEEK_API_KEY（platform.deepseek.com/api_keys 申请）

# 3. 运行
.venv/bin/python main.py --list                              # 查看任务列表
.venv/bin/python main.py --ask "什么是信贷风控中的 FPD30？"    # 即问即答
.venv/bin/python main.py --ask "你好" --stream               # 流式输出
.venv/bin/python main.py --task intro --var topic=DeepSeek   # 运行配置中的任务
.venv/bin/python main.py --task reasoning                    # thinking 模式示例
```

## 配置说明

三层优先级：**命令行参数 > 环境变量 > config.yaml**。

```yaml
llm:
  base_url: https://api.deepseek.com
  api_key_env: DEEPSEEK_API_KEY     # 从哪个环境变量读密钥
  model: deepseek-flash
  temperature: 1.0
  max_tokens: 2048
  timeout: 60
  usage_log: logs/usage.jsonl       # 用量日志路径，置 null 关闭

  # 切换 Qwen（DashScope）示例：
  # base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
  # api_key_env: DASHSCOPE_API_KEY
  # model: qwen-plus

tasks:
  intro:
    system_prompt: 你是一个简洁的助手
    user_prompt: 用一句话介绍 {{ topic }}。
    # model: deepseek-v4-pro        # 任务级模型覆盖
    # thinking: true                # 开启思考模式
```

新增任务只需在 `tasks` 下加一段配置，无需改代码。

## Roadmap

见 [docs/harness-plan.md](docs/harness-plan.md)：

- W2–W4：RAG 信贷政策知识库（embedding / 向量库 / 分块 / rerank / 召回率评测）
- W5–W7：Agent 工具注册、结构化输出、规则+模型双引擎
- W9–W11：指标异常归因 Agent + 评测体系（批量跑批 / LLM-as-judge / 结果落盘）
