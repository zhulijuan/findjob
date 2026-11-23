# findjob-llm-harness

配置驱动的 LLM 任务执行框架（Python）。所有模型参数、Prompt 模板、任务定义均外置于配置文件，代码零业务参数——新增任务只改 YAML，不改代码。

> 当前进度：W1 最小问答 demo ✅ → W2 RAG 信贷政策知识库 v1（进行中）→ 后续按 [docs/harness-plan.md](docs/harness-plan.md) 路线图迭代（Agent → 评测体系）。

## 功能特性

- **配置驱动**：任务、Prompt 模板、模型参数全部在 `config/config.yaml`，pydantic 严格校验（配错早报错）
- **多 Provider**：兼容 OpenAI 格式的 API 均可接入（DeepSeek、Qwen/DashScope、SiliconFlow 等），改配置即可切换
- **Prompt 模板**：jinja2 渲染，缺失变量立即报错（StrictUndefined）
- **任务级覆盖**：单个任务可覆盖模型、温度、max_tokens、thinking 模式
- **流式输出**：`--stream` 增量打印
- **用量记录**：每次调用自动落盘 token 用量 / 耗时 / 任务名（`logs/usage.jsonl`），LLM 与 embedding 调用均记录
- **环境变量覆盖**：`DEEPSEEK_MODEL`、`DEEPSEEK_BASE_URL` 等环境变量优先于配置文件，便于临时切换
- **RAG 知识库**（W2）：中文条款感知分块 → BGE-M3 向量化 → FAISS 检索 → 带引用来源的问答

## 目录结构

```
findjob/
├── config/config.yaml      # 全部可配置项：provider、模型参数、任务与 Prompt 模板、RAG 参数
├── harness/
│   ├── config.py           # 配置加载（YAML + 环境变量覆盖）+ pydantic 校验
│   ├── llm.py              # LLM 客户端（含用量日志，普通/流式均记录）
│   ├── embedding.py        # Embedding 客户端（批量，BGE-M3）
│   ├── chunker.py          # 中文条款感知分块（章/条/标题行切分 + 句级 overlap）
│   ├── vectorstore.py      # 向量库抽象 + FAISS 实现（归一化内积 ≡ 余弦）
│   ├── retriever.py        # 检索管线（查询向量化 → top_k）
│   ├── rag.py              # 建索引 + context 拼装 + RAG 问答
│   └── runner.py           # 任务运行器（模板渲染 + 消息构建）
├── main.py                 # 命令行入口
├── data/corpus/            # 语料：公开监管文件 + 合成文档（入库）
├── docs/harness-plan.md    # 能力路线图与工程约束
├── docs/w2-plan.md         # W2 详细计划
└── .env                    # 密钥（不入库，见 .env.example）
```

## 快速开始

```bash
# 1. 创建虚拟环境并安装依赖
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env：DEEPSEEK_API_KEY（platform.deepseek.com/api_keys）
#           SILICONFLOW_API_KEY（siliconflow.cn，RAG 需要）

# 3. LLM 任务
.venv/bin/python main.py --list                              # 查看任务列表
.venv/bin/python main.py --ask "什么是信贷风控中的 FPD30？"    # 即问即答
.venv/bin/python main.py --task intro --var topic=DeepSeek   # 运行配置中的任务

# 4. RAG 知识库问答
.venv/bin/python main.py --build-index                       # 语料 → 分块 → 向量化 → 索引
.venv/bin/python main.py --rag "互联网贷款单户授信额度上限是多少？"  # 带引用来源的问答
.venv/bin/python main.py --rag "FPD30 怎么定义" --stream     # 流式输出
```

## RAG 工作原理

```
建索引：data/corpus/*.md ──分块(章/条/标题感知, size+overlap可配)──> BGE-M3 向量化 ──> FAISS 持久化
问答：  问题 ──> 向量化 ──> FAISS top_k 检索 ──> 拼装带来源标注的 context ──> LLM 生成答案 + 引用列表
```

分块策略按中文条款结构（第X章/第X条/标题行）优先切分，超长块按句切分并携带 overlap；语料只用公开监管文件与自写合成文档（数据红线见 [docs/harness-plan.md](docs/harness-plan.md)）。

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

embedding:
  base_url: https://api.siliconflow.cn/v1
  api_key_env: SILICONFLOW_API_KEY
  model: BAAI/bge-m3
  batch_size: 16

vectorstore:
  type: faiss
  index_path: data/index/credit_policy

rag:
  corpus_dir: data/corpus
  chunk_size: 400                   # 块最大字符数
  chunk_overlap: 50                 # 块间重叠
  top_k: 5

tasks:
  intro:
    system_prompt: 你是一个简洁的助手
    user_prompt: 用一句话介绍 {{ topic }}。
    # model: deepseek-v4-pro        # 任务级模型覆盖
    # thinking: true                # 开启思考模式

  rag:
    temperature: 0.1
    system_prompt: 你是信贷政策助手。仅依据提供的资料回答问题……
    user_prompt: |
      资料：
      {{ context }}

      问题：{{ question }}
```

新增任务只需在 `tasks` 下加一段配置，无需改代码。

## Roadmap

见 [docs/harness-plan.md](docs/harness-plan.md)：

- ✅ W2：RAG 信贷政策知识库 v1（分块 / BGE-M3 / FAISS / 带引用问答）
- W3：混合检索（BM25+向量）、bge-reranker、20 条标注测试集 + recall@k 评测落盘
- W5–W7：Agent 工具注册、结构化输出、规则+模型双引擎
- W9–W11：指标异常归因 Agent + 评测体系（批量跑批 / LLM-as-judge / 结果落盘）
