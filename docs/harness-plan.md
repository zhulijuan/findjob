# Harness 规划：能力地图与约束

> 依据《AI转型三个月行动计划》制定 ｜ 2026-11-21 立项
> 定位：harness 是通用执行框架（配置 → 调用 → 记录），三个项目（RAG / 审批 Agent / 指标归因 Agent）各自独立模块，不混入 harness 核心。

## 现状（W1 已完成）

- 配置驱动任务 runner：任务定义、Prompt 模板、模型参数全部外置于 `config/config.yaml`
- 多 provider 支持：`base_url` / `api_key_env` / `model` 可配，DeepSeek 与 Qwen（DashScope）仅改配置切换
- 环境变量覆盖 YAML（`DEEPSEEK_BASE_URL` / `DEEPSEEK_MODEL`），不改文件临时切换
- pydantic 严格校验（extra=forbid）；jinja2 模板渲染（StrictUndefined，缺变量早报错）
- 流式输出、thinking 模式、任务级参数覆盖
- token 用量 JSONL 落盘（`logs/usage.jsonl`，含任务名/耗时/用量，流式亦记录）

## 能力需求地图（按计划周次）

| 阶段 | 计划事项 | Harness 待建能力 | 状态 |
|---|---|---|---|
| W2–W4 RAG | embedding、向量库、分块、rerank、召回率 | embedding 客户端（provider 可配）、向量库抽象（FAISS / Milvus Lite 配置切换）、分块参数外置（chunk_size/overlap）、检索参数（top_k、BM25/向量混合权重）、标注测试集加载 + 召回率计算 + 结果落盘对比 | 未建 |
| W4 产品化 | Streamlit/FastAPI、Docker | UI/服务入口独立于 harness，harness 保持纯后端库 | 未建 |
| W5–W7 Agent | function calling、ReAct、规则+模型双引擎 | 工具注册（YAML 定义工具名/描述/实现映射，自动生成 schema）、结构化输出（JSON Schema + pydantic 校验 + 失败重试参数）、多步 pipeline 定义（解析→校验→生成）、规则文件外置 | 未建 |
| W9–W10 归因 | 合成指标数据、语义层元数据、下钻 | 合成数据生成配置、指标语义层元数据文件（口径/上游链路/同环比规则）、下钻贡献分解与 ETL 状态查询工具集 | 未建 |
| W11 评测 | 归因准确性评测、LLM-as-judge、badcase | 批量跑批 runner、标准答案对比评测、judge 任务类型（judge prompt 外置）、结果 JSONL 落盘 + run_id/配置快照可复现 | 未建 |

## 约束铁律

1. **数据合规红线**：语料与测试数据只用公开监管文件 + 合成数据；`data/raw/` 已加入 .gitignore，银行真实数据/行内文档永不入库
2. **配置外置**：新能力全部走 YAML（provider、检索参数、工具注册、评测集路径），代码零业务参数
3. **成本可见**：每次 LLM 调用记录 token 用量到 `logs/usage.jsonl`，跑批前可估算成本
4. **可复现**：评测跑批 temperature 可固定；结果带时间戳、任务名、变量快照
5. **分期节奏**：按 W2 → W5 → W9 → W11 顺序实现，不提前造后续阶段轮子；裁剪顺序对齐计划（砍 Dify 对照 → 砍微调 → 归因维度精简）
6. **工程习惯**：连续 commit、每个能力有最小验证脚本（面试官会看）

## 实施顺序（下一步从 W2 开始）

1. W2：embedding + 向量库 + 分块配置 → 信贷政策知识库 v1
2. W3：混合检索 + 标注测试集 + 召回率记录（评测叙事第一块素材）
3. W4：Streamlit/FastAPI 封装
4. W5–W7：工具注册 + 结构化输出 + 双引擎 pipeline
5. W9–W11：合成数据 + 归因工具集 + 批量评测体系
