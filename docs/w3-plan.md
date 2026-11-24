# W3 计划：知识库 v2 —— 混合检索 + Rerank + 首个检索评测

> 依据《AI转型三个月行动计划》W3 与 [harness-plan.md](harness-plan.md) 能力地图制定
> 产出定位：**检索效果对比记录** —— B 路径"评测"叙事的第一块素材
> 延续决策：embedding/reranker 走 SiliconFlow API，向量库 FAISS，配置驱动零硬编码

## 目标与验收标准

- 混合检索可用：BM25 + 向量 + RRF 融合，`retrieval.mode` 一键切换（vector / bm25 / hybrid）
- Reranker 开关可用（SiliconFlow `bge-reranker-v2-m3`），默认关
- 查询改写开关可用（LLM 改写），默认关
- 20 条标注测试集 `data/eval/retrieval_testset.jsonl`（question + gold_chunk_id + 题型），我起草、用户审核
- 评测 runner：`--eval` 一键跑 4 种配置对比，输出 recall@3/5/10、MRR@10，结果 JSONL 落盘 + markdown 对比表
- 引用溯源：答案引用保留块级来源；评测记录含每次命中的明细
- 全部配置驱动；README 更新；连续 commit

## 架构与数据流

```
问答（v2）：
问题 ──> [查询改写·可选·LLM] ──> 双路并行：BM25(稀疏) + FAISS(稠密)
     ──> RRF 倒数排名融合 ──> [bge-reranker 精排·可选] ──> top_k ──> context 拼装 ──> LLM 生成答案 + 引用

评测：
testset.jsonl ──> 遍历 4 配置(vector / bm25 / hybrid / hybrid+rerank)
             ──> 每题检索 ──> 与 gold_chunk_ids 比对 ──> recall@k / MRR ──> 落盘 + 对比表
```

## 模块设计

| 模块 | 职责 | 关键点 |
|---|---|---|
| `harness/bm25.py` | BM25Index：构建 / 搜索 / 持久化 | jieba 分词 + rank-bm25；自定义词典可配（授信额度、FPD30 等专业词）；随 --build-index 一并构建 |
| `harness/fusion.py` | RRF 融合 | `score = Σ w_i / (rrf_k + rank_i)`，双路候选数可配 |
| `harness/reranker.py` | RerankerClient | SiliconFlow rerank API（query+documents → scores），批量、重试、用量日志 |
| `harness/retriever.py` v2 | HybridRetriever | 按配置组装：改写 → 双路 → 融合 → 精排；保持原 `Retriever` 兼容 |
| `harness/rewrite.py` | 查询改写 | 走 LLMClient + tasks.rewrite 模板；改写前后 query 都进评测明细 |
| `harness/eval.py` | 评测 runner | 加载测试集 → 多配置跑批 → recall@3/5/10、MRR@10 → `logs/eval/eval_*.jsonl` + markdown 对比表 |
| `main.py` | `--eval [path]` | 评测入口，默认 20 条测试集 |

## config.yaml 新增段（草案）

```yaml
retrieval:
  mode: hybrid              # vector | bm25 | hybrid
  bm25_candidates: 10       # BM25 路送融合的候选数
  vector_candidates: 10     # 向量路送融合的候选数
  rrf_k: 60                 # RRF 常数（惯例 60）
  rerank:
    enabled: false
    base_url: https://api.siliconflow.cn/v1
    api_key_env: SILICONFLOW_API_KEY
    model: BAAI/bge-reranker-v2-m3
    top_m: 10               # 送精排的候选数
  rewrite:
    enabled: false
  jieba_userdict: data/eval/userdict.txt   # 自定义分词词典，可空

tasks:
  rewrite:
    description: 查询改写（配合 retrieval.rewrite.enabled 使用）
    temperature: 0.0
    system_prompt: 将用户的口语化信贷问题改写为适合检索的书面表述，补充同义词与关键术语，只输出改写后的查询，不要回答。
    user_prompt: "{{ question }}"
```

`.gitignore` 追加：无新增（data/eval/ 入库——测试集与词典是评测叙事素材，必须提交）。

## 评测设计

**测试集格式**（`data/eval/retrieval_testset.jsonl`）：

```json
{"id": 1, "type": "fact", "question": "单户用于消费的个人信用贷款授信额度上限是多少？", "gold_chunk_ids": ["商业银行互联网贷款管理暂行办法.md#n"], "note": "第六条"}
{"id": 21, "type": "out_of_corpus", "question": "如何计算 LPR 利率？", "gold_chunk_ids": [], "note": "语料外，验证拒答"}
```

**题型分布**（20 条）：事实类 8、口径类 6、多跳/跨文档类 3、语料外类 3。

**指标**：recall@3 / recall@5 / recall@10（gold 块进前 k 名的题目占比）、MRR@10。

**对比配置**（一键跑完）：
1. vector（W2 现状基线）
2. bm25（纯稀疏）
3. hybrid（RRF 融合）
4. hybrid + rerank（融合后再精排）

**成本预估**：20 题 × 4 配置 ≈ 80 次 query embedding（约 2 万 token）+ rerank 调用，合计几分钱。

## 任务拆解（预算约 7h）

| # | 任务 | 预估 | 产出 |
|---|---|---|---|
| 1 | 我起草 20 条测试集（半自动定位 gold 块）+ 用户审核 | 0.5h（我）+ 0.5h（用户） | data/eval/retrieval_testset.jsonl |
| 2 | BM25 索引（jieba + rank-bm25 + 词典）+ 单测 | 1.5h | harness/bm25.py |
| 3 | RRF 融合 + HybridRetriever 重构 | 1.5h | fusion.py + retriever.py v2 |
| 4 | Reranker API 接入 | 1h | reranker.py |
| 5 | 查询改写任务 | 0.5h | rewrite.py + config |
| 6 | 评测 runner + 对比报告 | 1.5h | eval.py + --eval |
| 7 | 跑对比、调参（候选数/rrf_k 试 2 组）、README、commit | 1h | **检索效果对比记录** |

## 学习清单（开工前掌握，风控类比）

1. **BM25 / 稀疏检索**：TF=行为频次，IDF=区分度（类似 WOE：越稀有越有信息量），文档长度归一化防长文刷分。强项专有名词精确匹配，弱项同义改述。
2. **RRF 融合**：`Σ 1/(k+rank)` 只看排名不看分数——类比两个量纲不同的评分卡先转排名再合并。k=60 是惯例值。
3. **双塔 vs Cross-Encoder**：embedding 双塔（快、粗）vs reranker 拼接精排（准、慢）——类比评分卡初筛 vs 个案尽调，所以只对 top 候选精排。
4. **Recall@k / MRR**：类比 lift 表捕获率；MRR 衡量正确答案排名靠前程度。
5. **Vocab mismatch 与查询改写**：口语问法 vs 文档书面语的词表鸿沟；改写是治"检索没找到但资料里有"的主要手段之一。
6. **评测集原则**：gold 标到块级；必含语料外题目验证拒答；题型覆盖面决定评测说服力。

## 风险与应对

| 风险 | 应对 |
|---|---|
| jieba 切分专业词不佳（授信额度/FPD30 被切碎） | jieba 自定义词典 + 评测前后对比验证 |
| SiliconFlow rerank API 格式差异 | 薄适配层 + 单测；失败回退纯 hybrid |
| 测试集 gold 标注偏差 | 半自动定位 + 人工审核双保险；评测时打印未命中明细便于核对 |
| 时间不够 | 裁剪顺序：查询改写 → MRR（保 recall@k）→ 不砍混合检索与评测 |

## 验收 checklist

- [ ] `--build-index` 同时构建 FAISS + BM25 索引
- [ ] `retrieval.mode` 三种模式切换生效，`--rag` 行为对应变化
- [ ] rerank / rewrite 开关默认关，打开后行为正确且成本可控
- [ ] `--eval` 一键输出 4 配置对比表（recall@3/5/10、MRR@10），JSONL 与 markdown 落盘
- [ ] 测试集 20 条入库（含 3 条语料外），gold 标注经人工审核
- [ ] 对比记录能回答："混合检索相对纯向量提升多少？"（预期 hybrid ≥ vector，rerank 进一步提升；若不提升也是有效结论）
- [ ] README 更新 + 连续 commit

## W4 预告

Streamlit/Gradio 界面 + FastAPI 服务化 + Docker 基础 → 项目 1 完整版可演示（里程碑 1）。
