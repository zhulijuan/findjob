# W2 计划：信贷政策知识库 v1（RAG）

> 依据《AI转型三个月行动计划》W2 与 [docs/harness-plan.md](harness-plan.md) 能力地图制定
> 技术选型于 2026-11-23 拍板：**SiliconFlow BGE-M3 API / FAISS / rerank 顺延 W3**

## 目标与验收标准

- 可检索问答脚本版：`--rag "问题"` 基于语料回答，答案附来源（文件名/条款）
- 索引可重建：`--build-index` 一条命令从语料全量重建索引
- 全部参数配置化，代码零业务参数（延续 harness 铁律）
- 语料合规：只用公开监管文件 + 自写合成文档，无银行真实数据
- GitHub 连续 commit、README 更新

## 技术选型（已确认）

| 项 | 选择 | 理由 |
|---|---|---|
| Embedding | SiliconFlow API，模型 `BAAI/bge-m3` | OpenAI 兼容协议，复用现有客户端模式；注册送额度近乎免费；BGE-M3 与计划字面对齐，W11 评测叙事可复用 |
| 向量库 | FAISS（faiss-cpu，IndexFlatIP + 向量归一化 ≡ 余弦相似度） | Intel Mac 稳定轻量；建 VectorStore 抽象接口，后续可加 Milvus 实现作对比叙事 |
| Rerank | 顺延 W3 | 与混合检索、召回率评测同批落地，有/无 rerank 的对比本身就是评测素材 |

## 架构与数据流

```
建索引：data/corpus/*.md|txt → chunker(条款感知分块) → embedding(批量) → FaissStore 持久化
问答：  --rag 问题 → embedding → FaissStore.search(top_k) → context 拼装(带来源) → LLMClient 生成 → 答案 + 引用
```

## 模块设计

| 模块 | 职责 | 关键点 |
|---|---|---|
| `harness/embedding.py` | EmbeddingClient：`embed(texts) -> (n, dim)` | 批量调用、单次重试、空文本校验；调用同样记入 usage.jsonl（meta.task=embedding） |
| `harness/chunker.py` | 中文条款感知分块 | 优先按「第X章/第X条/一、」层级切，超长按句/换行递归；每块带 metadata：source_file / heading / chunk_id；纯函数可单测 |
| `harness/vectorstore.py` | 抽象接口 + FaissStore 实现 | 接口：add / search / save / load；向量存 faiss，文本+元数据存同名 .meta.json |
| `harness/retriever.py` | 检索管线 | 查询向量化 → search(top_k) → 带分数的块列表 |
| `harness/rag.py` | context 拼装与 rag 任务 | 格式：`【来源1｜文件名｜第X条】正文…`；v1 简单引用，W3 做严格段落级溯源 |
| `main.py` | `--build-index` / `--rag "问题"` | --rag 支持 --stream |

## config.yaml 新增段（草案）

```yaml
embedding:
  base_url: https://api.siliconflow.cn/v1
  api_key_env: SILICONFLOW_API_KEY
  model: BAAI/bge-m3
  batch_size: 16
  timeout: 60

vectorstore:
  type: faiss
  index_path: data/index/credit_policy

rag:
  corpus_dir: data/corpus
  chunk_size: 400
  chunk_overlap: 50
  top_k: 5

tasks:
  rag:
    description: 基于信贷知识库问答（配合 --rag 使用）
    system_prompt: 仅依据提供的资料回答信贷政策问题；资料中没有的内容明确说"资料中未提及"，不得编造；回答末尾列出引用来源。
    user_prompt: |
      资料：
      {{ context }}

      问题：{{ question }}
```

`.gitignore` 追加 `data/index/`（索引可重建，不入库）；`data/corpus/`（公开监管文件+合成文档）入库。

## 数据准备清单

1. 公开监管文件（金融监管总局官网下载 PDF → pdfplumber 抽取 → 清洗为 md/txt）：
   - 《商业银行互联网贷款管理暂行办法》（必做）
   - 《个人贷款管理办法》（可选，时间不够则砍）
2. 合成文档（自写，虚构机构名"某某银行"）：
   - 个人消费贷产品说明书（额度/利率/期限/准入条件）
   - 小微企业经营贷准入政策
   - 风控政策口径问答（Q&A 形式，方便演示）
3. `.env` 新增 `SILICONFLOW_API_KEY`（siliconflow.cn 注册获取，行动项）

## 任务拆解（预算约 6h）

| # | 任务 | 预估 | 产出 |
|---|---|---|---|
| 1 | 注册 SiliconFlow + key 进 .env；准备语料（下载、清洗、合成文档） | 2h | data/corpus/ 约 5 份文档 |
| 2 | chunker + 最小验证 | 1h | harness/chunker.py |
| 3 | embedding client + FaissStore + --build-index | 1.5h | 一条命令可重建索引 |
| 4 | retriever + rag 任务 + main.py --rag | 1h | 端到端问答 |
| 5 | 联调调参（chunk_size / top_k 试 2~3 组）、README、commit | 0.5~1h | 可演示 v1 |

## 学习清单（面试可讲的概念点）

- **embedding 直觉**：类比风控建模——把客户画像压成一个低维向量，距离近≈相似；文本同理，语义近的文本向量距离近
- **为什么 BGE-M3**：中文效果好、支持 8192 长文本、稠密+稀疏多向量能力（W3 混合检索的伏笔）
- **chunk size 权衡**：太大→噪声多、粒度糊；太小→语义断裂；条款类文档按「条」切天然对齐语义单元（类比大表按主键分区）
- **相似度度量**：向量归一化后内积 = 余弦相似度；为什么检索用 IP 不用 L2
- **RAG vs 长上下文直塞 vs 微调**：成本/时效/可控性对比（面试八股高频）

## 风险与应对

| 风险 | 应对 |
|---|---|
| SiliconFlow 注册或额度问题 | 备选 DashScope text-embedding-v3，改 embedding 配置三行即可切换 |
| 监管 PDF 解析质量差 | 换官网 HTML 版或人工整理；公开文件无合规顾虑 |
| faiss-cpu 在 Intel Mac 安装异常 | 官方 wheel 支持 macos x86_64；兜底退回 numpy 暴力检索（语料 <1 万块性能足够），接口不变 |
| 时间不够 | 裁剪顺序：砍第二份监管文件 → 砍合成文档数量（保 2 份）→ 不砍问答链路 |

## 验收 checklist

- [ ] `--build-index` 全量重建成功，索引与元数据文件生成
- [ ] `--rag` 回答附引用来源；问语料外内容时明确说"资料中未提及"
- [ ] usage.jsonl 同时记录 embedding 与 LLM 调用
- [ ] 新增能力全部走 YAML 配置，代码零业务参数
- [ ] README 更新 + 连续 commit

## W3 预告

混合检索（BM25 + 向量）、bge-reranker 开关、20 条标注测试集 + recall@k 落盘、引用溯源严格化（段落级出处）。
