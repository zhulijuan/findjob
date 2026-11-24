"""一次性脚本：起草 20 条测试集，基于索引元数据半自动定位 gold chunk_id。"""
import json
from pathlib import Path

meta = json.loads(Path("data/index/credit_policy.meta.json").read_text(encoding="utf-8"))
chunks = meta["chunks"]


def find(phrase: str, source: str | None = None) -> list[str]:
    """返回包含指定短语的 chunk_id 列表。"""
    ids = []
    for c in chunks:
        if phrase in c["text"] and (source is None or c["source_file"] == source):
            ids.append(c["chunk_id"])
    return ids


# 每题用高区分度短语定位 gold 块；多跳题给多个短语
DRAFT = [
    # fact 8
    (1, "fact", "单户用于消费的个人信用贷款授信额度上限是多少？", ["不超过人民币20万元"], "监管第六条"),
    (2, "fact", "互联网贷款资金禁止用于哪些用途？", ["购房及偿还住房抵押贷款"], "监管第二十四条"),
    (3, "fact", "授信与首笔贷款发放时间间隔超过1个月有什么要求？", ["再评估"], "监管第二十六条"),
    (4, "fact", "对风险模型自动审批的补充机制是什么？", ["人工复核验证机制"], "监管第二十二条"),
    (5, "fact", "首次开展互联网贷款业务应何时提交书面报告？", ["产品上线后10个工作日内"], "监管第五十八条"),
    (6, "fact", "惠享贷的年化利率区间是多少？", ["18.00%"], "产品说明书-产品要素"),
    (7, "fact", "微企贷的授信额度如何测算？", ["税前年开票额的 15%"], "准入政策-额度与定价"),
    (8, "fact", "微企贷优先支持哪些行业？", ["优先支持行业"], "准入政策-行业准入"),
    # 口径 6
    (9, "口径", "FPD30 指标的口径是什么？", ["FPD30 指首期还款日"], "口径问答-指标口径类"),
    (10, "口径", "通过率的口径是什么？", ["审批通过的申请件数"], "口径问答-指标口径类"),
    (11, "口径", "渠道归属怎么判定？", ["渠道编码"], "口径问答-指标口径类"),
    (12, "口径", "收入负债比超标的客户能否人工特批？", ["联签特批"], "口径问答-政策执行类"),
    (13, "口径", "什么情况下触发额度冻结？", ["自动冻结额度"], "口径问答-风险监控类"),
    (14, "口径", "vintage 曲线如何用于准入策略评估？", ["vintage 曲线"], "口径问答-风险监控类"),
    # 多跳 3（跨文档）
    (15, "多跳", "惠享贷的最高授信额度是否符合监管对消费信用贷款的额度上限要求？",
     ["不超过人民币20万元", "200,000 元"], "监管第六条+产品要素"),
    (16, "多跳", "征信有当前逾期的客户能否申请惠享贷？监管对查询征信信息有什么要求？",
     ["无当前逾期", "查询借款人的征信信息"], "产品准入+监管第二十条"),
    (17, "多跳", "微企贷 FPD30 监控红线是多少？监管要求对风险模型做哪些日常监测？",
     ["0.8%", "风险模型日常监测"], "准入政策贷后+监管第四十条"),
    # 语料外 3
    (18, "语料外", "如何计算 LPR 利率？", [], "语料外，验证拒答"),
    (19, "语料外", "个人所得税专项附加扣除有哪些？", [], "语料外，验证拒答"),
    (20, "语料外", "如何用 Python 实现快速排序？", [], "语料外，验证拒答"),
]

records = []
for qid, qtype, question, phrases, note in DRAFT:
    gold = []
    for p in phrases:
        ids = find(p)
        if not ids:
            print(f"!! 未定位: 题{qid} 短语[{p}] —— 需人工检查")
        gold.extend(ids)
    gold = sorted(set(gold))
    records.append({
        "id": qid, "type": qtype, "question": question,
        "gold_chunk_ids": gold, "gold_key_phrases": phrases, "note": note,
    })

out = Path("data/eval/retrieval_testset.jsonl")
with out.open("w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

n_gold = sum(1 for r in records if r["gold_chunk_ids"])
print(f"测试集已生成: {out}（共 {len(records)} 条，含 gold {n_gold} 条）")
for r in records[:5]:
    print(f"  题{r['id']}: gold={r['gold_chunk_ids']}")
