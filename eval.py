# ── eval.py：一键自动评估（14 天里最重要的一天）────────
# 之前判卷是"人眼看"（Day 4/5/6），今天把它变成"机器跑"。
# 评估对象：检索层（RAG 的地基）——给每段标准答案标好"在第几段"，
# 三种检索策略各跑一遍，算 Recall@3（前 3 段里有没有捞到答案所在的段）。
#
# 为什么不评估"回答"：
#   回答质量要靠 LLM 判或人工判，贵且慢。检索是一个查表问题，可以精确算分。
#   工业界的做法也是分两层：先评测检索（快、便宜、确定性），再评测生成。
#   所以这个脚本不花钱、不用 key、3 秒出结果 —— 随时改随时跑。

import re
import chromadb

# ============================================================
# 1. 评估集：题 + 标准答案所在的段号
#    "gold" = 答案在第几段（人工标好的标准答案，这是最值钱的资产）
# ============================================================
EVAL_SET = [
    # (问题, 答案所在段号列表)
    ("什么是相对挥发度？α 越接近 1 说明什么？", [2, 3, 5, 6, 8]),
    ("α 等于 1 时普通精馏为什么无法继续分离？", [2, 3, 5, 6, 8]),
    ("全回流操作有什么特点？最小理论板数用什么方程计算？", [3, 28, 30]),
    ("实际操作的回流比一般取最小回流比的多少倍？依据是什么？", [4, 20, 28]),
    ("恒沸精馏和萃取精馏有什么区别？", [2, 5, 6, 8]),
    ("乙醇-水体系可以用哪些方法分离？", [2, 5, 6, 8, 14]),
    ("板式塔常见的类型有哪些？工业上应用最广的是哪一种？", [7]),
    ("什么情况下应优先考虑特殊精馏或其他分离方法？", [2, 5, 6, 8]),
    ("化工生产中的「三传一反」指的是什么？", [10]),
    ("物料衡算的依据是什么？写出它的关系式。", [11]),
    ("绝对压强、表压强和真空度之间是什么关系？", [16]),
    ("确定离心泵安装高度时要注意哪几点？", [21, 23]),
    ("什么是间壁式换热器？举一个例子。", [26, 27]),
    ("稳态传热和非稳态传热的区别是什么？", [26]),
    ("最小回流比 Rmin 怎么求？苯-甲苯实例里 Rmin 和操作回流比分别是多少？", [28, 29]),
    ("塔径是怎么算出来的？为什么算完还要圆整和复核？", [32, 35]),
    ("塔板压降由哪几部分组成？如何防止液泛？", [35, 36]),
    ("塔板负荷性能图有哪五条边界线？设计点应该落在什么位置？", [38]),
    ("什么是操作弹性？一般要求多少？苯-甲苯实例的弹性是多少？", [39]),
    ("实际板数怎么从理论板数换算？全塔效率取多少合适？", [30, 31, 32]),
    # --- 自己新加的 5 道（2026-10-10）---
    ("雾沫夹带用什么指标衡量？要求控制在多少以内？", [37, 38]),
    ("淹塔校核要满足什么条件？Hd 怎么算？", [36]),
    ("浮阀数是怎么确定的？孔速用什么指标控制？", [34]),
    ("降液管停留时间有什么要求？实例里算出来是多少秒？", [33, 38]),
    ("板间距一般取多少？板上液层高度怎么定？", [33]),
]

# ============================================================
# 2. 语料与向量库
# ============================================================
with open("data/notes.txt", encoding="utf-8") as f:
    text = f.read()
segments = [s for s in text.split("\n\n") if s.strip()]

chroma = chromadb.Client()
try:
    chroma.delete_collection("eval_notes")
except Exception:
    pass
col = chroma.get_or_create_collection("eval_notes")
col.add(documents=segments, ids=[str(i) for i in range(len(segments))])

# ============================================================
# 3. 三种检索策略
# ============================================================
def retrieve_keyword_char(q, k=3):
    """Day 4 版：数单字重复次数"""
    def rel(seg):
        return sum(seg.count(ch) for ch in set(q) if ch in seg)
    ranked = sorted(enumerate(segments), key=lambda x: -rel(x[1]))[:k]
    return [i + 1 for i, _ in ranked]

def retrieve_keyword_bigram(q, k=3):
    """Day 6 版：数二字词组（'恒沸'、'精馏'当整体算）"""
    grams = [q[i:i + 2] for i in range(len(q) - 1)]
    def rel(seg):
        return sum(seg.count(g) for g in set(grams))
    ranked = sorted(enumerate(segments), key=lambda x: -rel(x[1]))[:k]
    return [i + 1 for i, _ in ranked]

def retrieve_semantic(q, k=3):
    """Day 5 版：chromadb 语义向量"""
    r = col.query(query_texts=[q], n_results=k)
    return [int(i) + 1 for i in r["ids"][0]]

def retrieve_hybrid(q, k=3, topn=5):
    """Day 6 版：二字词组 + 语义，RRF 融合"""
    score = {}
    for lst in (retrieve_keyword_bigram(q, topn), retrieve_semantic(q, topn)):
        for rank, idx in enumerate(lst):
            score[idx - 1] = score.get(idx - 1, 0) + 1 / (60 + rank + 1)
    top = sorted(score.items(), key=lambda x: -x[1])[:k]
    return [i + 1 for i, _ in top]

STRATEGIES = [
    ("关键词·单字 (Day4)", retrieve_keyword_char),
    ("语义·向量 (Day5)", retrieve_semantic),
    ("混合·RRF (Day6)", retrieve_hybrid),
]

# ============================================================
# 4. 跑评估：Recall@3
# ============================================================
def recall_at_k(hits, gold):
    """前 k 个结果里只要有一个落在标准答案段里，就算这题捞到了"""
    return 1 if any(h in gold for h in hits) else 0

report = []
print(f"评估集 {len(EVAL_SET)} 题 / 语料 {len(segments)} 段\n")
print(f"{'策略':<24}{'命中':<8}{'Recall@3':<10}")
print("-" * 42)

for name, fn in STRATEGIES:
    scores = []
    details = []
    for q, gold in EVAL_SET:
        hits = fn(q)
        ok = recall_at_k(hits, gold)
        scores.append(ok)
        details.append((q, gold, hits, ok))
    acc = sum(scores) / len(scores)
    report.append((name, sum(scores), acc, details))
    print(f"{name:<24}{sum(scores):<8}{acc:.1%}")

best_name, best_hit, best_acc, _ = max(report, key=lambda x: x[2])
print("-" * 42)
print(f"最优策略：{best_name}  Recall@3 = {best_acc:.1%} ({best_hit}/{len(EVAL_SET)})")

# 5. 落盘报告
with open("eval_report.md", "w", encoding="utf-8") as f:
    f.write("# 检索层评估报告（自动跑出来的，不是靠感觉）\n\n")
    f.write(f"评估集：{len(EVAL_SET)} 题 ｜ 语料：{len(segments)} 段 ｜ 指标：Recall@3\n\n")
    f.write("| 策略 | 命中 | Recall@3 |\n|---|---|---|\n")
    for name, hit, acc, _ in report:
        f.write(f"| {name} | {hit}/{len(EVAL_SET)} | {acc:.1%} |\n")
    f.write("\n## 逐题明细（最优策略）\n\n")
    for q, gold, hits, ok in report[-1][3]:
        mark = "OK" if ok else "MISS"
        f.write(f"- [{mark}] {q}\n  - 标准答案段：{gold}　实际检索：{hits}\n")

print("\n已生成 eval_report.md —— README 里的准确率数字就从这里来。")
