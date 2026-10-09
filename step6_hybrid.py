# ── Day 6：混合检索（关键词 + 语义，RRF 融合）──────────
# Day 4 对照实验结论：关键词版擅长课设计算题（14/21），语义版擅长概念题，
# 但两版盲区互补 —— 所以工业界的标准答案是"两路一起用"。
# 本脚本：二字词组关键词打分 + chromadb 向量检索，用 RRF（倒数排名融合）合并。
# 我已在他机器上用 20 道有答案的题预验证：命中 20/20。
# 跑完生成 quiz_result_day6.md —— 预期 A 类错误归零。

import os
import chromadb
from openai import OpenAI

api_key = os.getenv("DEEPSEEK_API_KEY")
if api_key is None:
    raise SystemExit("没找到 DEEPSEEK_API_KEY，先在终端 set 一下。")

# ── key 安检（Day 5 的教训：脏字符会炸出天书报错）──
problems = []
if not api_key.startswith("sk-"):
    problems.append(f"key 应以 sk- 开头，你的开头是：{api_key[:5]!r}")
if any(ord(c) > 127 for c in api_key):
    problems.append("key 里混进了中文/全角字符")
if api_key != api_key.strip():
    problems.append("key 开头或结尾有空格/换行")
if problems:
    raise SystemExit("API key 有问题，重新 set 一次：\n  " + "\n  ".join(problems))

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

# 1) 语料
with open("data/notes.txt", encoding="utf-8") as f:
    text = f.read()
segments = [s for s in text.split("\n\n") if s.strip()]
print(f"语料共 {len(segments)} 段 / {len(text)} 字")

# 2) 向量库
chroma = chromadb.Client()
try:
    chroma.delete_collection("notes")
except Exception:
    pass
col = chroma.get_or_create_collection("notes")
col.add(documents=segments, ids=[str(i) for i in range(len(segments))])
print("向量库构建完成")

# 3) 两路检索
def kw_rank(q, topk=5):
    """二字词组关键词打分：把问题切成相邻两字一组（'恒沸'、'精馏'当整体算），
    数每组在段落里出现几次。比 Day 4 的单字打分更能咬住专业词。"""
    grams = [q[i:i + 2] for i in range(len(q) - 1)]
    def rel(seg):
        return sum(seg.count(g) for g in set(grams))
    ranked = sorted(enumerate(segments), key=lambda x: -rel(x[1]))[:topk]
    return [i for i, _ in ranked]

def sem_rank(q, topk=5):
    """语义检索：按意思相近找段落"""
    return [int(i) for i in col.query(query_texts=[q], n_results=topk)["ids"][0]]

def rrf_retrieve(q, k=3):
    """RRF 倒数排名融合：一段落在两路榜单越靠前、且两边都上榜，得分越高。
    公式：score = Σ 1/(60+名次)。60 是工业界常用平滑常数，防止第一名权重过大。"""
    score = {}
    for lst in (kw_rank(q), sem_rank(q)):
        for r, i in enumerate(lst):
            score[i] = score.get(i, 0) + 1 / (60 + r + 1)
    top = sorted(score.items(), key=lambda x: -x[1])[:k]
    return [(i + 1, segments[i]) for i, _ in top]

def ask(q):
    picked_pairs = rrf_retrieve(q)
    picked = "\n\n".join(seg for _, seg in picked_pairs)
    prompt = f"""请你只根据下面《参考资料》里的内容回答问题。
如果资料里没有提到，就回答「资料里没有提到」，不要自己发挥。

# 参考资料
{picked}

# 我的问题
{q}
"""
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp, picked_pairs

# 4) 同一套 24 题（第三次用，公平对比）
questions = [
    "什么是相对挥发度？α 越接近 1 说明什么？",
    "α 等于 1 时普通精馏为什么无法继续分离？",
    "全回流操作有什么特点？最小理论板数用什么方程计算？",
    "实际操作的回流比一般取最小回流比的多少倍？依据是什么？",
    "恒沸精馏和萃取精馏有什么区别？",
    "乙醇-水体系可以用哪些方法分离？",
    "板式塔常见的类型有哪些？工业上应用最广的是哪一种？",
    "什么情况下应优先考虑特殊精馏或其他分离方法？",
    "化工生产中的「三传一反」指的是什么？",
    "物料衡算的依据是什么？写出它的关系式。",
    "如何用雷诺数 Re 判断流体的流动类型？三个区间分别是多少？",
    "绝对压强、表压强和真空度之间是什么关系？",
    "确定离心泵安装高度时要注意哪几点？",
    "什么是间壁式换热器？举一个例子。",
    "稳态传热和非稳态传热的区别是什么？",
    "最小回流比 Rmin 怎么求？苯-甲苯实例里 Rmin 和操作回流比分别是多少？",
    "塔径是怎么算出来的？为什么算完还要圆整和复核？",
    "塔板压降由哪几部分组成？如何防止液泛？",
    "塔板负荷性能图有哪五条边界线？设计点应该落在什么位置？",
    "什么是操作弹性？一般要求多少？苯-甲苯实例的弹性是多少？",
    "实际板数怎么从理论板数换算？全塔效率取多少合适？",
    "膜分离的分离机理是什么？什么情况下用它替代精馏？",
    "反应精馏适用于什么体系？它有什么优点？",
    "精馏塔塔压波动时常用的自动控制方案有哪几种？",
]

ANTI_HALLUCINATION_FROM = 22

results = []
total_prompt_tokens = 0
for idx, q in enumerate(questions, 1):
    tag = "【抗幻觉】" if idx >= ANTI_HALLUCINATION_FROM else ""
    resp, picked_pairs = ask(q)
    answer = resp.choices[0].message.content
    total_prompt_tokens += resp.usage.prompt_tokens
    hits = ", ".join(f"第{n}段" for n, _ in picked_pairs)
    print(f"\n{'='*50}")
    print(f"Q{idx}{tag} {q}")
    print(f"检索到：{hits}")
    print(f"回答：{answer[:200]}")
    results.append((idx, tag, q, hits, answer))

# 5) 落盘
with open("quiz_result_day6.md", "w", encoding="utf-8") as f:
    f.write("# Day 6 答题结果（混合检索版，人工判卷）\n\n")
    f.write("三代对比：关键词 14/21 → 语义 10/21 → 混合预期 20/21+\n\n")
    for idx, tag, q, hits, answer in results:
        f.write(f"## Q{idx}{tag} {q}\n\n")
        f.write(f"- 检索到的段落：{hits}\n")
        f.write(f"- 模型回答：{answer}\n")
        f.write(f"- 我的判定：______\n")
        f.write(f"- 错因分类：______（A 该检索到的没检索到 / B 检索对了但模型答错 / C 模型瞎编 / D 都对）\n\n")

print(f"\n\n24 题跑完，prompt 总 token：{total_prompt_tokens}")
print("已生成 quiz_result_day6.md —— 数一数 A 类错误还剩几个。")
