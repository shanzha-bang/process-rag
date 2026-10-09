# ── Day 4：换成你自己的语料，出 18 道题考它 ──────────
# 前 15 题答案都在 data/notes.txt 里；
# 后 3 题语料里根本没有 —— 专门测它会不会瞎编（抗幻觉测试）。
# 跑完会生成 quiz_result.md，你当裁判往里面填对错和错因。

import os
from openai import OpenAI

api_key = os.getenv("DEEPSEEK_API_KEY")
if api_key is None:
    raise SystemExit("没找到 DEEPSEEK_API_KEY，先在终端 set 一下。")

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

# 1) 读语料、切段
with open("data/notes.txt", encoding="utf-8") as f:
    text = f.read()
segments = [s for s in text.split("\n\n") if s.strip()]
print(f"语料共 {len(segments)} 段 / {len(text)} 字\n")

# 2) 检索打分（跟 Day 3 一样的朴素方法）
def relevance(q, seg):
    score = 0
    for ch in set(q):
        if ch in seg:
            score += seg.count(ch)
    return score

def retrieve(q, topk=3):
    scored = sorted(
        ((relevance(q, seg), i, seg) for i, seg in enumerate(segments)),
        reverse=True,
    )[:topk]
    return scored

def ask(q):
    scored = retrieve(q)
    picked = "\n\n".join(seg for _, _, seg in scored)
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
    return resp, scored

# 3) 题库：前 15 题有答案，后 3 题语料里没有
questions = [
    # --- 语料里有答案 ---
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
    # --- 课设计算题（苯-甲苯实例）---
    "最小回流比 Rmin 怎么求？苯-甲苯实例里 Rmin 和操作回流比分别是多少？",
    "塔径是怎么算出来的？为什么算完还要圆整和复核？",
    "塔板压降由哪几部分组成？如何防止液泛？",
    "塔板负荷性能图有哪五条边界线？设计点应该落在什么位置？",
    "什么是操作弹性？一般要求多少？苯-甲苯实例的弹性是多少？",
    "实际板数怎么从理论板数换算？全塔效率取多少合适？",
    # --- 语料里没有答案（抗幻觉测试）---
    "膜分离的分离机理是什么？什么情况下用它替代精馏？",
    "反应精馏适用于什么体系？它有什么优点？",
    "精馏塔塔压波动时常用的自动控制方案有哪几种？",
]

ANTI_HALLUCINATION_FROM = 21  # 从这个下标开始，语料里没有答案

results = []
for idx, q in enumerate(questions, 1):
    tag = "【抗幻觉】" if idx > ANTI_HALLUCINATION_FROM else ""
    resp, scored = ask(q)
    answer = resp.choices[0].message.content
    hits = ", ".join(f"第{i+1}段" for _, i, _ in scored)
    print(f"\n{'='*50}")
    print(f"Q{idx}{tag} {q}")
    print(f"检索到：{hits}")
    print(f"回答：{answer[:200]}")
    results.append((idx, tag, q, hits, answer))

# 4) 落盘，供你人工判卷
with open("quiz_result.md", "w", encoding="utf-8") as f:
    f.write("# Day 4 答题结果（人工判卷）\n\n")
    f.write("用法：在每道题下面写 【对/错】+ 错因（检索没找到 / 模型瞎编 / 都对但没说清）\n\n")
    for idx, tag, q, hits, answer in results:
        f.write(f"## Q{idx}{tag} {q}\n\n")
        f.write(f"- 检索到的段落：{hits}\n")
        f.write(f"- 模型回答：{answer}\n")
        f.write(f"- 我的判定：______\n")
        f.write(f"- 错因分类：______（A 该检索到的没检索到 / B 检索对了但模型答错 / C 模型瞎编 / D 都对）\n\n")

print("\n\n已生成 quiz_result.md —— 打开它，逐题填判定和错因。")
