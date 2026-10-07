# ── Day 3：加检索（关键词打分版）──────────────────
# 目的：只把「跟问题最相关的 3 段」塞给模型，对比 Day 2 的 477。
# 今天的检索不用任何 AI，就是最朴素的打分：
#   问题里的字，在某段里出现得越多，这段就越相关。

import os
from openai import OpenAI

api_key = os.getenv("DEEPSEEK_API_KEY")
if api_key is None:
    raise SystemExit("没找到 DEEPSEEK_API_KEY，先在终端 set 一下。")

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

# 1) 读语料并切段 —— 这套逻辑就是 Day 0.1 你亲手写的那个
with open("data/notes.txt", encoding="utf-8") as f:
    text = f.read()
segments = [s for s in text.split("\n\n") if s.strip()]
print(f"语料共 {len(segments)} 段")

# 2) 给每段打相关性分
def relevance(question, seg):
    score = 0
    for ch in set(question):          # 问题里出现过的每个不同的字
        if ch in seg:                 # 这一段里也有这个字
            score += seg.count(ch)    # 就加上它在这段里出现的次数
    return score

question = "什么时候该用恒沸精馏，什么时候该用萃取精馏？"

scored = []
for i, seg in enumerate(segments):
    scored.append((relevance(question, seg), i, seg))

# 3) 按分数从高到低排，取前 3 段
scored.sort(reverse=True)
top3 = scored[:3]

print("\n── 检索到的 3 段（按相关度排序）──")
for s, i, seg in top3:
    print(f"[第{i+1}段 | 相关度 {s}] {seg[:24]}...")

# 4) 只把这 3 段拼进 prompt
picked = "\n\n".join(seg for _, _, seg in top3)

prompt = f"""请你只根据下面《参考资料》里的内容回答问题。
如果资料里没有提到，就回答「资料里没有提到」，不要自己发挥。

# 参考资料
{picked}

# 我的问题
{question}
"""

resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": prompt}],
)

print("\n── 模型的回答 ──")
print(resp.choices[0].message.content)

# 5) 记账：跟 Day 2 的 477 对比
usage = resp.usage
print("\n── 本次消耗 ──")
print("prompt 区数量：", usage.prompt_tokens, "　（Day 2 全文版 = 477）")
print("回答区数量：", usage.completion_tokens)
print("合计：", usage.total_tokens)
print("对比 Day 2：降到原来的", round(usage.prompt_tokens / 477 * 100), "%")

# ── 跑通之后，自己动手玩这三件事（比看十遍都有用）：
# 1. 把 top3 改成 top1（[:1]），看回答质量变化
# 2. 把问题换成「板式塔有哪几种」，看检索到的是不是换了一批段落
# 3. 故意问一个语料里没有的：「膜分离和精馏哪个好？」——看它是不是老实说没提到
