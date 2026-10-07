# ── Day 2：不用检索的最小问答闭环 ─────────────────────
# 目的：先把整篇资料「全塞进去」喂给模型，看看这样问答一次要花多少 token。
#       明天（Day 3）我们只塞最相关的 3 段，到时对比这个数字——那才是 RAG 的意义。

import os
from openai import OpenAI

api_key = os.getenv("DEEPSEEK_API_KEY")
if api_key is None:
    raise SystemExit("没找到环境变量 DEEPSEEK_API_KEY，先在终端 set / export 一下。")

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

# 1) 读入全部语料
with open("data/notes.txt", encoding="utf-8") as f:
    notes = f.read()

# 2) 把资料和问题拼成一段 prompt（这一步就是「全塞进去」）
question = "什么时候该用恒沸精馏，什么时候该用萃取精馏？"

prompt = f"""请你只根据下面《参考资料》里的内容回答问题。
如果资料里没有提到，就回答「资料里没有提到」，不要自己发挥。

# 参考资料
{notes}

# 我的问题
{question}
"""

# 3) 发出去
resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": prompt}],
)

answer = resp.choices[0].message.content
print("── 模型的回答 ──")
print(answer)

# 4) 关键：这次花了多少 token？把这个数字记下来，Day 3 要跟它对比
usage = resp.usage
print("\n── 本次消耗 ──")
print("prompt 区数量：", usage.prompt_tokens)
print("回答区数量：", usage.completion_tokens)
print("合计：", usage.total_tokens)

# ── 做完想一想（回答写在 README 里也行）：
# 1. 现在语料只有 8 段（667 字）。如果换成整本《化工原理》，这份 prompt 会变成多大？
# 2. 如果每次提问都重复塞一遍全文，一天问 100 次会发生什么？
# 3. 猜一猜：只塞「跟问题最相关的 3 段」之后，prompt_tokens 大概会变成几分之一？
