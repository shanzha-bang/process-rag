# ── Day 0 练习 2：跟大模型说上话 ──────────────────────
# 目标：终端里打印出模型的回答
#
# 前置条件：先去 https://platform.deepseek.com 注册、充 10 块钱、拿到 API key
# 然后在终端里执行（注意：这只是临时生效，关掉窗口就没了）：
#   Git Bash 里：  export DEEPSEEK_API_KEY="sk-你自己的key"
#   PowerShell 里：$env:DEEPSEEK_API_KEY="sk-你自己的key"

import os

# 思考题 A：为什么这里不直接写成 api_key = "sk-xxxx"？
# 提示：你想想，如果这个文件的代码被传到 GitHub 上，会发生什么？

api_key = os.getenv("DEEPSEEK_API_KEY")
if api_key is None:
    raise SystemExit(
        "没找到环境变量 DEEPSEEK_API_KEY。\n"
        "先在终端里 export / set 一下再运行本文件。"
    )

from openai import OpenAI

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",  # 思考题 B：为什么要有这一行？DeepSeek 和 OpenAI 是同一家公司吗？
)

# 思考题 C：把这段话换成你想问的任何一个化工问题。
#           建议换成你自己真正想不通的那个 —— 反正就几厘钱。
question = "什么是相对挥发度？用一句话给大一新生讲明白。"

resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": question}],
)

# 思考题 D：resp 是一个对象。为什么答案藏在 .choices[0].message.content 这么深的地方？
#           提示：choices 为什么是「列表」？什么情况下会有不止一个选择？
print(resp.choices[0].message.content)

# ── 扩展动手（选做，但做了你就比大部分人多走一步）：
# 在下面加一行 {"role": "system", "content": "你是一位化工厂的老师傅，说话实在，不爱讲术语"},
# 加在最前面，看看同一个问题的回答会发生什么变化。
# 这个「system」身份框的技术名字叫 system prompt —— 记住这个词，它会跟你一辈子。
