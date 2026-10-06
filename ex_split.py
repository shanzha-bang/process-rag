# ── Day 0 练习 1：让 Python 读一个文件 ──────────────────
# 目标：终端里打印出 data/notes.txt 的前 3 段，以及总共有多少段
# 卡住的时候：先看报错的【最后一行】，那一行才是真正的原因

with open("data/notes.txt", encoding="utf-8") as f:
    text = f.read()

# TODO 1：打印一下这篇文档一共有多少个字符
print("字符数：", len(text))
# TODO 2：这篇文档是用「空行」来分段的。
#         想一想，字符串有个 .split() 方法 —— 传给它 "\n\n" 会发生什么？
segments = text.split("\n\n")

# TODO 3：打印前 3 段，再打印总段数
for seg in segments[:3]:
    print("---")
    print(seg)

print("总段数：", len(segments))

# ── 做完之后回答这三个问题（不用写给我，自己想清楚）：
# 1. `with open(...) as f` 里的 with 是干什么的？删掉它会怎样？
# 2. `for seg in segments[:3]` 里的 [:3] 叫什么？改成 [:5] 会发生什么？
# 3. 为什么这里要写 encoding="utf-8"？删掉它在你的电脑上可能怎样？
