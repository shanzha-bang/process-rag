# ── find_seg.py：找关键词在第几段 ──────────
# 用法：
#   .\.venv\Scripts\python.exe find_seg.py 恒沸
#   .\.venv\Scripts\python.exe find_seg.py 液泛
# 会打印：每个包含该词的段号 + 前 40 字预览。
# 给 EVAL_SET 标注"答案在第几段"就用它，别再肉眼数了。

import sys

with open("data/notes.txt", encoding="utf-8") as f:
    text = f.read()
segments = [s for s in text.split("\n\n") if s.strip()]

if len(sys.argv) < 2:
    print("用法：python find_seg.py 关键词")
    raise SystemExit(1)

kw = sys.argv[1]
hits = [(i + 1, seg) for i, seg in enumerate(segments) if kw in seg]

if not hits:
    print(f"语料里没有「{kw}」——这词不能出题（或者换个说法搜搜）")
else:
    print(f"「{kw}」出现在 {len(hits)} 个段：")
    for n, seg in hits:
        print(f"  第{n}段  {seg[:40].replace(chr(10), ' ')}")
