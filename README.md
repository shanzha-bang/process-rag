# process-rag

> 化工工艺选型小助手 —— 一个从零搭起来的最小 RAG 项目

它回答一类问题：「乙醇-水体系该怎么分离？」「最小回流比取多少合适？」
回答不是模型编的，而是从我自己的化工笔记里检索出依据，再让它组织成话。

**这个项目是学出来的过程，不是成品。** 每一步的进度都在下面这张表里。

## 进度

- [x] Day 0　建仓库、搭环境
- [ ] Day 1　跑通第一次 API 调用
- [ ] Day 2　不用检索的最小问答闭环（先体验「为什么要检索」）
- [ ] Day 3　加向量检索，token 消耗降到 1/10
- [ ] Day 4　换成自己的化工语料，做裁判挑错
- [ ] Day 5　加防瞎编约束
- [ ] Day 6　建 20 题评估集 + 自动跑分
- [ ] Day 7　第一篇 README + push 到 GitHub
- [ ] Day 8-10　加工具调用（芬斯克方程 / 恩德伍德方程）
- [ ] Day 11-12　四段式复盘文档
- [ ] Day 13　60 秒 demo 录屏
- [ ] Day 14　登记 FDE 中国社区人才库

## 怎么跑起来

```bash
# 1. 进入项目目录
cd /d/workbody/process-rag

# 2. 激活虚拟环境（每次开新终端都要做这一步）
source .venv/Scripts/activate

# 3. 设置 API key（换成你自己的）
export DEEPSEEK_API_KEY="sk-你的key"

# 4. 跑
python hello_api.py
```

## 技术栈

Python · DeepSeek API · openai SDK · chromadb

（第一周故意不用 LangChain —— 先手写一遍完整链路，才知道它到底在替你干什么。）
