# ── Day 8：给小助手装"手"——工具调用（Function Calling）──────────
# 之前它只会"查资料、组织成话"；今天它会"判断该算什么、调用工具算、再用结果回答"。
# 关键区别：数字不再是模型编的，是 Python 真算出来的。
#
# 这一步对应 JD 原文：LangChain / AI Agent 项目实践

import json
import math
import os
from openai import OpenAI

api_key = os.getenv("DEEPSEEK_API_KEY")
if api_key is None:
    raise SystemExit("没找到 DEEPSEEK_API_KEY，先在终端 set 一下。")

problems = []
if not api_key.startswith("sk-"):
    problems.append("key 应以 sk- 开头")
if any(ord(c) > 127 for c in api_key):
    problems.append("key 里有中文/全角字符")
if problems:
    raise SystemExit("API key 有问题：" + "；".join(problems))

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

# ============================================================
# 1. 真实的计算工具（Python 函数 —— 结果是可复现的，不是模型编的）
# ============================================================
def fenske_min_stages(xD, xW, alpha):
    """芬斯克方程：全回流下的最小理论板数
    Nmin = ln[(xD/(1-xD)) * ((1-xW)/xW)] / ln(alpha)"""
    if not (0 < xW < xD < 1):
        return {"error": "需要满足 0 < xW < xD < 1"}
    if alpha <= 1:
        return {"error": "alpha 必须大于 1，否则普通精馏分不开"}
    nmin = math.log((xD / (1 - xD)) * ((1 - xW) / xW)) / math.log(alpha)
    return {
        "Nmin": round(nmin, 2),
        "说明": f"全回流、α={alpha} 下，xD={xD}、xW={xW} 需要的最少理论板数",
    }

def min_reflux_ratio(xD, xq, yq):
    """最小回流比：Rmin = (xD - yq) / (yq - xq)，(xq,yq) 是 q 线与平衡线交点"""
    if yq <= xq:
        return {"error": "需要 yq > xq，检查 q 线交点坐标"}
    rmin = (xD - yq) / (yq - xq)
    return {
        "Rmin": round(rmin, 3),
        "建议操作回流比R_1.5倍": round(1.5 * rmin, 2),
        "说明": f"q线交点({xq}, {yq})，塔顶组成 xD={xD}",
    }

def tower_diameter(gas_flow, u):
    """塔径：D = sqrt(4*qv / (pi*u))
    gas_flow 单位 m3/s，u 单位 m/s。
    圆整必须【向上】取标准塔径 —— 往下取会让实际气速超过设计值，诱发雾沫夹带甚至液泛。"""
    if gas_flow <= 0 or u <= 0:
        return {"error": "气相流量和设计气速都必须大于 0"}
    d = math.sqrt(4 * gas_flow / (math.pi * u))
    standards = [0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
    picked = next((s for s in standards if s >= d), round(d + 0.05, 1))
    # 圆整后复核真实气速：塔截面积变大，气速必然下降
    u_real = 4 * gas_flow / (math.pi * picked ** 2)
    return {
        "D_计算值": round(d, 3),
        "圆整后塔径": picked,
        "复核实际气速": round(u_real, 3),
        "说明": f"原取 u={u} m/s，圆整后实际气速降到 {round(u_real,3)} m/s（在安全范围内）",
    }

# 工具表：告诉模型"我有哪几只手、每只手需要什么参数"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fenske_min_stages",
            "description": "用芬斯克方程算全回流下的最小理论板数。输入塔顶组成xD、塔底组成xW、相对挥发度alpha。",
            "parameters": {
                "type": "object",
                "properties": {
                    "xD": {"type": "number", "description": "塔顶轻组分摩尔分率"},
                    "xW": {"type": "number", "description": "塔底轻组分摩尔分率"},
                    "alpha": {"type": "number", "description": "相对挥发度"},
                },
                "required": ["xD", "xW", "alpha"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "min_reflux_ratio",
            "description": "用 q 线与平衡线交点算最小回流比 Rmin。输入塔顶组成xD和交点坐标xq、yq。",
            "parameters": {
                "type": "object",
                "properties": {
                    "xD": {"type": "number", "description": "塔顶轻组分摩尔分率"},
                    "xq": {"type": "number", "description": "q线与平衡线交点的液相组成"},
                    "yq": {"type": "number", "description": "q线与平衡线交点的气相组成"},
                },
                "required": ["xD", "xq", "yq"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tower_diameter",
            "description": "由气相流量和设计空塔气速算塔径。输入 gas_flow(m3/s) 和 u(m/s)。",
            "parameters": {
                "type": "object",
                "properties": {
                    "gas_flow": {"type": "number", "description": "气相体积流量 m3/s"},
                    "u": {"type": "number", "description": "设计空塔气速 m/s"},
                },
                "required": ["gas_flow", "u"],
            },
        },
    },
]

AVAILABLE = {
    "fenske_min_stages": fenske_min_stages,
    "min_reflux_ratio": min_reflux_ratio,
    "tower_diameter": tower_diameter,
}

# ============================================================
# 2. Agent 循环：模型决定调工具 → 我们执行 → 把结果喂回去 → 模型给出最终答案
# ============================================================
def run_agent(question, max_rounds=4):
    messages = [
        {
            "role": "system",
            "content": (
                "你是化工工艺设计助手。用户问到需要定量计算的问题时，"
                "必须调用你手上的工具算出结果，不准自己编数字。"
                "工具返回什么就用什么，最后用中文把计算过程和结论讲清楚。"
                "回答用纯文字和简单表格，不要用 LaTeX 公式标记（如 $、\\frac），"
                "终端里显示不了那些符号。"
            ),
        },
        {"role": "user", "content": question},
    ]

    for round_no in range(1, max_rounds + 1):
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=TOOLS,
        )
        msg = resp.choices[0].message

        # 情况 A：模型说"我要调工具"
        if msg.tool_calls:
            messages.append(msg)
            for call in msg.tool_calls:
                fn = AVAILABLE[call.function.name]
                args = json.loads(call.function.arguments)
                result = fn(**args)
                print(f"  [第{round_no}轮] 调用工具 {call.function.name}({args})")
                print(f"           -> 返回 {result}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })
            continue

        # 情况 B：模型给最终答案
        print(f"  [第{round_no}轮] 模型给出最终答案")
        return msg.content

    return "（超出最大轮次，没拿到最终答案）"

# ============================================================
# 3. 试它
# ============================================================
QUESTIONS = [
    "苯-甲苯精馏，塔顶组成 xD=0.966，塔底 xW=0.01，相对挥发度 α=2.47。"
    "全回流下最少需要几块理论板？",

    "接上题，泡点进料，q 线与平衡线交点是 (0.45, 0.667)，塔顶 xD=0.966。"
    "最小回流比是多少？操作回流比取 1.5 倍的话是多少？",

    "气相流量 0.8 m3/s，设计空塔气速取 0.71 m/s，塔径该取多大？",
]

for q in QUESTIONS:
    print(f"\n{'='*56}")
    print(f"问：{q}")
    answer = run_agent(q)
    print(f"\n答：{answer}")

print("\n\n看看打印的'调用工具'那几行 —— 数字是 Python 算出来的，不是模型编的。")
print("这就是 Agent：模型负责判断'该算什么'，工具负责'算准'。")
