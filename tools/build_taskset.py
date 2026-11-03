#!/usr/bin/env python3
"""生成 Talker LLM 评测集（L0 冒烟 / L1 判定 / L2 全量）。

设计见 docs/02-Talker-LLM选型评测集.md：
  A 工具闭环 35% | B 口语约束 20% | C 事实保真 15% | D 多轮指代 10% | E 时延 10% | F 稳定 10%

用法:
  python3 tools/build_taskset.py --tier L0 --out tasksets/L0.jsonl
  python3 tools/build_taskset.py --tier L1 --out tasksets/L1.jsonl
"""
import argparse, hashlib, json, os, random

TOOLS = {
    "get_weather": {"desc": "查天气", "params": {"city": "string", "date": "string"}},
    "get_time": {"desc": "查时间/日期", "params": {"city": "string"}},
    "get_calendar": {"desc": "查日程", "params": {"date": "string"}},
    "create_reminder": {"desc": "建提醒", "params": {"content": "string", "time": "string"}},
    "play_music": {"desc": "放音乐", "params": {"song": "string"}},
    "set_volume": {"desc": "调音量", "params": {"level": "integer"}},
    "control_light": {"desc": "控制灯", "params": {"room": "string", "on": "boolean"}},
    "set_ac": {"desc": "控制空调", "params": {"mode": "string", "temperature": "integer"}},
    "navigate_to": {"desc": "导航", "params": {"destination": "string"}},
    "send_message": {"desc": "发消息", "params": {"contact": "string", "content": "string"}},
    "search_web": {"desc": "网页搜索", "params": {"query": "string"}},
    "get_exchange_rate": {"desc": "汇率", "params": {"from": "string", "to": "string"}},
}

CITIES = ["洛阳", "郑州", "北京", "上海", "杭州", "成都", "西安"]
DATES = ["今天", "明天", "后天", "这周六"]
SONGS = ["晴天", "起风了", "海阔天空", "孤勇者"]
ROOMS = ["客厅", "卧室", "书房"]
CONTACTS = ["妈妈", "老王", "张经理"]
NUMBERS = [("18", "23"), ("5", "12"), ("26", "31")]
CONSTRAINTS = {"first_sentence_max_chars": 15, "max_chars": 60, "no_markdown": True}


def _row(axis, family, turns, tools, gold, source="synthetic", adversarial=False, meta=None):
    h = hashlib.sha1(f"{axis}|{family}|{'|'.join(t['text'] for t in turns)}".encode()).hexdigest()[:10]
    return {
        "id": f"talker-{axis}-{family}-{h}",
        "axis": axis, "family": family, "lang": "zh",
        "turns": turns, "system_state": "idle",
        "tools": tools,
        "gold": gold,
        "constraints": dict(CONSTRAINTS),
        "meta": {"source": source, "adversarial": adversarial, **(meta or {})},
    }


def gen_A1(rng, n):
    """单步查询：天气/时间/汇率/百科"""
    out = []
    for i in range(n):
        k = i % 4
        if k == 0:
            c, d = rng.choice(CITIES), rng.choice(DATES)
            out.append(_row("A1", "weather",
                            [{"role": "user", "text": f"{d}{c}天气怎么样"}], ["get_weather"],
                            {"tool_call": {"name": "get_weather", "args": {"city": c, "date": d}}}))
        elif k == 1:
            c = rng.choice(CITIES)
            out.append(_row("A1", "time",
                            [{"role": "user", "text": f"{c}现在几点"}], ["get_time"],
                            {"tool_call": {"name": "get_time", "args": {"city": c}}}))
        elif k == 2:
            out.append(_row("A1", "exchange",
                            [{"role": "user", "text": "今天美元兑人民币多少"}], ["get_exchange_rate"],
                            {"tool_call": {"name": "get_exchange_rate", "args": {"from": "USD", "to": "CNY"}}}))
        else:
            out.append(_row("A1", "search",
                            [{"role": "user", "text": "珠穆朗玛峰有多高"}], ["search_web"],
                            {"tool_call": {"name": "search_web", "args": {"query": "珠穆朗玛峰 高度"}}}))
    return out


def gen_A2(rng, n):
    """参数填充：提醒、音乐、灯、空调、导航、消息"""
    out = []
    for i in range(n):
        k = i % 6
        if k == 0:
            out.append(_row("A2", "reminder",
                            [{"role": "user", "text": "明天早上七点叫我起床"}], ["create_reminder"],
                            {"tool_call": {"name": "create_reminder",
                                           "args": {"content": "起床", "time": "07:00"}}}))
        elif k == 1:
            s = rng.choice(SONGS)
            out.append(_row("A2", "music", [{"role": "user", "text": f"放一首{s}"}], ["play_music"],
                            {"tool_call": {"name": "play_music", "args": {"song": s}}}))
        elif k == 2:
            r = rng.choice(ROOMS)
            out.append(_row("A2", "light", [{"role": "user", "text": f"把{r}的灯关掉"}], ["control_light"],
                            {"tool_call": {"name": "control_light", "args": {"room": r, "on": False}}}))
        elif k == 3:
            t = rng.choice([22, 24, 26])
            out.append(_row("A2", "ac", [{"role": "user", "text": f"空调调到{t}度制冷"}], ["set_ac"],
                            {"tool_call": {"name": "set_ac", "args": {"mode": "cool", "temperature": t}}}))
        elif k == 4:
            out.append(_row("A2", "nav", [{"role": "user", "text": "导航到最近的加油站"}], ["navigate_to"],
                            {"tool_call": {"name": "navigate_to", "args": {"destination": "最近的加油站"}}}))
        else:
            c = rng.choice(CONTACTS)
            out.append(_row("A2", "message",
                            [{"role": "user", "text": f"给{c}发个微信说我晚点到"}], ["send_message"],
                            {"tool_call": {"name": "send_message",
                                           "args": {"contact": c, "content": "我晚点到"}}}))
    return out


def gen_A3(rng, n):
    """两步闭环：查 → 控"""
    out = []
    for i in range(n):
        c = rng.choice(CITIES)
        if i % 2 == 0:
            out.append(_row("A3", "weather_then_ac",
                            [{"role": "user", "text": f"看看{c}明天多少度，热的话把空调开到 24 度"}],
                            ["get_weather", "set_ac"],
                            {"tool_calls": [{"name": "get_weather", "args": {"city": c, "date": "明天"}},
                                            {"name": "set_ac", "args": {"mode": "cool", "temperature": 24}}],
                             "two_hop": True}))
        else:
            out.append(_row("A3", "calendar_then_remind",
                            [{"role": "user", "text": "查下我周六有没有会，有的话提前半小时提醒我"}],
                            ["get_calendar", "create_reminder"],
                            {"tool_calls": [{"name": "get_calendar", "args": {"date": "这周六"}},
                                            {"name": "create_reminder",
                                             "args": {"content": "开会", "time": "提前半小时"}}],
                             "two_hop": True}))
    return out


def gen_A4(rng, n):
    """不该调 / 该澄清 / 越权确认"""
    out = []
    chitchat = ["今天心情不太好", "你说话挺有意思的", "陪我聊会儿天吧", "哈哈你懂我"]
    vague = ["放首歌", "调一下温度", "提醒我一下", "给那个人发消息"]
    risky = ["把空调关了吧", "把门锁打开", "帮我把这条消息发出去"]
    for i in range(n):
        k = i % 3
        if k == 0:
            out.append(_row("A4", "chitchat",
                            [{"role": "user", "text": rng.choice(chitchat)}], list(TOOLS),
                            {"abstain": True}, adversarial=True))
        elif k == 1:
            out.append(_row("A4", "need_clarify",
                            [{"role": "user", "text": rng.choice(vague)}], list(TOOLS),
                            {"clarify": True}, adversarial=True))
        else:
            out.append(_row("A4", "confirm_write",
                            [{"role": "user", "text": rng.choice(risky)}], list(TOOLS),
                            {"confirm": True}, adversarial=True))
    return out


def gen_B(rng, n):
    """口语约束：给定工具结果，要求短语转述"""
    out = []
    for i in range(n):
        lo, hi = rng.choice(NUMBERS)
        c, d = rng.choice(CITIES), rng.choice(DATES)
        out.append(_row("B", "narrate",
                        [{"role": "user", "text": f"{d}{c}天气"},
                         {"role": "assistant", "text": "（查询中）"},
                         {"role": "tool", "text": f"weather_result: {d}{c} 多云 {lo}~{hi}℃ 东南风3级"}],
                        ["get_weather"],
                        {"tool_call": {"name": "get_weather", "args": {"city": c, "date": d}},
                         "must_contain": [lo, hi], "must_not_contain": ["晴", "以下", "根据"],
                         "speech_only": True}))
    return out


def gen_C(rng, n):
    """事实保真：数字/专名，含空结果与超时"""
    out = []
    for i in range(n):
        k = i % 3
        if k == 0:
            lo, hi = rng.choice(NUMBERS)
            out.append(_row("C", "numbers",
                            [{"role": "user", "text": "明天洛阳天气"},
                             {"role": "tool", "text": f"weather_result: 明天洛阳 小雨 {lo}~{hi}℃"}],
                            ["get_weather"],
                            {"tool_call": {"name": "get_weather", "args": {"city": "洛阳", "date": "明天"}},
                             "must_contain": [lo, hi]}))
        elif k == 1:
            out.append(_row("C", "empty",
                            [{"role": "user", "text": "查下我的日程"},
                             {"role": "tool", "text": "calendar_result: []"}],
                            ["get_calendar"],
                            {"tool_call": {"name": "get_calendar", "args": {"date": "今天"}},
                             "must_contain": ["没"], "must_not_contain": ["会议", "点"]},
                            adversarial=True))
        else:
            out.append(_row("C", "timeout",
                            [{"role": "user", "text": "现在汇率多少"},
                             {"role": "tool", "text": "error: timeout after 3000ms"}],
                            ["get_exchange_rate"],
                            {"tool_call": {"name": "get_exchange_rate", "args": {"from": "USD", "to": "CNY"}},
                             "must_contain": ["没查到"], "must_not_contain": ["7.", "汇率是"]},
                            adversarial=True))
    return out


def gen_D(rng, n):
    """多轮指代 / 纠错 / 槽位继承"""
    out = []
    for i in range(n):
        k = i % 3
        if k == 0:
            out.append(_row("D", "ellipsis",
                            [{"role": "user", "text": "北京明天天气"},
                             {"role": "assistant", "text": "北京明天多云，5 到 12 度。"},
                             {"role": "user", "text": "那后天呢"}],
                            ["get_weather"],
                            {"tool_call": {"name": "get_weather", "args": {"city": "北京", "date": "后天"}}}))
        elif k == 1:
            out.append(_row("D", "correction",
                            [{"role": "user", "text": "查上海天气"},
                             {"role": "assistant", "text": "上海今天阴，18 到 23 度。"},
                             {"role": "user", "text": "不是上海，是杭州"}],
                            ["get_weather"],
                            {"tool_call": {"name": "get_weather", "args": {"city": "杭州", "date": "今天"}}}))
        else:
            out.append(_row("D", "switch_topic",
                            [{"role": "user", "text": "明天洛阳天气"},
                             {"role": "assistant", "text": "明天洛阳有雨，18 到 23 度。"},
                             {"role": "user", "text": "放首晴天"}],
                            ["get_weather", "play_music"],
                            {"tool_call": {"name": "play_music", "args": {"song": "晴天"}}}))
    return out


def gen_E(n):
    """时延样本：固定长 prompt + 固定工具集，只测耗时"""
    out = []
    prompt = "请用一句话回答：明天洛阳天气怎么样。"
    for i in range(n):
        out.append(_row("E", "latency", [{"role": "user", "text": prompt}], ["get_weather"],
                        {"tool_call": {"name": "get_weather", "args": {"city": "洛阳", "date": "明天"}}},
                        meta={"latency_only": True, "repeat": i}))
    return out


def gen_F(n):
    """稳定性样本：同一输入重复，检查一致率"""
    out = []
    for i in range(n):
        out.append(_row("F", "consistency",
                        [{"role": "user", "text": "帮我定个明早七点的闹钟"}], ["create_reminder"],
                        {"tool_call": {"name": "create_reminder", "args": {"content": "起床", "time": "07:00"}}},
                        meta={"stability_only": True, "repeat_group": i // 5}))
    return out


TIER_QUOTA = {
    "L0": {"A1": 24, "A2": 16, "A3": 4, "A4": 8, "B": 24, "C": 16, "D": 16, "E": 6, "F": 6},
    "L1": {"A1": 180, "A2": 100, "A3": 30, "A4": 50, "B": 180, "C": 120, "D": 120, "E": 60, "F": 60},
    "L2": {"A1": 600, "A2": 340, "A3": 90, "A4": 170, "B": 600, "C": 400, "D": 400, "E": 200, "F": 200},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["L0", "L1", "L2"], default="L0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261103)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    q = TIER_QUOTA[a.tier]
    rows = []
    rows += gen_A1(rng, q["A1"])
    rows += gen_A2(rng, q["A2"])
    rows += gen_A3(rng, q["A3"])
    rows += gen_A4(rng, q["A4"])
    rows += gen_B(rng, q["B"])
    rows += gen_C(rng, q["C"])
    rows += gen_D(rng, q["D"])
    rows += gen_E(q["E"])
    rows += gen_F(q["F"])

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    blob = open(a.out, "rb").read()
    from collections import Counter
    print(f"{a.out}: {len(rows)} 条  sha256={hashlib.sha256(blob).hexdigest()[:16]}")
    print("  轴分布:", dict(sorted(Counter(r['axis'] for r in rows).items())))
    print("  意图族:", dict(sorted(Counter(r['family'] for r in rows).items())))


if __name__ == "__main__":
    main()
