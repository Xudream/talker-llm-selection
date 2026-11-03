#!/usr/bin/env python3
"""生成 Talker LLM 业务评测集（尺子2）：家庭音箱 / 手机助手。

设计见 docs/02-Talker-LLM评测体系.md §3：
  六轴 A(工具) / B(口语) / C(保真) / D(多轮) / E(时延) / F(稳定)
  两场景各自独立工具集，不混用 schema

用法:
  python3 tools/build_taskset.py --scene home  --tier L1 --out tasksets/home_L1.jsonl
  python3 tools/build_taskset.py --scene phone --tier L1 --out tasksets/phone_L1.jsonl
  python3 tools/build_taskset.py --scene both  --tier L1 --out tasksets/L1.jsonl
"""
import argparse, hashlib, json, os, random

# ---------------- 工具集 ----------------
TOOLS_HOME = {
    "control_light": {"desc": "控制灯", "params": {"room": "string", "on": "boolean", "level": "integer"}},
    "set_ac": {"desc": "控制空调", "params": {"mode": "string", "temperature": "integer"}},
    "control_curtain": {"desc": "控制窗帘", "params": {"room": "string", "action": "string"}},
    "control_vacuum": {"desc": "扫地机", "params": {"mode": "string", "room": "string"}},
    "play_music": {"desc": "放音乐", "params": {"song": "string"}},
    "create_reminder": {"desc": "建提醒", "params": {"content": "string", "time": "string", "who": "string"}},
    "get_weather": {"desc": "查天气", "params": {"city": "string", "date": "string"}},
    "get_recipe": {"desc": "查菜谱", "params": {"dish": "string"}},
    "family_announce": {"desc": "家庭广播", "params": {"content": "string"}},
}

TOOLS_PHONE = {
    "send_message": {"desc": "发消息", "params": {"app": "string", "contact": "string", "content": "string"}},
    "read_notifications": {"desc": "读通知", "params": {"limit": "integer"}},
    "summarize_notifications": {"desc": "通知摘要", "params": {"app": "string"}},
    "create_calendar": {"desc": "建日程", "params": {"title": "string", "time": "string"}},
    "navigate_to": {"desc": "导航", "params": {"destination": "string", "mode": "string"}},
    "get_route_eta": {"desc": "查路况耗时", "params": {"destination": "string"}},
    "play_music": {"desc": "放音乐", "params": {"song": "string"}},
    "call_contact": {"desc": "打电话", "params": {"name": "string"}},
    "get_weather": {"desc": "查天气", "params": {"city": "string", "date": "string"}},
}

TOOLS = {**TOOLS_HOME, **TOOLS_PHONE}

CITIES = ["洛阳", "郑州", "北京", "上海", "杭州", "成都"]
DATES = ["今天", "明天", "后天", "这周六"]
SONGS = ["晴天", "起风了", "海阔天空", "孤勇者"]
ROOMS = ["客厅", "卧室", "书房"]
CONTACTS = ["妈妈", "老王", "张经理"]
DISHES = ["番茄炒蛋", "红烧肉", "酸辣土豆丝"]
APPS = ["微信", "短信", "飞书"]
NUMBERS = [("18", "23"), ("5", "12"), ("26", "31")]
CONSTRAINTS = {"first_sentence_max_chars": 15, "max_chars": 60, "no_markdown": True}


def _row(scene, axis, family, turns, tools, gold, adversarial=False, meta=None):
    h = hashlib.sha1(f"{scene}|{axis}|{family}|{'|'.join(t['text'] for t in turns)}".encode()).hexdigest()[:10]
    return {"id": f"talker-{scene}-{axis}-{family}-{h}", "scene": scene, "axis": axis,
            "family": family, "lang": "zh", "turns": turns, "system_state": "idle",
            "tools": tools, "gold": gold, "constraints": dict(CONSTRAINTS),
            "meta": {"source": "synthetic", "adversarial": adversarial, **(meta or {})}}


def gen_home(rng, n, axis):
    """家庭音箱：灯/空调/窗帘/扫地机/音乐/提醒/天气/菜谱/广播"""
    out = []
    for i in range(n):
        k = i % 6
        if axis == "A1":
            if k == 0:
                c, d = rng.choice(CITIES), rng.choice(DATES)
                out.append(_row("home", "A1", "weather", [{"role": "user", "text": f"{d}{c}下雨吗"}],
                                ["get_weather"], {"tool_call": {"name": "get_weather", "args": {"city": c, "date": d}}}))
            elif k == 1:
                dish = rng.choice(DISHES)
                out.append(_row("home", "A1", "recipe", [{"role": "user", "text": f"{dish}怎么做"}],
                                ["get_recipe"], {"tool_call": {"name": "get_recipe", "args": {"dish": dish}}}))
            elif k == 2:
                s = rng.choice(SONGS)
                out.append(_row("home", "A1", "music", [{"role": "user", "text": f"放首{s}"}],
                                ["play_music"], {"tool_call": {"name": "play_music", "args": {"song": s}}}))
            elif k == 3:
                c = rng.choice(CITIES)
                out.append(_row("home", "A1", "weather_city", [{"role": "user", "text": f"{c}现在多少度"}],
                                ["get_weather"], {"tool_call": {"name": "get_weather", "args": {"city": c, "date": "今天"}}}))
            elif k == 4:
                out.append(_row("home", "A1", "announce", [{"role": "user", "text": "跟大家说一声吃饭了"}],
                                ["family_announce"], {"tool_call": {"name": "family_announce", "args": {"content": "吃饭了"}}}))
            else:
                out.append(_row("home", "A1", "recipe2", [{"role": "user", "text": "红烧肉需要炖多久"}],
                                ["get_recipe"], {"tool_call": {"name": "get_recipe", "args": {"dish": "红烧肉"}}}))
        elif axis == "A2":
            if k == 0:
                r = rng.choice(ROOMS)
                out.append(_row("home", "A2", "light", [{"role": "user", "text": f"把{r}灯调暗一点"}],
                                list(TOOLS_HOME), {"tool_call": {"name": "control_light",
                                                                 "args": {"room": r, "on": True, "level": 30}}}))
            elif k == 1:
                t = rng.choice([22, 24, 26])
                out.append(_row("home", "A2", "ac", [{"role": "user", "text": f"空调开到{t}度"}],
                                list(TOOLS_HOME), {"tool_call": {"name": "set_ac", "args": {"mode": "cool", "temperature": t}}}))
            elif k == 2:
                out.append(_row("home", "A2", "curtain", [{"role": "user", "text": "把窗帘拉上"}],
                                list(TOOLS_HOME), {"tool_call": {"name": "control_curtain",
                                                                 "args": {"room": "客厅", "action": "close"}}}))
            elif k == 3:
                out.append(_row("home", "A2", "vacuum", [{"role": "user", "text": "去把卧室扫一下"}],
                                list(TOOLS_HOME), {"tool_call": {"name": "control_vacuum",
                                                                 "args": {"mode": "clean", "room": "卧室"}}}))
            elif k == 4:
                out.append(_row("home", "A2", "reminder", [{"role": "user", "text": "明早七点叫我起床"}],
                                list(TOOLS_HOME), {"tool_call": {"name": "create_reminder",
                                                                 "args": {"content": "起床", "time": "07:00", "who": "我"}}}))
            else:
                out.append(_row("home", "A2", "light_off", [{"role": "user", "text": "把书房灯关了"}],
                                list(TOOLS_HOME), {"tool_call": {"name": "control_light",
                                                                 "args": {"room": "书房", "on": False, "level": 0}}}))
    return out


def gen_phone(rng, n, axis):
    """手机助手：消息/通知/日程/导航/音乐/电话/天气"""
    out = []
    for i in range(n):
        k = i % 6
        if axis == "A1":
            if k == 0:
                out.append(_row("phone", "A1", "notifications", [{"role": "user", "text": "我有什么新消息"}],
                                ["read_notifications"], {"tool_call": {"name": "read_notifications", "args": {"limit": 5}}}))
            elif k == 1:
                c = rng.choice(CITIES)
                out.append(_row("phone", "A1", "weather", [{"role": "user", "text": f"{c}今天要带伞吗"}],
                                ["get_weather"], {"tool_call": {"name": "get_weather", "args": {"city": c, "date": "今天"}}}))
            elif k == 2:
                out.append(_row("phone", "A1", "eta", [{"role": "user", "text": "到公司要多久"}],
                                ["get_route_eta"], {"tool_call": {"name": "get_route_eta", "args": {"destination": "公司"}}}))
            elif k == 3:
                s = rng.choice(SONGS)
                out.append(_row("phone", "A1", "music", [{"role": "user", "text": f"来首{s}"}],
                                ["play_music"], {"tool_call": {"name": "play_music", "args": {"song": s}}}))
            elif k == 4:
                app = rng.choice(APPS)
                out.append(_row("phone", "A1", "notify_summary", [{"role": "user", "text": f"{app}有什么重要的"}],
                                ["summarize_notifications"], {"tool_call": {"name": "summarize_notifications", "args": {"app": app}}}))
            else:
                out.append(_row("phone", "A1", "calendar_q", [{"role": "user", "text": "我下午有会吗"}],
                                ["read_notifications"], {"tool_call": {"name": "read_notifications", "args": {"limit": 10}}}))
        elif axis == "A2":
            if k == 0:
                c, app = rng.choice(CONTACTS), rng.choice(APPS)
                out.append(_row("phone", "A2", "message", [{"role": "user", "text": f"给{c}发{app}说我晚点到"}],
                                list(TOOLS_PHONE), {"tool_call": {"name": "send_message",
                                                                  "args": {"app": app, "contact": c, "content": "我晚点到"}}}))
            elif k == 1:
                out.append(_row("phone", "A2", "calendar", [{"role": "user", "text": "明天上午十点加个会"}],
                                list(TOOLS_PHONE), {"tool_call": {"name": "create_calendar",
                                                                  "args": {"title": "会议", "time": "明天10:00"}}}))
            elif k == 2:
                out.append(_row("phone", "A2", "nav", [{"role": "user", "text": "导航回家，走高速"}],
                                list(TOOLS_PHONE), {"tool_call": {"name": "navigate_to",
                                                                  "args": {"destination": "家", "mode": "driving"}}}))
            elif k == 3:
                c = rng.choice(CONTACTS)
                out.append(_row("phone", "A2", "call", [{"role": "user", "text": f"打给{c}"}],
                                list(TOOLS_PHONE), {"tool_call": {"name": "call_contact", "args": {"name": c}}}))
            elif k == 4:
                out.append(_row("phone", "A2", "eta2", [{"role": "user", "text": "去机场堵不堵"}],
                                list(TOOLS_PHONE), {"tool_call": {"name": "get_route_eta", "args": {"destination": "机场"}}}))
            else:
                out.append(_row("phone", "A2", "music_play", [{"role": "user", "text": "放我的跑步歌单"}],
                                list(TOOLS_PHONE), {"tool_call": {"name": "play_music", "args": {"song": "跑步歌单"}}}))
    return out


def gen_A3(rng, n, scene):
    out = []
    for i in range(n):
        if scene == "home":
            c = rng.choice(CITIES)
            out.append(_row("home", "A3", "weather_then_ac",
                            [{"role": "user", "text": f"看看{c}明天多少度，热的话把空调开到 24 度"}],
                            ["get_weather", "set_ac"],
                            {"tool_calls": [{"name": "get_weather", "args": {"city": c, "date": "明天"}},
                                            {"name": "set_ac", "args": {"mode": "cool", "temperature": 24}}],
                             "two_hop": True}))
        else:
            out.append(_row("phone", "A3", "eta_then_message",
                            [{"role": "user", "text": "看下我到公司要多久，超过半小时就发微信跟老王说我会晚到"}],
                            ["get_route_eta", "send_message"],
                            {"tool_calls": [{"name": "get_route_eta", "args": {"destination": "公司"}},
                                            {"name": "send_message",
                                             "args": {"app": "微信", "contact": "老王", "content": "我会晚到"}}],
                             "two_hop": True}))
    return out


def gen_A4(rng, n, scene):
    out = []
    if scene == "home":
        chit = ["今天有点累", "你说话真有意思", "陪我说说话", "孩子今天考试"]
        vague = ["把灯弄一下", "调一下温度", "放首歌", "提醒我一下"]
        risky = ["把空调关了", "把门打开", "全屋灯关掉"]
    else:
        chit = ["今天好烦", "你懂的挺多", "陪我聊聊", "刚下班"]
        vague = ["发个消息", "定个闹钟", "放首歌", "导航一下"]
        risky = ["把这条消息发出去", "清空通知", "给老王打电话"]
    for i in range(n):
        k = i % 3
        if k == 0:
            out.append(_row(scene, "A4", "chitchat", [{"role": "user", "text": rng.choice(chit)}],
                            list(TOOLS_HOME if scene == "home" else TOOLS_PHONE), {"abstain": True}, adversarial=True))
        elif k == 1:
            out.append(_row(scene, "A4", "need_clarify", [{"role": "user", "text": rng.choice(vague)}],
                            list(TOOLS_HOME if scene == "home" else TOOLS_PHONE), {"clarify": True}, adversarial=True))
        else:
            out.append(_row(scene, "A4", "confirm_write", [{"role": "user", "text": rng.choice(risky)}],
                            list(TOOLS_HOME if scene == "home" else TOOLS_PHONE), {"confirm": True}, adversarial=True))
    return out


def gen_B(rng, n, scene):
    out = []
    for i in range(n):
        lo, hi = rng.choice(NUMBERS)
        if scene == "home":
            c, d = rng.choice(CITIES), rng.choice(DATES)
            out.append(_row("home", "B", "narrate",
                            [{"role": "user", "text": f"{d}{c}天气"},
                             {"role": "tool", "text": f"weather_result: {d}{c} 多云 {lo}~{hi}℃ 东南风3级"}],
                            ["get_weather"],
                            {"tool_call": {"name": "get_weather", "args": {"city": c, "date": d}},
                             "must_contain": [lo, hi], "must_not_contain": ["晴", "以下", "根据"],
                             "speech_only": True}))
        else:
            out.append(_row("phone", "B", "narrate_eta",
                            [{"role": "user", "text": "到公司要多久"},
                             {"role": "tool", "text": f"route_result: 到公司 {lo} 公里，预计 {hi} 分钟"}],
                            ["get_route_eta"],
                            {"tool_call": {"name": "get_route_eta", "args": {"destination": "公司"}},
                             "must_contain": [lo, hi], "must_not_contain": ["以下", "根据"],
                             "speech_only": True}))
    return out


def gen_C(rng, n, scene):
    """事实保真：数字搬运 / 空结果 / 超时（按场景选工具）"""
    out = []
    for i in range(n):
        k = i % 3
        if k == 0:
            lo, hi = rng.choice(NUMBERS)
            if scene == "home":
                out.append(_row("home", "C", "numbers",
                                [{"role": "user", "text": "明天洛阳天气"},
                                 {"role": "tool", "text": f"weather_result: 明天洛阳 小雨 {lo}~{hi}℃"}],
                                ["get_weather"],
                                {"tool_call": {"name": "get_weather", "args": {"city": "洛阳", "date": "明天"}},
                                 "must_contain": [lo, hi]}))
            else:
                out.append(_row("phone", "C", "numbers",
                                [{"role": "user", "text": "到公司要多久"},
                                 {"role": "tool", "text": f"route_result: 到公司 {lo} 公里，预计 {hi} 分钟"}],
                                ["get_route_eta"],
                                {"tool_call": {"name": "get_route_eta", "args": {"destination": "公司"}},
                                 "must_contain": [lo, hi]}))
        elif k == 1:
            if scene == "home":
                out.append(_row("home", "C", "empty",
                                [{"role": "user", "text": "红烧肉怎么做"},
                                 {"role": "tool", "text": "recipe_result: {}"}],
                                ["get_recipe"],
                                {"tool_call": {"name": "get_recipe", "args": {"dish": "红烧肉"}},
                                 "must_contain": ["没"], "must_not_contain": ["分钟", "克", "步骤"]},
                                adversarial=True))
            else:
                out.append(_row("phone", "C", "empty",
                                [{"role": "user", "text": "我有什么新消息"},
                                 {"role": "tool", "text": "notifications_result: []"}],
                                ["read_notifications"],
                                {"tool_call": {"name": "read_notifications", "args": {"limit": 5}},
                                 "must_contain": ["没"], "must_not_contain": ["条", "收到"]},
                                adversarial=True))
        else:
            if scene == "home":
                out.append(_row("home", "C", "timeout",
                                [{"role": "user", "text": "明天洛阳天气"},
                                 {"role": "tool", "text": "error: timeout after 3000ms"}],
                                ["get_weather"],
                                {"tool_call": {"name": "get_weather", "args": {"city": "洛阳", "date": "明天"}},
                                 "must_contain": ["没查到"], "must_not_contain": ["度", "雨"]},
                                adversarial=True))
            else:
                out.append(_row("phone", "C", "timeout",
                                [{"role": "user", "text": "现在路况怎么样"},
                                 {"role": "tool", "text": "error: timeout after 3000ms"}],
                                ["get_route_eta"],
                                {"tool_call": {"name": "get_route_eta", "args": {"destination": "公司"}},
                                 "must_contain": ["没查到"], "must_not_contain": ["公里", "分钟"]},
                                adversarial=True))
    return out


def gen_D(rng, n, scene):
    out = []
    for i in range(n):
        k = i % 3
        if scene == "home":
            if k == 0:
                out.append(_row("home", "D", "ellipsis",
                                [{"role": "user", "text": "北京明天天气"},
                                 {"role": "assistant", "text": "北京明天多云，5 到 12 度。"},
                                 {"role": "user", "text": "那后天呢"}],
                                ["get_weather"],
                                {"tool_call": {"name": "get_weather", "args": {"city": "北京", "date": "后天"}}}))
            elif k == 1:
                out.append(_row("home", "D", "correction",
                                [{"role": "user", "text": "把客厅灯关了"},
                                 {"role": "assistant", "text": "客厅灯已经关了。"},
                                 {"role": "user", "text": "不是客厅，是卧室"}],
                                ["control_light"],
                                {"tool_call": {"name": "control_light",
                                               "args": {"room": "卧室", "on": False, "level": 0}}}))
            else:
                out.append(_row("home", "D", "switch_topic",
                                [{"role": "user", "text": "明天洛阳天气"},
                                 {"role": "assistant", "text": "明天洛阳有雨，18 到 23 度。"},
                                 {"role": "user", "text": "放首晴天"}],
                                ["get_weather", "play_music"],
                                {"tool_call": {"name": "play_music", "args": {"song": "晴天"}}}))
        else:
            if k == 0:
                out.append(_row("phone", "D", "ellipsis",
                                [{"role": "user", "text": "导航到公司"},
                                 {"role": "assistant", "text": "已为你规划到公司的路线，约 35 分钟。"},
                                 {"role": "user", "text": "走高速"}],
                                ["navigate_to"],
                                {"tool_call": {"name": "navigate_to",
                                               "args": {"destination": "公司", "mode": "driving"}}}))
            elif k == 1:
                out.append(_row("phone", "D", "correction",
                                [{"role": "user", "text": "给老王发微信说我到了"},
                                 {"role": "assistant", "text": "已发给老王。"},
                                 {"role": "user", "text": "发错了，发给张经理"}],
                                ["send_message"],
                                {"tool_call": {"name": "send_message",
                                               "args": {"app": "微信", "contact": "张经理", "content": "我到了"}}}))
            else:
                out.append(_row("phone", "D", "switch_topic",
                                [{"role": "user", "text": "到公司要多久"},
                                 {"role": "assistant", "text": "到公司 5 公里，预计 12 分钟。"},
                                 {"role": "user", "text": "放首起风了"}],
                                ["get_route_eta", "play_music"],
                                {"tool_call": {"name": "play_music", "args": {"song": "起风了"}}}))
    return out


def gen_E(n, scene):
    out = []
    prompt = "请用一句话回答：明天洛阳天气怎么样。" if scene == "home" else "请用一句话回答：到公司要多久。"
    tool = "get_weather" if scene == "home" else "get_route_eta"
    args = {"city": "洛阳", "date": "明天"} if scene == "home" else {"destination": "公司"}
    for i in range(n):
        out.append(_row(scene, "E", "latency", [{"role": "user", "text": prompt}], [tool],
                        {"tool_call": {"name": tool, "args": args}},
                        meta={"latency_only": True, "repeat": i}))
    return out


def gen_F(n, scene):
    """稳定性：同输入重复 5 次，看一致率"""
    out = []
    for i in range(n):
        if scene == "home":
            out.append(_row("home", "F", "consistency", [{"role": "user", "text": "明早七点叫我起床"}],
                            list(TOOLS_HOME),
                            {"tool_call": {"name": "create_reminder",
                                           "args": {"content": "起床", "time": "07:00", "who": "我"}}},
                            meta={"stability_only": True, "repeat_group": f"{scene}-{i // 5}"}))
        else:
            out.append(_row("phone", "F", "consistency", [{"role": "user", "text": "明天上午十点加个会"}],
                            list(TOOLS_PHONE),
                            {"tool_call": {"name": "create_calendar",
                                           "args": {"title": "会议", "time": "明天10:00"}}},
                            meta={"stability_only": True, "repeat_group": f"{scene}-{i // 5}"}))
    return out


TIER = {
    "L0": {"A1": 24, "A2": 14, "A3": 6, "A4": 8, "B": 24, "C": 14, "D": 16, "E": 8, "F": 6},
    "L1": {"A1": 90, "A2": 60, "A3": 25, "A4": 40, "B": 90, "C": 50, "D": 50, "E": 30, "F": 20},
    "L2": {"A1": 300, "A2": 200, "A3": 80, "A4": 120, "B": 300, "C": 160, "D": 160, "E": 100, "F": 80},
}
# 手机助手在 B 轴与 D 轴上配比略调（见文档 §3.2）
TIER_PHONE_OVERRIDE = {"L1": {"B": 70, "D": 60}, "L2": {"B": 240, "D": 200}}


def build(scene, tier, rng):
    q = dict(TIER[tier])
    if scene == "phone":
        q.update(TIER_PHONE_OVERRIDE.get(tier, {}))
    rows = []
    gen = gen_home if scene == "home" else gen_phone
    rows += gen(rng, q["A1"], "A1")
    rows += gen(rng, q["A2"], "A2")
    rows += gen_A3(rng, q["A3"], scene)
    rows += gen_A4(rng, q["A4"], scene)
    rows += gen_B(rng, q["B"], scene)
    rows += gen_C(rng, q["C"], scene)
    rows += gen_D(rng, q["D"], scene)
    rows += gen_E(q["E"], scene)
    rows += gen_F(q["F"], scene)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", choices=["home", "phone", "both"], default="both")
    ap.add_argument("--tier", choices=["L0", "L1", "L2"], default="L0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261103)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    scenes = ["home", "phone"] if a.scene == "both" else [a.scene]
    rows = []
    for s in scenes:
        rows += build(s, a.tier, rng)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    blob = open(a.out, "rb").read()
    from collections import Counter
    print(f"{a.out}: {len(rows)} 条  sha256={hashlib.sha256(blob).hexdigest()[:16]}")
    print("  场景:", dict(sorted(Counter(r['scene'] for r in rows).items())))
    print("  轴:", dict(sorted(Counter(r['axis'] for r in rows).items())))
    print("  族:", dict(sorted(Counter(r['family'] for r in rows).items())))


if __name__ == "__main__":
    main()
