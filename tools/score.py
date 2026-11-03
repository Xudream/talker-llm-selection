#!/usr/bin/env python3
"""Talker LLM 判分：六轴指标 + 硬门槛 + 加权排序 + 三态结论。

用法:
  python3 tools/score.py --taskset tasksets/L1.jsonl --pred predictions/qwen-plus__L1.jsonl \
      --report reports/qwen-plus__L1.md
"""
import argparse, json, math, re
from collections import Counter, defaultdict

BANNED_PREFIX = ("根据", "如下", "以下", "为您", "已为您")
MD_PAT = re.compile(r"[*#`|]|^\s*[-+]\s|^\s*\d+[.)]\s", re.M)
PAREN_PAT = re.compile(r"[（(][^）)]*[）)]")
EMOJI_PAT = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")

GATES = {"json_valid": 0.99, "over_call": 0.05, "speech_violate": 0.02,
         "fact_exact": 0.98, "empty_honest": 0.95}
WEIGHTS = {"A": 0.35, "B": 0.20, "C": 0.15, "D": 0.10, "E": 0.10, "F": 0.10}


def hw(p, n, z=1.96):
    """Wilson 区间半宽（p 接近 0/1 时比正态近似更稳）。"""
    if n == 0:
        return float("nan")
    denom = 1 + z * z / n
    half = z * math.sqrt(max(p * (1 - p), 0) / n + z * z / (4 * n * n)) / denom
    return half * 100


def norm_args(d):
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, bool):
            out[k] = v
        elif isinstance(v, (int, float)):
            out[k] = float(v)
        else:
            s = str(v).strip().lower()
            s = re.sub(r"\s+", "", s)
            out[k] = s
    return out


def match_call(pred, gold):
    """名称精确 + 参数逐槽位比较；返回 'exact' | 'partial' | 'wrong'"""
    if not pred or pred.get("name") != gold.get("name"):
        return "wrong"
    pa, ga = norm_args(pred.get("args")), norm_args(gold.get("args"))
    if pa == ga:
        return "exact"
    common = set(pa) & set(ga)
    return "partial" if common else "wrong"


def speech_violations(text, row):
    c = row.get("constraints") or {}
    v = []
    if not text:
        return ["empty"]
    first = re.split(r"[。！？!?]", text.strip())[0]
    if c.get("first_sentence_max_chars") and len(first) > c["first_sentence_max_chars"]:
        v.append("first_sentence_long")
    if c.get("max_chars") and len(text) > c["max_chars"]:
        v.append("too_long")
    if MD_PAT.search(text):
        v.append("markdown")
    if PAREN_PAT.search(text):
        v.append("parenthesis")
    if EMOJI_PAT.search(text):
        v.append("emoji")
    if text.strip().startswith(BANNED_PREFIX):
        v.append("banned_prefix")
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taskset", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--report", default=None)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.taskset, encoding="utf-8") if l.strip()]
    preds = {json.loads(l)["id"]: json.loads(l) for l in open(a.pred, encoding="utf-8") if l.strip()}
    missing = [r["id"] for r in rows if r["id"] not in preds]

    stat = defaultdict(lambda: {"n": 0, "exact": 0, "partial": 0, "wrong": 0})
    speech = {"n": 0, "bad": 0}
    fact = {"n": 0, "ok": 0}
    empty = {"n": 0, "ok": 0}
    overcall = {"n": 0, "bad": 0}
    ttft = []
    consistency = defaultdict(list)
    n_invalid_json = sum(1 for p in preds.values() if p.get("error"))
    n_official = 0
    total = len([r for r in rows if r["id"] in preds])

    for row in rows:
        p = preds.get(row["id"])
        if not p:
            continue
        gold = row.get("gold") or {}
        axis = row["axis"]
        if row.get("score_by") == "official":
            n_official += 1
            continue
        text = p.get("text") or ""
        calls = p.get("tool_calls") or []

        if p.get("ttft_ms"):
            ttft.append(p["ttft_ms"])
        if row.get("meta", {}).get("stability_only"):
            key = row["meta"].get("repeat_group", row["id"])
            consistency[key].append((calls[0]["name"] if calls else "") + json.dumps(
                norm_args(calls[0]["args"]) if calls else {}, sort_keys=True))

        # A4：不该调 / 该澄清 / 该确认
        if axis == "A4":
            overcall["n"] += 1
            if gold.get("abstain") or gold.get("clarify"):
                if calls:
                    overcall["bad"] += 1
            elif gold.get("confirm"):
                if calls:  # 写操作直接执行 = 未确认即调用
                    overcall["bad"] += 1
            continue

        # 工具类：A1/A2/A3/B/C/D/E/F
        stat[axis]["n"] += 1
        if gold.get("tool_calls"):
            got = calls[:len(gold["tool_calls"])]
            res = [match_call(g, x) for g, x in zip(gold["tool_calls"], got)]
            if len(got) < len(gold["tool_calls"]) or "wrong" in res:
                stat[axis]["wrong"] += 1
            elif "partial" in res:
                stat[axis]["partial"] += 1
            else:
                stat[axis]["exact"] += 1
        elif gold.get("tool_call"):
            r = match_call(calls[0] if calls else {}, gold["tool_call"])
            stat[axis][r] += 1

        if gold.get("speech_only") or axis == "B":
            speech["n"] += 1
            if speech_violations(text, row):
                speech["bad"] += 1
        if gold.get("must_contain") or gold.get("must_not_contain"):
            fact["n"] += 1
            ok = all(s in text for s in gold.get("must_contain", [])) and \
                not any(s in text for s in gold.get("must_not_contain", []))
            fact["ok"] += ok
            if row["family"] in ("empty", "timeout"):
                empty["n"] += 1
                empty["ok"] += any(k in text for k in ("没", "抱歉", "查不到", "不知道"))

    def acc(axis):
        s = stat[axis]
        return (s["exact"] + 0.5 * s["partial"]) / s["n"] * 100 if s["n"] else float("nan")

    tool_n = sum(stat[x]["n"] for x in ("A1", "A2", "A3"))
    tool_acc = sum(stat[x]["exact"] + 0.5 * stat[x]["partial"] for x in ("A1", "A2", "A3"))
    metrics = {
        "样本": total, "缺失预测": len(missing),
        "ToolAccLoose": tool_acc / tool_n * 100 if tool_n else float("nan"),
        "A1": acc("A1"), "A2": acc("A2"), "A3": acc("A3"),
        "D": acc("D"),
        "overCall": overcall["bad"] / overcall["n"] * 100 if overcall["n"] else 0.0,
        "speechViolate": speech["bad"] / speech["n"] * 100 if speech["n"] else float("nan"),
        "factExact": fact["ok"] / fact["n"] * 100 if fact["n"] else float("nan"),
        "emptyHonest": empty["ok"] / empty["n"] * 100 if empty["n"] else float("nan"),
        "jsonValid": (1 - n_invalid_json / total) * 100 if total else 0.0,
        "TTFT_P50": sorted(ttft)[len(ttft) // 2] if ttft else None,
        "TTFT_P95": sorted(ttft)[int(len(ttft) * 0.95)] if ttft else None,
    }
    cons = [len(set(v)) == 1 for v in consistency.values() if len(v) > 1]
    metrics["consistency"] = sum(cons) / len(cons) * 100 if cons else float("nan")

    gates = {
        "JSON 合法率 ≥99%": metrics["jsonValid"] >= GATES["json_valid"] * 100,
        "误调率 ≤5%": metrics["overCall"] <= GATES["over_call"] * 100,
        "口语违规 ≤2%": (metrics["speechViolate"] <= GATES["speech_violate"] * 100)
                        if metrics["speechViolate"] == metrics["speechViolate"] else True,
        "事实保真 ≥98%": (metrics["factExact"] >= GATES["fact_exact"] * 100)
                        if metrics["factExact"] == metrics["factExact"] else True,
        "空结果诚实 ≥95%": (metrics["emptyHonest"] >= GATES["empty_honest"] * 100)
                        if metrics["emptyHonest"] == metrics["emptyHonest"] else True,
    }

    def lin(v, lo, hi):
        if v != v:
            return None
        return max(0.0, min(1.0, (v - lo) / (hi - lo)))

    parts = {
        "工具闭环": lin(metrics["ToolAccLoose"], 60, 90),
        "口语约束": None if metrics["speechViolate"] != metrics["speechViolate"]
                    else 1 - min(1.0, metrics["speechViolate"] / 5),
        "事实保真": lin(metrics["factExact"], 90, 100),
        "多轮指代": lin(metrics["D"], 70, 95),
        "时延": None if not metrics["TTFT_P95"] else 1 - min(1.0, max(0.0, (metrics["TTFT_P95"] - 300) / 500)),
        "稳定性": lin(metrics.get("consistency", float("nan")), 90, 100),
    }
    key_map = {"工具闭环": "A", "口语约束": "B", "事实保真": "C", "多轮指代": "D", "时延": "E", "稳定性": "F"}
    num = den = 0.0
    for name, v in parts.items():
        if v is None:
            continue
        w = WEIGHTS[key_map[name]]
        num += w * v * 100
        den += w
    score = num / den if den else float("nan")

    if not all(gates.values()):
        verdict = "淘汰（硬门槛未过）"
    elif score >= 75:
        verdict = "保留（进 L2 复核）"
    else:
        verdict = "待定"

    lines = [f"# 判分报告：{a.pred}", "",
             f"- 样本 {metrics['样本']}｜缺失预测 {metrics['缺失预测']}｜公开集待官方判分 {n_official}",
             f"- **结论：{verdict}**｜加权得分 {score:.1f}" if score == score else f"- **结论：{verdict}**", "",
             "## 六轴指标", "",
             "| 指标 | 数值 | 95% CI |", "|---|---:|---|"]
    def rowc(name, v, n=None):
        ci = f"±{hw(v/100, n):.1f}pt" if n and v == v else "—"
        return f"| {name} | {v:.1f}% | {ci} |" if v == v else f"| {name} | — | — |"
    lines += [
        rowc("工具闭环 ToolAccLoose", metrics["ToolAccLoose"], tool_n),
        rowc("A1 单步选择", metrics["A1"], stat["A1"]["n"]),
        rowc("A2 参数填充", metrics["A2"], stat["A2"]["n"]),
        rowc("A3 两步闭环", metrics["A3"], stat["A3"]["n"]),
        rowc("D 多轮指代", metrics["D"], stat["D"]["n"]),
        rowc("口语违规率（越低越好）", metrics["speechViolate"], speech["n"]),
        rowc("事实保真", metrics["factExact"], fact["n"]),
        rowc("空结果诚实", metrics["emptyHonest"], empty["n"]),
        rowc("误调率（A4，越低越好）", metrics["overCall"], overcall["n"]),
        rowc("5 次一致率", metrics.get("consistency", float("nan")), None),
    ]
    p50 = metrics["TTFT_P50"] if metrics["TTFT_P50"] is not None else "—"
    p95 = metrics["TTFT_P95"] if metrics["TTFT_P95"] is not None else "—"
    lines += ["", "## 时延与格式", "",
              f"- TTFT P50 {p50} ms｜P95 {p95} ms",
              f"- JSON 合法率 {metrics['jsonValid']:.2f}%", "",
              "## 硬门槛", "", "| 门槛 | 结果 |", "|---|---|"]
    lines += [f"| {k} | {'✅' if v else '❌'} |" for k, v in gates.items()]

    out = "\n".join(lines) + "\n"
    if a.report:
        with open(a.report, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"写入 {a.report}")
    print(out)


if __name__ == "__main__":
    main()
