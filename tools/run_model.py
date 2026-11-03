#!/usr/bin/env python3
"""跑一个候选模型：OpenAI 兼容接口 + 原生 tool calling，记录 TTFT 与时延。

用法:
  python3 tools/run_model.py --taskset tasksets/L1.jsonl --model qwen-plus \
    --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
    --out predictions/qwen-plus__L1.jsonl

  # 自检（不调模型，用 gold 造 oracle，验证判分链路）
  python3 tools/run_model.py --taskset tasksets/L0.jsonl --backend dryrun --out predictions/oracle__L0.jsonl
"""
import argparse, json, os, sys, time, urllib.error, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_taskset import TOOLS  # noqa: E402

TYPE_MAP = {"string": "string", "integer": "integer", "boolean": "boolean"}


def tool_specs(names):
    specs = []
    for n in names:
        if n not in TOOLS:
            continue
        t = TOOLS[n]
        props = {k: {"type": TYPE_MAP.get(v, "string")} for k, v in t["params"].items()}
        specs.append({"type": "function", "function": {
            "name": n, "description": t["desc"],
            "parameters": {"type": "object", "properties": props,
                           "required": list(t["params"].keys())}}})
    return specs


def build_messages(row):
    msgs = [{"role": "system", "content":
             "你是车载/家居语音助手。回复必须口语化：首句不超过 15 字，总长不超过 60 字，"
             "不要 markdown、不要列表、不要括号补充、不要 emoji。需要工具时调用工具，"
             "信息不足时向用户澄清，闲聊时不要调用任何工具。"}]
    for t in row["turns"]:
        role = t.get("role", "user")
        if role == "tool":
            msgs.append({"role": "user", "content": f"[工具返回] {t['text']}"})
        else:
            msgs.append({"role": role, "content": t.get("text", "")})
    return msgs


def call_once(base_url, model, key, msgs, tools, timeout=60):
    body = {"model": model, "messages": msgs, "temperature": 0, "stream": True,
            "max_tokens": 200}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}",
                                          "Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    text, calls = [], []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except Exception:
                continue
            ch = (d.get("choices") or [{}])[0]
            delta = ch.get("delta") or {}
            if delta.get("content"):
                if ttft is None:
                    ttft = int((time.time() - t0) * 1000)
                text.append(delta["content"])
            for tc in (delta.get("tool_calls") or []):
                i = tc.get("index", 0)
                while len(calls) <= i:
                    calls.append({"name": "", "arguments": ""})
                fn = tc.get("function") or {}
                if fn.get("name"):
                    calls[i]["name"] += fn["name"]
                if fn.get("arguments"):
                    calls[i]["arguments"] += fn["arguments"]
    total = int((time.time() - t0) * 1000)
    parsed = []
    for c in calls:
        try:
            args = json.loads(c["arguments"] or "{}")
        except Exception:
            args = {}
        parsed.append({"name": c["name"], "args": args})
    return {"text": "".join(text).strip(), "tool_calls": parsed,
            "ttft_ms": ttft if ttft is not None else total, "total_ms": total}


def dryrun(row):
    gold = row.get("gold") or {}
    calls = []
    if gold.get("tool_calls"):
        calls = gold["tool_calls"]
    elif gold.get("tool_call"):
        calls = [gold["tool_call"]]
    nums = gold.get("must_contain") or []
    text = ("好的，" + "，".join(nums) + "。") if nums else "好的。"
    if gold.get("abstain") or gold.get("clarify") or gold.get("confirm"):
        text = "请问你想放到哪个音箱？" if gold.get("clarify") else "好的。"
    return {"text": text, "tool_calls": calls, "ttft_ms": 0, "total_ms": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taskset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=["openai_compat", "dryrun"], default="openai_compat")
    ap.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    ap.add_argument("--model", default="qwen-plus")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--sleep", type=float, default=0.0)
    a = ap.parse_args()

    key = a.api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if a.backend != "dryrun" and not key:
        sys.exit("缺少 API key（--api-key 或 DASHSCOPE_API_KEY / OPENAI_API_KEY）")

    rows = [json.loads(l) for l in open(a.taskset, encoding="utf-8") if l.strip()]
    if a.limit:
        rows = rows[:a.limit]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    n_err = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows, 1):
            try:
                if a.backend == "dryrun":
                    res = dryrun(row)
                else:
                    res = call_once(a.base_url, a.model, key, build_messages(row),
                                    tool_specs(row["tools"]), a.timeout)
                    if a.sleep:
                        time.sleep(a.sleep)
            except urllib.error.HTTPError as e:
                n_err += 1
                res = {"text": "", "tool_calls": [], "ttft_ms": 0, "total_ms": 0,
                       "error": f"http{e.code}"}
            except Exception as e:
                n_err += 1
                res = {"text": "", "tool_calls": [], "ttft_ms": 0, "total_ms": 0,
                       "error": type(e).__name__}
            res.update({"id": row["id"], "model": a.model if a.backend != "dryrun" else "oracle",
                        "axis": row["axis"]})
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            if i % 50 == 0:
                print(f"  {i}/{len(rows)}", flush=True)
    print(f"{a.out}: {len(rows)} 条，错误 {n_err}")


if __name__ == "__main__":
    main()
