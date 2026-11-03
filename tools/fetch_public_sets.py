#!/usr/bin/env python3
"""拉取公开集（尺子1）子集并转成本仓库统一题集格式。

目前支持：
  bfcl_v4      Berkeley Function Calling Leaderboard v4（单轮 / 并行 / 幻觉 / 多轮）
  （ACE / τ² / IFEval / HaluEval 走官方 harness，见 docs/04-尺子1接入清单.md）

用法:
  python3 tools/fetch_public_sets.py --set bfcl_v4 --per-file 150 --out tasksets/public_bfcl_v4.jsonl
  HTTPS_PROXY=http://127.0.0.1:7890 python3 tools/fetch_public_sets.py --set bfcl_v4 --per-file 150 --out ...

设计说明：
  - 只做"抽样 + 转格式"，**不重写官方判分**；严格分数以官方 harness 为准
  - 转出来的 gold 只用于本地快速比对（func 名 + 参数），官方 AST 判分更严
  - 每条带 public_set / official_id，方便把官方分数按 id 回填
"""
import argparse, json, os, random, shutil, subprocess, tempfile, urllib.request

RAW = "https://raw.githubusercontent.com/ShishirPatil/gorilla/main/berkeley-function-call-leaderboard/bfcl_eval/data"
UA = {"User-Agent": "Mozilla/5.0"}

# 文件 → (轴, 说明)
BFCL_FILES = {
    "BFCL_v4_simple_python.json": ("A1", "单轮 simple（python 风格工具）"),
    "BFCL_v4_multiple.json": ("A1", "单轮 multiple"),
    "BFCL_v4_parallel.json": ("A2", "并行调用"),
    "BFCL_v4_irrelevance.json": ("A4", "不该调用（幻觉切面）"),
    "BFCL_v4_multi_turn_base.json": ("D", "多轮基础"),
}
ANSWER_DIR = "possible_answer"


def _load(text):
    """BFCL 的数据文件是 JSONL（每行一个对象），也可能遇到标准 JSON 数组。"""
    try:
        d = json.loads(text)
        return d if isinstance(d, list) else [d]
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def get_json(path):
    """优先用 curl（能走系统代理、对大文件更稳），没有 curl 再退回 urllib。"""
    url = f"{RAW}/{path}"
    if shutil.which("curl"):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            tmp = tf.name
        try:
            subprocess.run(["curl", "-sSL", "--fail", "--retry", "3", "--max-time", "180",
                            "-o", tmp, url], check=True, capture_output=True)
            return _load(open(tmp, encoding="utf-8").read())
        finally:
            os.unlink(tmp)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        return _load(r.read().decode("utf-8"))


def norm_turns(q):
    """BFCL question 可能是 [[{role,content}...]] 或 [{...}]，统一成 turn 列表。"""
    if not q:
        return []
    turns = q[0] if isinstance(q, list) and q and isinstance(q[0], list) else q
    out = []
    for t in turns:
        if not isinstance(t, dict):
            continue
        role = t.get("role", "user")
        content = t.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(c.get("text", c)) if isinstance(c, dict) else str(c) for c in content)
        out.append({"role": "tool" if role == "tool" else role, "text": str(content)})
    return out


def parse_gold(gt):
    """ground_truth 形如 [{"func": {"a": 1}}] 或 ["func(a=1)"]。
    含字符串形式/嵌套列表（多轮）时返回 None —— 一律交给官方 AST 判分。"""
    if not gt:
        return {"abstain": True}
    if isinstance(gt, str):
        gt = [gt]
    calls = []
    for item in gt:
        if isinstance(item, dict):
            for name, args in item.items():
                calls.append({"name": name, "args": args if isinstance(args, dict) else {}})
        elif isinstance(item, (str, list)):
            return None  # 字符串/嵌套列表形式交给官方 harness 判
    if not calls:
        return {"abstain": True}
    return {"tool_calls": calls} if len(calls) > 1 else {"tool_call": calls[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["bfcl_v4"], default="bfcl_v4")
    ap.add_argument("--per-file", type=int, default=150)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261103)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    rows, skipped = [], 0
    for fname, (axis, desc) in BFCL_FILES.items():
        try:
            items = get_json(fname)
        except Exception as e:
            print(f"[warn] {fname} 拉取失败：{type(e).__name__}")
            continue
        try:
            answers = {x["id"]: x.get("ground_truth") for x in get_json(f"{ANSWER_DIR}/{fname}")}
        except Exception:
            # irrelevance 没有答案文件（ground truth 天然为空），其它文件失败要提示
            answers = {}
            if "irrelevance" not in fname:
                print(f"  [warn] {fname} 的答案文件缺失，该文件条目只能走官方 harness")
        rng.shuffle(items)
        taken = 0
        for it in items:
            if taken >= a.per_file:
                break
            gold = parse_gold(answers.get(it["id"]))
            if "multi_turn" in fname:
                # 多轮只有官方 AST 能判；gold 留空，靠 official_answer 回填
                gold = {} if answers.get(it["id"]) else {"abstain": True}
            if gold is None:
                skipped += 1
                continue
            def _names(x):
                if isinstance(x, dict):
                    return [x["name"]] if x.get("name") else []
                if isinstance(x, list):
                    out = []
                    for y in x:
                        out += _names(y)
                    return out
                return []
            tools = _names(it.get("function") or [])
            raw_gt = answers.get(it["id"])
            score_by = "local" if "irrelevance" in fname else "official"
            rows.append({
                "id": f"bfcl-{os.path.basename(fname).replace('.json','')}-{it['id']}",
                "scene": "public", "axis": axis, "family": fname.replace(".json", ""),
                "lang": "en", "public_set": "bfcl_v4", "official_id": it["id"],
                "turns": norm_turns(it.get("question")),
                "tools": tools,
                "gold": {**gold, "official_answer": raw_gt},
                "score_by": score_by,
                "constraints": {},          # 公开集不适用口语约束
                "meta": {"source": "public", "desc": desc,
                         "schema": it.get("function") or [],
                         "involved_classes": it.get("involved_classes"),
                         # 多轮的工具有些以 class 形式给出（involved_classes），本地跑不了
                         "run_local": "multi_turn" not in fname},
            })
            taken += 1
        print(f"  {fname}: 取 {taken} 条")

    rng.shuffle(rows)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{a.out}: {len(rows)} 条（跳过无法解析 gold 的 {skipped} 条）")


if __name__ == "__main__":
    main()
