#!/usr/bin/env python3
"""口语化检查器（尺子1 的 B' 轴 / 尺子2 的 B 轴补充）。

背景：公开集（BFCL/τ²/ACEBench/IFEval）**不评测"说得像不像人话"**，
VoiceBench 测的是"语音提问后的回答质量"，也不判口语风格。
所以口语化只能靠两类信号：
  1) 规则：书面特征密度（连接词、长定语、分号、书面词表…）与口语标记密度（语气词、短句…）
  2) 参照：与口语语料（LCCC / CDial-GPT）的 n-gram 似然比（需本地语料，可选）

用法:
  python3 tools/style_check.py --pred predictions/mockbad__L0.jsonl
  python3 tools/style_check.py --pred predictions/x.jsonl --lccc /path/to/lccc.txt
"""
import argparse, json, math, re
from collections import Counter, defaultdict

# 书面特征（出现即减分）
WRITTEN_MARKERS = [
    "首先", "其次", "再次", "最后", "综上", "总之", "因此", "此外", "另外",
    "根据", "基于", "针对", "关于", "通过", "以及", "并且", "从而", "使得",
    "可以", "能够", "进行", "实施", "采取", "具有", "具备", "属于", "存在",
    "方面", "情况", "问题在于", "需要注意", "建议您", "为您", "如下", "以下",
]
# 口语标记（出现即加分）
SPOKEN_MARKERS = [
    "吧", "呢", "啊", "呀", "啦", "哈", "嘛", "哦", "嗯", "诶",
    "咱", "挺", "特别", "有点", "马上", "刚好", "就差", "要不要",
    "好的", "行", "没事", "放心", "记得", "快了", "先",
]
WRITTEN_PUNCT = "；："
SENT_SPLIT = re.compile(r"[。！？!?]")


def analyze(text):
    t = (text or "").strip()
    if not t:
        return {"spoken": None, "written_hits": [], "spoken_hits": [], "avg_sent_len": 0, "punct": 0}
    sents = [s for s in SENT_SPLIT.split(t) if s.strip()]
    avg_len = sum(len(s) for s in sents) / len(sents) if sents else len(t)
    wh = [w for w in WRITTEN_MARKERS if w in t]
    sh = [w for w in SPOKEN_MARKERS if w in t]
    punct = sum(t.count(p) for p in WRITTEN_PUNCT)
    # 打分：口语标记 +，书面标记 -，长句 -，书面标点 -
    score = 0.0
    score += min(0.6, 0.2 * len(sh))
    score -= min(0.7, 0.2 * len(wh))
    if avg_len > 25:
        score -= 0.3
    elif avg_len <= 12:
        score += 0.2
    score -= min(0.3, 0.15 * punct)
    if len(t) > 80:
        score -= 0.2
    return {"spoken": max(-1.0, min(1.0, score)), "written_hits": wh, "spoken_hits": sh,
            "avg_sent_len": round(avg_len, 1), "punct": punct}


def ngram_lm(texts, n=2):
    lm = Counter()
    for t in texts:
        body = re.sub(r"\s+", "", t)
        s = "<s>" + body + "</s>"
        for i in range(len(s) - n + 1):
            lm[s[i:i + n]] += 1
    total = sum(lm.values()) or 1
    return {k: v / total for k, v in lm.items()}


def loglik(lm, text, floor=1e-6, n=2):
    s = "<s>" + re.sub(r"\s+", "", text) + "</s>"
    if len(s) < n:
        return math.log(floor)
    return sum(math.log(lm.get(s[i:i + n], floor)) for i in range(len(s) - n + 1)) / (len(s) - n + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--lccc", default=None, help="口语语料文本（每行一句），用于 n-gram 参照")
    ap.add_argument("--sample", type=int, default=8, help="打印多少条低分样例")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.pred, encoding="utf-8") if l.strip()]
    lm = None
    if a.lccc:
        lines = [l.strip() for l in open(a.lccc, encoding="utf-8") if l.strip()]
        lm = ngram_lm(lines[:200000])
        print(f"[LCCC] 用 {min(len(lines), 200000)} 句建 2-gram 参照")

    scored = []
    for r in rows:
        d = analyze(r.get("text"))
        if lm and r.get("text"):
            lip = loglik(lm, r["text"])
            # 与口语语料的平均似然比（log 空间），越大越像口语
            d["lccc_loglik"] = round(lip, 3)
        scored.append((r.get("id"), d, r.get("text", "")))

    vals = [d["spoken"] for _, d, _ in scored if d["spoken"] is not None]
    n = len(vals)
    avg = sum(vals) / n if n else float("nan")
    written = sum(1 for v in vals if v < 0)
    print(f"\n样本 {len(rows)}｜可评 {n}")
    print(f"口语均分 {avg:+.2f}（-1 最书面，+1 最口语）｜书面判定率 {written/n*100:.1f}%" if n else "无可评样本")
    by_axis = defaultdict(list)
    for _, d, _ in scored:
        if d["spoken"] is not None:
            by_axis["all"].append(d["spoken"])
    if lm:
        lips = [d["lccc_loglik"] for _, d, _ in scored if "lccc_loglik" in d]
        if lips:
            print(f"LCCC 似然均值 {sum(lips)/len(lips):.2f}（越高越接近口语语料）")

    print(f"\n最书面的 {a.sample} 条：")
    ranked = [x for x in sorted(scored, key=lambda x: (x[1]["spoken"] is None, x[1]["spoken"])) if x[1]["spoken"] is not None]
    for i, (rid, d, text) in enumerate(ranked[:a.sample], 1):
        print(f"  {i}. [{d['spoken']:+.2f}] {text[:60]}")
        if d["written_hits"]:
            print(f"     书面标记: {d['written_hits']}")


if __name__ == "__main__":
    main()
