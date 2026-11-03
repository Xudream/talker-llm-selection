# talker-llm-selection

Talker 层 LLM 的选型研究：**业界方案洞察 + 评测集设计 + 可跑工装**。
范围：闲聊、单步查询、两步闭环、多轮指代——不含复杂推理（那是 Thinker）、不含拒识（那是 L1）。

## 结论速览

| 判断 | 依据 |
|---|---|
| 端侧 4B 够"能用"，9B 是"好用"分水岭 | Qwen3.5-9B BFCL-V4 66.1 / τ² 79.1（官方）；端侧三强 Qwen3-4B-2507 / Gemma 4 E4B / Phi-4-mini |
| **查询必须走工具**，不能靠参数知识 | 小模型在 AA-Omniscience 上幻觉率 80–82% |
| LLM 首 token 只占 200–1000 ms，判停 200–800 ms 往往更贵 | 级联链路实测区间汇总 |
| 口语化是独立能力，公开榜单完全不覆盖 | BFCL/τ² 只判工具正确性 |
| 选型必须按部署形态分档（端侧/边缘/云），不跨档比 | 三档量化与成本结构不同 |

## 文档

| 文件 | 内容 |
|---|---|
| `docs/01-业界方案洞察.md` | 三档候选池、时延预算、榜单能证明什么、业界五种模式、五条启示 |
| `docs/02-Talker-LLM选型评测集.md` | 六轴设计（A 工具 35% / B 口语 20% / C 保真 15% / D 多轮 10% / E 时延 10% / F 稳定 10%）、样本量、判分规则、决策规则、成本排期 |
| `docs/03-评测执行手册.md` | 三步跑通、自检、加候选、读报告、常见坑 |

## 工装

```bash
# 生成题集：L0 120 / L1 900 / L2 3000
python3 tools/build_taskset.py --tier L1 --out tasksets/L1.jsonl

# 跑一个模型（OpenAI 兼容，原生 tool calling，记录 TTFT）
python3 tools/run_model.py --taskset tasksets/L1.jsonl --model qwen-plus \
  --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --out predictions/qwen-plus__L1.jsonl

# 判分（六轴 + 硬门槛 + 加权 + 三态结论）
python3 tools/score.py --taskset tasksets/L1.jsonl \
  --pred predictions/qwen-plus__L1.jsonl --report reports/qwen-plus__L1.md
```

自检（不花 token）：`run_model.py --backend dryrun` 用 gold 造 oracle，应得 100 分；仓库里另带一个 `mockbad` 夹具，应被判"淘汰"。

## 与其它研究的关系

- 继承 `duplex-voice/docs/L3模型选型评测方法.md` 的六轴与统计纪律
- 时延预算取自 `duplex-voice/docs/四层架构方案.md`
- 判"该不该说话"不在本仓库，见 `sidetalk-eval`
- 口语约束与 `duplex-voice/docs/垫词设计与评测.md` 同源

## 目录

```
docs/        洞察 / 评测集设计 / 执行手册
tools/       build_taskset.py  run_model.py  score.py
tasksets/    题集（L0/L1/L2，可重建）
predictions/ 模型输出（含 oracle 与 mockbad 自检夹具）
reports/     判分报告
```
