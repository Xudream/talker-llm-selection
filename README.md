# talker-llm-selection

Talker 层 LLM 选型研究：**业界洞察 + 两把尺子（公开集 / 业务集）+ 可跑工装**。
部署边界：**只考虑云侧**；场景：**家庭音箱 + 手机助手**；模型分**开源轨（选型）**与**闭源 SOTA 轨（摸高）**。

## 结论速览

| 判断 | 依据 |
|---|---|
| 云侧开源 9B–27B 是主力档，122B-A10B 才谈得上接近闭源 | Qwen3.5-9B BFCL-V4 66.1 / τ² 79.1；27B、122B-A10B 均 Apache-2.0 |
| 闭源 SOTA 的领先在多轮工具与长上下文，不在单轮查询 | BFCL v4：Overall 77.47 / 单轮 88.58 / 多轮 73.76 |
| 查询必须走工具，任何模型都不能靠参数知识 | 小模型幻觉率 80–82% |
| 时延瓶颈是判停（200–800 ms）而非 LLM（200–1000 ms） | 级联链路区间汇总 |
| 家庭音箱与手机助手差异在声学与工具集，不在模型能力 | 两套 schema，混用即失真 |

## 两把尺子

| | 尺子1 公开集 | 尺子2 业务集 |
|---|---|---|
| 数据 | BFCL v4、τ²-bench、ACEBench、IFEval、HaluEval-zh/CMMLU | 自建：家庭音箱 455 + 手机助手 445（L1） |
| 回答 | 谁不配进下一轮 + 上限在哪 | 谁最适合业务 |
| 决策 | 淘汰 + 摸高 | 定选 |

## 文档

| 文件 | 内容 |
|---|---|
| `docs/01-业界方案洞察.md` | 开源轨候选池（9B/27B/122B-A10B/V4-Flash）、闭源 SOTA 摸高轨、场景差异、时延预算、榜单结构、五条启示 |
| `docs/02-Talker-LLM评测体系.md` | 尺子1 公开集映射与抽样、尺子2 业务集（两场景工具集、六轴、配额、判分、硬门槛、三态结论）、两把尺子的串联流程、成本 |
| `docs/03-评测执行手册.md` | 三步跑通、自检夹具、加候选、读报告、常见坑 |
| `docs/04-尺子1接入清单.md` | 五个公开集的仓库、抽样、官方 harness 命令、判分归属、门槛、结果汇总表模板、三个坑 |

## 工装

```bash
# 尺子2 题集：场景 home / phone / both，档位 L0 / L1 / L2
python3 tools/build_taskset.py --scene both --tier L1 --out tasksets/L1.jsonl

# 尺子1 题集：从 BFCL v4 抽样并转成统一格式（750 条）
python3 tools/fetch_public_sets.py --per-file 150 --out tasksets/public_bfcl_v4.jsonl

# 跑模型（OpenAI 兼容，原生 tool calling，流式测 TTFT）
python3 tools/run_model.py --taskset tasksets/L1.jsonl --model qwen-plus \
  --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --out predictions/qwen-plus__L1.jsonl

# 判分（六轴 + 硬门槛 + 加权 + 三态）
python3 tools/score.py --taskset tasksets/L1.jsonl \
  --pred predictions/qwen-plus__L1.jsonl --report reports/qwen-plus__L1.md
```

自检（不花 token）：`--backend dryrun` 用 gold 造 oracle，应得 100 分；仓库另带 `mockbad` 夹具，应被判"淘汰"。

## 目录

```
docs/        洞察 / 评测体系 / 执行手册
tools/       build_taskset.py  run_model.py  score.py  fetch_public_sets.py
tasksets/    home_L0/L1、phone_L0/L1、合并 L0/L1、public_bfcl_v4（750 条：本地 600 + 官方 harness 150）
predictions/ 模型输出 + 两个自检夹具
reports/     判分报告
```

## 与其它研究的关系

- 六轴与统计纪律继承 `duplex-voice/docs/L3模型选型评测方法.md`
- 时延门槛取自 `duplex-voice/docs/四层架构方案.md`
- 口语约束与 `duplex-voice/docs/垫词设计与评测.md` 同源
- 拒识（该不该说话）不在本仓库，见 `sidetalk-eval`
