# SLEEPJEV

SLEEPJEV 是一个面向长时程 PSG 的研究型运行时语义决策框架：整晚信号只编码一次，形成可复用的 `SleepCache`，再通过运行时 `SleepQuery` 进行时间、睡眠阶段、体位和事件查询，使用稀疏检索与共享 option-conditioned scorer 输出当前选项集合上的概率。

```text
整晚 PSG -> shared overnight state -> event/stage postings
         -> runtime query -> sparse retrieval
         -> option-conditioned scorer -> runtime option probabilities
```

它体现的是 JEV-inspired 的接口思想：把“证据、问题、候选语义”作为一个明确的运行时决策契约，而不是把所有能力固定在一个分类头中。项目输出是证据/状态关系，不是疾病概率，也不是临床诊断。

## 快速开始

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[test,dev]"
python -m sleepjev demo
pytest -q
```

无私有数据的示例位于 `examples/`，可复现实验入口位于 `benchmarks/`，方法、数据、评估和发布约束位于 `docs/`。

## 当前状态

这是 alpha research prototype。当前首页结果是最终 two-seed 平均，但仍需要公开数据版本、subject-level split、checkpoint、硬件、线程策略和完整计时 manifest 才能完全复现；不能视为临床验证或普适性能结论。

当前最成立的结论是：Independent DL 的 stage macro-F1 仍最高（0.5948），而 SLEEPJEV 在 Q=512 的 hit@5 最高（0.6704）。在 warm serving 下，Q=512 的 SLEEPJEV 为 0.833 ms，Independent DL 为 8.983 ms；按 `QAI = sqrt(hit@5 * PU-recall@5)` 和 `QA-QPS = Q * QAI / serving_time_ms` 计算，质量调整后的吞吐优势约为 10.53x。这个数字不包含 EDF 读取和 overnight cache 构建，也不代表 SLEEPJEV 在所有质量指标上领先。

完整英文文档请看 [`README.md`](README.md)。

仓库风格参考、方法图、Logo 及 GPT-IMAGE 提示词分别位于 [`docs/repo-style-references.md`](docs/repo-style-references.md)、[`docs/assets/`](docs/assets/)；当前结果边界见 [`docs/results.md`](docs/results.md)。
