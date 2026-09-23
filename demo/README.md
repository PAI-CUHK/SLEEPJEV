# SLEEPJEV Runtime Query Console

这是 SLEEPJEV 的交互式产品 Demo，不是临床应用。当前版本已经接入本地真实 checkpoint 和 SHHS cached feature slice，把项目的核心 JEV-inspired 契约展示成一个可操作的工作面：

```text
shared overnight state + runtime question + runtime options
    -> evidence readout
    -> probabilities over the supplied options
```

## Run locally

在本目录的上一级运行：

```powershell
python demo/server.py
```

然后打开 `http://127.0.0.1:8765/`。

当前默认加载：

- 权重：`artifacts/formal_small_smoke/sleepjev_checkpoint.pt`
- 数据：`artifacts/experiment1_shhs_full/cache/shhs1-200001.npz`
- 任务：真实 SHHS 特征上的 Choice / Noul / Score 多任务 runtime fan-out
- 并行监控：一次 batch 同时服务 4 个不同时间窗、20 个 typed decisions
- 自动 replay：约 650 ms 切换一轮窗口与候选集；四个窗口共享同一次 overnight encoding
- Arena 视图：每个窗口同时显示 Choice、Noul、Score、Score confidence/action gate，以及 leading probability 的实时 delta
- 顶部 workload summary：fan-out、batch latency、cache reuse 和 active gate

该 checkpoint 早于 label-free runtime event index heads，因此 Demo 不使用事件 postings 或 gold labels 做 selector。所有任务仍通过同一个真实 encoder、query encoder 和 option scorer 计算；事件 index 本身明确不在本 Demo 的 claim 范围内。

## Parallel batch endpoint

除了单窗口 `POST /api/query`，Demo 还提供 `POST /api/multi`，用于展示长时程状态在多个 runtime workload 上并行复用：

```json
{
  "views": [
    {"kind": "full", "start": 338, "end": 343, "label": "ACTIVE"},
    {"kind": "rem", "start": 420, "end": 425, "label": "REM"}
  ]
}
```

返回结果中的每个 view 都包含 Choice、3 个 Noul 和 1 个 Score；主 view 进行 K-Symmetry 审计，其余 view 标记为共享本轮审计状态。

## Product contract

- 左侧 `OVERNIGHT STATE`：一次编码后的长时程 PSG 状态、时间窗和可审计的事件 posting。
- 中间 `RUNTIME QUERY`：target、stage、position 和显式 candidate meanings；对应 `SleepQuery` 的运行时字段。
- 右侧 `JEV DECISION`：只在当前 query 的 options 上归一化的概率分布，并同时展示 selected tokens、latency 和 readout mode。
- `RETRIEVED EVIDENCE`：展示 query-conditioned sparse readout 的证据组，不把 gold event label 暴露给 selector。
- `Cache reused`：强调高 query 数场景的系统价值——overnight encoding 只做一次，问题可以连续改变。

## Next integration step

将 `demo/index.html` 里的 `presets` 替换为后端返回的 `SleepQuery` / `model.answer()` JSON 即可。建议保留以下字段：

```json
{
  "query": {"target": "hypopnea", "options": ["negative", "positive"]},
  "probabilities": {"negative": 0.31, "positive": 0.62},
  "selected_tokens": 128,
  "n_epochs": 964,
  "latency_ms": 0.42,
  "readout_mode": "sparse",
  "evidence": [{"name": "Respiratory", "score": 0.86}]
}
```
