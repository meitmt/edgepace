# EdgePace profile `oneplus13_phone4cam_qwen25_1p5b_f16`

status **profiled**, created 2026-10-25T21:31:26

## Workload

- model `/data/local/tmp/geniex/models/Qwen2.5-1.5B-Instruct-F16/Qwen2.5-1.5B-Instruct-F16.gguf`
- foreground `/data/local/tmp/edgepace/bin/fourcam_phone --video /data/local/tmp/edgepace/media/input_loop70s.mp4 --frames {frames} --warmup-frames 12 --period-us 33333 --deadline-us 33333 --condition llm --main-cpu 0 --worker-cpus 4,5,6,7 --output {phase_dir}/foreground.csv --semantic-output {phase_dir}/semantic_records.csv --task-page {task_page} --progress-page {phase_dir}/progress.page`
- LLM request `{repo}/requests/text_4tok.json`, period 1600 ms (frame), 1 slot(s), 0 extra stream(s), max_tokens 4, freshness 1800 ms
- foreground period / deadline 33333 / 33333 us

## Foreground alone

- misses 0.00%, response p95 16.3 ms
- reference scales r0 2.297e+06/s, m0 0.2609, s0 1.625
- quiet D0 0.236, S0 0.391

## Memory stress (llm:text_256tok.json x4)

- frame stall AUC 0.988 (median +14.3%), MPKI AUC 0.559 -> wS 0.7

## Thresholds

- theta_U 0.2226 / 0.5226 + shift 0.3093 = 0.5319 / 0.8319

## Warning horizon

unmanaged: misses 10.11%, LLM 4.50

| H (ms) | misses | LLM | retention |
|---|---|---|---|
| 4 | 0.73% | 3.95 | 87.8% |
| 8 | 0.79% | 3.91 | 86.8% |
| 12 | 0.51% | 3.89 | 86.4% |
| 16 | 0.40% | 3.89 | 86.3% |
| 20 | 0.34% | 3.95 | 87.8% |
| 24 ← | 0.23% | 3.85 | 85.5% |
| 32 | 0.11% | 3.79 | 84.3% |

objective `miss` (min retention 0.6, miss budget 0.02)

## Parameters

```
EDGEPACE_ONLINE_RHO=1
EDGEPACE_RHO_W_DEMAND=0.15
EDGEPACE_RHO_W_STALL=0.7
EDGEPACE_RHO_W_URGENCY=0.7
EDGEPACE_RHO_THETA_LOW=0.5319
EDGEPACE_RHO_THETA_HIGH=0.8319
EDGEPACE_RHO_HORIZON_MS=24
```

sampler: `--gate-margin 0.05 --gate-persistence 3 --gate-min-ms 6 --sample-interval-ms 4 --gate-persistence 2 --stall-threshold 1.62538 --miss-rate-threshold 2.29729e+06 --mpki-threshold 0.260904`
