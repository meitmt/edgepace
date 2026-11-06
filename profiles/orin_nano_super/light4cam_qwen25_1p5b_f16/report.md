# EdgePace profile `light4cam_qwen25_1p5b_f16`

status **profiled**, created 2026-10-24T00:02:43

## Workload

- model `{models}/Qwen2.5-1.5B-Instruct-F16.gguf`
- foreground `{repo}/build/foreground/fourcam_light --video /path/to/paper_clip_1080p30_hevc.mp4 --frames {frames} --warmup-frames 12 --period-us 33333 --deadline-us 33333 --condition llm --output {phase_dir}/foreground.csv --semantic-output {phase_dir}/semantic_records.csv --task-page {task_page} --progress-page {phase_dir}/progress.page --active-cameras 4 --partial-camera-period 0`
- LLM request `{repo}/requests/text_4tok.json`, period 1600 ms (frame), 1 slot(s), 0 extra stream(s), max_tokens 4, freshness 1800 ms
- foreground period / deadline 33333 / 33333 us

## Foreground alone

- misses 0.11%, response p95 29.6 ms
- reference scales r0 8.677e+08/s, m0 77.34, s0 0.7653
- quiet D0 0.241, S0 0.353

## GPU memory stress

- frame stall AUC 0.952 (median +31.3%), MPKI AUC 0.414 -> wS 0.7

## Thresholds

- theta_U 0.2226 / 0.5226 + shift 0.2834 = 0.5060 / 0.8060

## Warning horizon

unmanaged: misses 13.10%, LLM 8.44

| H (ms) | misses | LLM | retention |
|---|---|---|---|
| 12 | 2.99% | 5.99 | 71.0% |
| 16 ← | 0.92% | 5.72 | 67.8% |
| 20 | 1.38% | 5.29 | 62.7% |
| 24 | 1.61% | 4.96 | 58.8% |

objective `miss` (min retention -, miss budget -)

## Parameters

```
EDGEPACE_ONLINE_RHO=1
EDGEPACE_RHO_W_DEMAND=0.15
EDGEPACE_RHO_W_STALL=0.7
EDGEPACE_RHO_W_URGENCY=0.7
EDGEPACE_RHO_THETA_LOW=0.5060
EDGEPACE_RHO_THETA_HIGH=0.8060
EDGEPACE_RHO_HORIZON_MS=16
```

sampler: `--gate-margin 0.05 --gate-persistence 3 --gate-min-ms 6 --sample-interval-ms 4 --gate-persistence 2 --stall-threshold 0.765297 --miss-rate-threshold 8.67706e+08 --mpki-threshold 77.3386`
