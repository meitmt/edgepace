# Measuring EdgePace

`tools/edgepace evaluate PROFILE` runs the profile's workload — the same
application, LLM, request pattern and clocks — in alternating phases with the LLM
unmanaged (plain llama.cpp scheduling) and under the profile, and reports for each
phase the application's deadline misses and the LLM's throughput.

```bash
tools/edgepace evaluate runs/profile_light4cam_qwen25_1p5b_f16_<time>/profile --repeats 3
#   --frames 1800     frames per phase (60 s at 30 fps)
#   --conditions unmanaged,edgepace_profile
#   --dry-run         print the python/edgepace_eval.py command it runs
```

On a Jetson it runs under `scripts/locked_run.sh` (sudo); on a phone every phase
takes the clock lock itself, after the phone has cooled down. The output directory
(`runs/eval_<name>_<time>/`) has one directory per phase (`foreground.csv`: every
job's release, completion and miss; `requests.csv`: every LLM request; the policy's
decision trace `edgepace_trace.csv`), `summary.json`, and the table:

```
phase                     miss%  p95ms  llm/s  fresh  lat p50 tok/act-s  alarmed(miss)  quiet(miss)
r1_unmanaged              14.29   39.4  0.644     38    337.9      8.58    264( 229)   1506(  24)
r1_edgepace_profile        2.37   32.4  0.644     38    496.1      5.97     84(  15)   1686(  27)
```

| Column | Meaning |
|---|---|
| `miss%` | jobs that completed after their deadline |
| `p95ms` | 95th percentile job response time |
| `llm/s`, `fresh` | requests per second; requests answered within their freshness |
| `lat p50` | median request latency (ms) |
| `tok/act-s` | LLM output tokens per second the LLM was active: the throughput metric; EdgePace's retention = managed / unmanaged |
| `alarmed(miss)`, `quiet(miss)` | jobs with / without a progress alarm from the sampler's predictor, and the misses among them |

Several profiles can be compared in one run:
`python/edgepace_eval.py ... --profile a=dirA/profile.json --profile b=dirB/profile.json
--conditions unmanaged,profile_a,profile_b` (`tools/edgepace evaluate --dry-run` shows the
other arguments).

## Measurement hygiene

- Compare phases of one run: conditions alternate within each repeat, so slow drifts
  (temperature, the Orin Nano Super's over-current GPU throttling) affect all of them.
  `LOCK_CONTINUOUS=0` in `locked/command.log` (Jetson) or the phase lines (phone) marks
  phases whose clocks left the lock.
- Use ≥ 2 repeats; one frame is 0.06 % of an 1800-frame phase, and run-to-run
  differences of 1–3 points occur on both platforms.
- Do not load the device with anything else (including file-system traffic from a
  host) during a run.
- Phone specifics (thermal capping, the slow state): `android/README.md`.
