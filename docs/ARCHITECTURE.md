# How EdgePace works

```
 application (real-time, periodic jobs)      LLM server (llama.cpp + EdgePace)
 ───────────────────────────────────────     ──────────────────────────────────────
 task page:     release, deadline of the     ggml-backend.cpp scheduler
                current job        ─────────┐   graph ──cut (Fine Plan)──> quanta
 progress page: how far the job is ────────┤   before each quantum: ask the policy
                                           ├──> policy DSO: rho -> window 2 / 1 / 0
 PMU sampler (monitor/):                   │   submit while outstanding < window
   D, S on the application's threads ──────┤
 client: request page (running,            │
   freshness deadline) ────────────────────┘
```

All shared state is a set of small memory-mapped pages in a run directory
(`/dev/shm/edgepace/<name>` on Linux, `$DEVICE_ROOT/run/<name>` on a phone); their
layouts are in `llama.cpp/ggml/src/edgepace/abi.h` (task, request, signal pages) and
`monitor/third_party/v3_progress/foreground_progress.h` (progress page).

## 1. Fine Plan: cutting llama.cpp graphs into quanta

`ggml_backend_sched_graph_compute` normally submits a whole graph. EdgePace
classifies each graph (`classify` in `ggml-backend.cpp`): a text graph exposes
`l_out-N` nodes (decode: one boundary per layer; prefill: `ffn_inp` and `l_out` per
layer, or finer names with `EDGEPACE_PREFILL_BOUNDARIES`), a VLM's vision-encoder graph
exposes the clip callback names of `EDGEPACE_VISION_BOUNDARIES` per encoder layer. The
output norm and the LM head are the terminal quanta. A quantum is the span between
two consecutive boundaries of the plan; the scheduler splits the graph at those
nodes and submits the pieces one by one.

The plan comes from measurement (`python/edgepace_plan.py plan`, run by
`edgepace profile`): llama-server runs with `EDGEPACE_POLICY=profile_static` (one quantum
outstanding, every legal boundary a quantum), serves three real requests, and the
trace gives each boundary's runtime. Text phases keep every legal boundary; vision
boundaries (and finer prefill names) are merged into 1–4 ms quanta
(`merge_to_range`). The plan is cached by model content, context/batch/ubatch, cut
points and device (`build/plan-cache*`, `tools/edgepace plans`); the runtime reads
it as `EDGEPACE_PREFILL_MAP`, `EDGEPACE_DECODE_MAP`, `EDGEPACE_VISION_MAP`.

A quantum that is a single legal span cannot be cut further (an LM head, a patch
embedding); the SLO check (`assess_slo`) rejects plans whose quanta exceed the
application's deadline unless `allow_uncuttable` accepts such spans.

## 2. Sliding-window submission

Before each quantum the scheduler asks the policy for a decision with an
outstanding limit:

| decision | window | meaning |
|---|---|---|
| Bypass | 2 | llama.cpp's natural pipelining: the next quantum is queued while one runs |
| Admit | 1 | one quantum at a time: the LLM never runs ahead of the decision |
| Hold | 0 | wait; re-asked every 1 ms |

Window 2 reproduces what unmodified llama.cpp does on an asynchronous backend, so a
quiet application costs the LLM nothing; window 1 bounds the LLM's interference to
one 1–4 ms quantum after each decision.

## 3. The ρ policy

`llama.cpp/external/edgepace_policy/edgepace_policy_engine.h` (`evaluate_rho`), loaded from
`EDGEPACE_POLICY_LIBRARY` (only consulted with `EDGEPACE_CONTROL_REQUIRED=1`):

```
rho = wD·D + wS·S + wU·U
U   = clip(1 − L/H, 0, 1)                    while the application has an active job
L   = (d − t) − (1 + k)·W                    laxity: time to the deadline minus the
                                             remaining work W, stretched by k
W   = progress prediction (sampler) or R − (t − r)   (R: recent response time)
Bypass if rho < θ_low;  Admit if θ_low ≤ rho < θ_high;  Hold otherwise
Admit (freshness rescue) when the request's remaining freshness ≤ its remaining service
```

D is the memory demand rate and S the stall share on the application's threads,
each normalised by the application's own quiet-time scale; an invalid or stale PMU
sample counts as D = S = 1. H (`EDGEPACE_RHO_HORIZON_MS`) is how much laxity counts as
"no hurry": with quiet D and S the policy bypasses while L > 0.68·H, admits for
0.25–0.68·H and holds below that. k (`EDGEPACE_RHO_INTERFERENCE`) is how much longer the
application's work takes while the LLM runs.

Environment (all written by the profile into `server.env`): `EDGEPACE_ONLINE_RHO=1`,
`EDGEPACE_RHO_W_DEMAND`, `EDGEPACE_RHO_W_STALL`, `EDGEPACE_RHO_W_URGENCY`, `EDGEPACE_RHO_THETA_LOW`,
`EDGEPACE_RHO_THETA_HIGH`, `EDGEPACE_RHO_HORIZON_MS`, `EDGEPACE_RHO_INTERFERENCE`.

## 4. The profile

`python/edgepace_profile.py`, driven by `tools/edgepace profile` (quick mode):

1. **Application alone** (600 frames): the PMU scales r0, m0, s0 and the quiet D0, S0.
   The thresholds are θ = θ_U + wD·D0 + wS·S0 with θ_U = 0.2226 / 0.5226.
2. **Beside the unmanaged LLM** (600 frames): frames slowed past the solo p99 —
   whether S separates them from solo frames (AUC) sets wS (0.7 if it does, else 0);
   the per-frame slowdown against the time an LLM request overlapped the frame
   gives k (least squares).
3. H: 20 ms by default (`horizon_ms`), or measured: `mode = "sweep"` runs EdgePace
   at each H of `horizons` and picks one by the workload's objective (fewest misses
   at a minimum LLM throughput, or most throughput within a miss budget).

The profile directory holds `server.env`, `sampler.args`, `cut_plan.json`,
`profile.json` (parameters, workload, platform and binary fingerprint) and
`report.md`.

## Code map

| Piece | Files |
|---|---|
| Scheduler, graph cutting | `llama.cpp/ggml/src/ggml-backend.cpp`, `ggml-edgepace.h`; phone: `android/llama-overlay/ggml/src/ggml-backend.cpp` |
| Page layouts | `llama.cpp/ggml/src/edgepace/abi.h`, `edgepace_policy_abi.h` |
| Policy DSO | `llama.cpp/external/edgepace_policy/edgepace_policy.cpp`, `edgepace_policy_engine.h` |
| PMU sampler, progress predictor | `monitor/src/pmu_sampler.cpp`, `monitor/src/progress_predictor.h` |
| Fine Plan | `python/edgepace_fineplan.py` (planner), `python/edgepace_plan.py` (profiling, cache) |
| Profile | `python/edgepace_profile.py`, `python/edgepace_workload.py` |
| Deployment CLI | `python/edgepace_cli.py` (`tools/edgepace`) |
| Evaluation | `python/edgepace_eval.py`, `python/edgepace_phase.py` |
| Phone | `python/edgepace_android.py`, `android/` |
