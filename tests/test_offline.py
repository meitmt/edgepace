"""Offline tests: no GPU, phone or model needed.

    python3 -m unittest discover -s tests -v        (or tests/run_offline.sh)

They cover the host-side pieces a deployment depends on: GGUF inspection, the Fine
Plan (graph cutting) from a profile trace, the SLO check, the interference factor k
the profile measures, and the workload description.
"""
from __future__ import annotations

import csv
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import edgepace_fineplan as cut  # noqa: E402
import edgepace_workload as workload  # noqa: E402
from edgepace_profile import interference_k  # noqa: E402

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore


def write_gguf(path: Path, metadata: dict, tensors: list[tuple[str, tuple[int, ...], int]]) -> None:
    """A minimal GGUF v3 file: metadata and tensor infos, no tensor data."""
    def string(s: str) -> bytes:
        b = s.encode()
        return struct.pack("<Q", len(b)) + b

    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata))
    for key, value in metadata.items():
        if isinstance(value, str):
            out += string(key) + struct.pack("<I", 8) + string(value)
        elif isinstance(value, bool):
            out += string(key) + struct.pack("<I?", 7, value)
        else:
            out += string(key) + struct.pack("<II", 4, value)
    for name, shape, kind in tensors:
        out += string(name) + struct.pack("<I", len(shape)) + b"".join(struct.pack("<Q", d) for d in shape)
        out += struct.pack("<IQ", kind, 0)
    path.write_bytes(out)


def write_trace(path: Path, phases: dict[str, list[float]]) -> None:
    """A Fine Plan profile trace (EDGEPACE_LOG_PATH): one complete event per quantum."""
    t = 1_000_000
    with path.open("w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["event", "phase", "quantum_id", "submit_ns", "complete_ns"])
        for phase, runtimes in phases.items():
            for q, ms in enumerate(runtimes):
                w.writerow(["complete", phase, q, t, t + int(ms * 1e6)])
                t += int(ms * 1e6) + 1000


class GgufTest(unittest.TestCase):
    def test_text_model_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.gguf"
            meta = {"general.architecture": "qwen2", "general.name": "tiny", "qwen2.block_count": 4,
                    "general.file_type": 1}
            write_gguf(p, meta, [("token_embd.weight", (64, 100), 1), ("blk.0.attn_q.weight", (64, 64), 1)])
            info = cut.inspect_gguf(p)
            self.assertEqual((info.architecture, info.layer_count, info.file_type), ("qwen2", 4, 1))
            # the fingerprint follows the layout (and the file size), not general.* or the path
            q = Path(d) / "renamed.gguf"
            write_gguf(q, {**meta, "general.name": "tinz"},
                       [("token_embd.weight", (64, 100), 1), ("blk.0.attn_q.weight", (64, 64), 1)])
            self.assertEqual(cut.inspect_gguf(q).fingerprint, info.fingerprint)
            write_gguf(q, meta, [("token_embd.weight", (64, 101), 1), ("blk.0.attn_q.weight", (64, 64), 1)])
            self.assertNotEqual(cut.inspect_gguf(q).fingerprint, info.fingerprint)

    def test_vision_encoder_boundaries(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "mmproj.gguf"
            write_gguf(p, {"general.architecture": "clip", "clip.has_vision_encoder": True,
                           "clip.vision.block_count": 2, "clip.projector_type": "qwen3vl_merger"},
                       [("v.blk.0.ffn_up.weight", (16, 64), 1)])
            vision = cut.inspect_mmproj(p)
            self.assertEqual(vision.boundaries, cut.QWEN3_VL_BOUNDARIES)
            self.assertEqual(vision.candidate_count, 2 * len(cut.QWEN3_VL_BOUNDARIES) + 1)

    def test_rejects_moe(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "moe.gguf"
            write_gguf(p, {"general.architecture": "qwen2", "qwen2.block_count": 4, "qwen2.expert_count": 8}, [])
            with self.assertRaises(ValueError):
                cut.inspect_gguf(p)


class FinePlanTest(unittest.TestCase):
    def setUp(self):
        self.info = cut.ModelInfo(path="m.gguf", size=1, mtime_ns=0, architecture="qwen2", layer_count=3,
                                  expert_count=0, file_type=1, fingerprint="f")

    def plan(self, prefill, decode, vision_runtimes=None, vision=None):
        with tempfile.TemporaryDirectory() as d:
            trace = Path(d) / "trace.csv"
            phases = {"prefill": prefill, "decode": decode}
            if vision_runtimes:
                phases["vision"] = vision_runtimes
            write_trace(trace, phases)
            profile = cut.profile_from_trace(trace, self.info.layer_count, vision)
            return cut.make_manifest(self.info, profile, 2048, 256, 128, vision)

    def test_text_keeps_every_legal_boundary(self):
        # prefill: 2 boundaries (ffn_inp, l_out) per layer + output norm + LM head
        m = self.plan([2.0] * 6 + [0.1, 9.0], [1.0] * 3 + [0.1, 5.0])
        self.assertEqual(len(m["phases"]["prefill"]["chunks"]), 8)
        self.assertEqual(m["phases"]["decode"]["terminal_runtime_ms"], 5.0)
        self.assertTrue(m["phases"]["decode"]["chunks"][-1]["terminal"])

    def test_missing_boundary_is_an_error(self):
        with self.assertRaises(ValueError):
            self.plan([2.0] * 5, [1.0] * 5)

    def test_vision_quanta_merged_into_range(self):
        vision = cut.VisionInfo(path="p", size=1, mtime_ns=0, projector="x", layer_count=4, image_size=448,
                                patch_size=14, boundaries=("attn_out", "layer_out"))
        runtimes = [0.6] * 8 + [3.0]                   # 8 body boundaries, then the terminal
        m = self.plan([2.0] * 6 + [0.1, 9.0], [1.0] * 3 + [0.1, 5.0], runtimes, vision)
        body = [c["estimated_runtime_ms"] for c in m["phases"]["vision"]["chunks"][:-1]]
        self.assertTrue(all(1.0 <= ms <= 4.0 for ms in body), body)
        self.assertAlmostEqual(sum(body), 0.6 * 8)

    def test_merge_to_range_keeps_a_long_boundary_alone(self):
        chunks = cut.merge_to_range([0.5, 9.0, 0.5, 0.5, 2.0], 1.0, 4.0)
        self.assertIn(9.0, [c.estimated_runtime_ms for c in chunks])
        self.assertTrue(chunks[-1].terminal)

    def test_one_slow_sample_does_not_reject_the_plan(self):
        # three profile requests; one of them hits a 12 ms spike on boundary 1
        prefill = [2.0] * 6 + [0.1, 9.0]
        with tempfile.TemporaryDirectory() as d:
            trace = Path(d) / "trace.csv"
            with trace.open("w", newline="") as handle:
                w = csv.writer(handle)
                w.writerow(["event", "phase", "quantum_id", "submit_ns", "complete_ns"])
                t = 1_000_000
                for request in range(3):
                    for phase, runtimes in (("prefill", prefill), ("decode", [1.0] * 3 + [0.1, 5.0])):
                        for q, ms in enumerate(runtimes):
                            ms = 12.0 if (phase, q, request) == ("prefill", 1, 2) else ms
                            w.writerow(["complete", phase, q, t, t + int(ms * 1e6)])
                            t += int(ms * 1e6) + 1000
            profile = cut.profile_from_trace(trace, self.info.layer_count)
            m = cut.make_manifest(self.info, profile, 2048, 256, 128)
        self.assertEqual(m["phases"]["prefill"]["chunks"][1]["estimated_runtime_ms"], 12.0)   # estimate: max
        a = cut.assess_slo(m, 33333, 1800, 4)                    # reference 11.1 ms
        self.assertTrue(a["admissible"], a["blockers"])          # check: median 2.0 ms
        self.assertEqual(a["phases"]["prefill"]["checked_statistic"], "median")

    def test_slo_check_and_uncuttable_quanta(self):
        m = self.plan([2.0] * 6 + [0.1, 40.0], [1.0] * 3 + [0.1, 40.0])
        strict = cut.assess_slo(m, 33333, 1800, 4)
        self.assertFalse(strict["admissible"])            # a 40 ms LM head over a 33 ms deadline
        relaxed = cut.assess_slo(m, 33333, 1800, 4, uncuttable=True)
        self.assertTrue(relaxed["admissible"], relaxed["blockers"])
        self.assertTrue(any("terminal" in a for a in relaxed["advisories"]))


class InterferenceTest(unittest.TestCase):
    def test_k_from_solo_and_unmanaged(self):
        # pipeline time grows by k per ms of LLM request overlap
        k, base = 0.25, 20.0
        with tempfile.TemporaryDirectory() as d:
            solo, unmanaged = Path(d) / "solo", Path(d) / "unmanaged"
            solo.mkdir(), unmanaged.mkdir()

            def frames(path: Path, durations: list[float]) -> list[tuple[int, int]]:
                spans = []
                with (path / "prediction.csv").open("w", newline="") as h:
                    w = csv.writer(h)
                    w.writerow(["frame_id", "pipeline_ns", "complete_ns"])
                    for i, ms in enumerate(durations, 1):
                        start = i * 40_000_000
                        w.writerow([i, start, start + int(ms * 1e6)])
                        spans.append((start, start + int(ms * 1e6)))
                return spans

            frames(solo, [base] * 50)
            overlaps = [0.0 if i % 3 else 10.0 for i in range(50)]   # a request overlaps every third frame
            durations = [base + k * o for o in overlaps]
            spans = frames(unmanaged, durations)
            with (unmanaged / "requests.csv").open("w", newline="") as h:
                w = csv.writer(h)
                w.writerow(["request_id", "start_ns", "latency_ms"])
                for n, ((start, _), o) in enumerate(zip(spans, overlaps)):
                    if o:
                        w.writerow([n, start, o])
            self.assertAlmostEqual(interference_k(solo, unmanaged, 0), k, places=2)


class WorkloadTest(unittest.TestCase):
    def test_examples_are_valid_toml(self):
        for path in sorted((ROOT / "examples/workloads").glob("*.toml")):
            with self.subTest(path.name), path.open("rb") as h:
                config = tomllib.load(h)
                self.assertTrue({"name", "platform", "foreground", "llm", "profile"} <= set(config))

    def test_phone_examples_load(self):
        for path in sorted((ROOT / "examples/workloads").glob("oneplus13_*.toml")):
            with self.subTest(path.name):
                config = workload.load(path)
                args = workload.plan_arguments(config, Path("/tmp/out"))
                self.assertIn("--device", args)
                if config["llm"].get("mmproj"):
                    self.assertIn("--mmproj", args)

    def test_repo_placeholder_and_vlm_arguments(self):
        with tempfile.TemporaryDirectory() as d:
            model = Path(d) / "m.gguf"
            model.write_bytes(b"")
            toml = Path(d) / "w.toml"
            toml.write_text(f'''
name = "t"
[platform]
power_mode = "MAXN_SUPER"
[foreground]
command = "/bin/sh -c true --video {{repo}}/media/x.mp4 --frames {{frames}} {{task_page}} {{progress_page}}"
[llm]
model = "{model}"
mmproj = "{model}"
request = "{ROOT / 'requests/vlm_image_16tok.json'}"
ubatch_size = 128
allow_uncuttable = true
[profile]
''')
            config = workload.load(toml)
            self.assertIn(f"{ROOT}/media/x.mp4", config["foreground"]["command"])
            args = workload.plan_arguments(config, Path(d) / "out")
            for flag in ("--mmproj", "--allow-uncuttable", "--ubatch-size"):
                self.assertIn(flag, args)


if __name__ == "__main__":
    unittest.main()
