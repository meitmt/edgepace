#!/usr/bin/env python3
"""Fine Plan tail-aware dense-model inspection and fine static cut planning."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import statistics
import struct
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO


SCHEMA_VERSION = "edgepace.g2.tail_aware_cut.v1"
PROTOTYPE_ID = "edgepace-fine-plan"
SUPPORTED_DENSE_ARCHITECTURES = {
    "gemma", "gemma2", "gemma3", "llama", "mistral", "phi2", "phi3",
    "qwen", "qwen2", "qwen3", "smollm", "smollm3", "starcoder2",
    # the text models of VLMs (their vision encoder comes from --mmproj)
    "qwen2vl", "qwen3vl",
}
# vision-encoder cut points per layer (clip callback names, graph order; a view
# node cuts before the next non-view node) by projector, and the quantum range the
# vision plan merges them into.  Qwen2/2.5/3-VL (dumped with EDGEPACE_VISION_DUMP): after
# the attention norm (layer 0: after the patch embedding), Q/K projection + rope, V
# projection, flash attention, output projection, FFN gate, FFN up + SwiGLU, FFN down +
# residual.  Any clip ViT has attn_out and layer_out.
QWEN_VL_BOUNDARIES = ("ln1", "Kcur_rope", "Vcur", "kqv_out", "attn_out", "ffn_gate_b", "ffn_swiglu",
                      "layer_out")
# Qwen3-VL's vision MLP has no gate (up, GELU, down); its deepstack layers reuse
# the ffn_* names after layer_out, behind the cut position, so they are passed over
QWEN3_VL_BOUNDARIES = ("ln1", "Kcur_rope", "kqv_out", "attn_out", "ffn_up_b", "ffn_out", "layer_out")
VISION_BOUNDARIES_BY_PROJECTOR = {
    "qwen2vl_merger": QWEN_VL_BOUNDARIES,
    "qwen2.5vl_merger": QWEN_VL_BOUNDARIES,
    "qwen3vl_merger": QWEN3_VL_BOUNDARIES,
}
DEFAULT_VISION_BOUNDARIES = ("attn_out", "layer_out")
VISION_QUANTUM_MS = (1.0, 4.0)
# text prefill cut points per layer.  The historical pair keeps every legal
# boundary (Fine Plan); finer points (inside the FFN: after the gate and the up projection,
# in graph order — llama's FFN computes the gate first, clip's the up) are merged into
# 1-4 ms quanta like the vision plan.  A quantum's length grows with the ubatch: a
# VLM's 256-token image ubatch makes an FFN half ~4x a short text prompt's (6.3 ms
# for Qwen2.5-VL-3B on the Orin); --ubatch-size 128 brings it to 3.7 ms with the
# historical pair at +0.5% request time (64: +8%), or use the finer points.
PREFILL_BODY_ADVISORY_MS = 4.0
LEGACY_PREFILL_BOUNDARIES = ("ffn_inp", "l_out")
FINE_PREFILL_BOUNDARIES = ("ffn_inp", "ffn_gate", "ffn_up", "l_out")
PREFILL_QUANTUM_MS = (1.0, 4.0)


@dataclass(frozen=True)
class VisionInfo:
    path: str
    size: int
    mtime_ns: int
    projector: str
    layer_count: int
    image_size: int
    patch_size: int
    boundaries: tuple[str, ...]
    fingerprint: str = ""

    @property
    def candidate_count(self) -> int:
        """k boundaries per layer, then the terminal (post-norm, projector)."""
        return len(self.boundaries) * self.layer_count + 1


@dataclass(frozen=True)
class ModelInfo:
    path: str
    size: int
    mtime_ns: int
    architecture: str
    layer_count: int
    expert_count: int
    file_type: int | None
    # what the quanta's runtimes depend on (gguf_fingerprint); the plan cache key
    fingerprint: str = ""


@dataclass(frozen=True)
class CutChunk:
    chunk_id: int
    start_boundary: int
    end_boundary: int
    estimated_runtime_ms: float
    terminal: bool


def _read_exact(handle: BinaryIO, size: int) -> bytes:
    value = handle.read(size)
    if len(value) != size:
        raise ValueError("truncated GGUF metadata")
    return value


def _u32(handle: BinaryIO) -> int:
    return struct.unpack("<I", _read_exact(handle, 4))[0]


def _u64(handle: BinaryIO) -> int:
    return struct.unpack("<Q", _read_exact(handle, 8))[0]


def _string(handle: BinaryIO) -> str:
    return _read_exact(handle, _u64(handle)).decode("utf-8")


def _value(handle: BinaryIO, value_type: int) -> Any:
    scalar_formats = {
        0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
        6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
    }
    if value_type in scalar_formats:
        fmt = scalar_formats[value_type]
        return struct.unpack(fmt, _read_exact(handle, struct.calcsize(fmt)))[0]
    if value_type == 8:
        return _string(handle)
    if value_type == 9:
        element_type = _u32(handle)
        return [_value(handle, element_type) for _ in range(_u64(handle))]
    raise ValueError(f"unsupported GGUF metadata type {value_type}")


def gguf_metadata(path: Path) -> dict[str, Any]:
    return gguf_layout(path)[0]


def gguf_layout(path: Path) -> tuple[dict[str, Any], list[tuple[str, tuple[int, ...], int]]]:
    """GGUF key/value metadata and tensor infos (name, shape, type)."""
    metadata: dict[str, Any] = {}
    tensors: list[tuple[str, tuple[int, ...], int]] = []
    with path.open("rb") as handle:
        if _read_exact(handle, 4) != b"GGUF":
            raise ValueError(f"not a GGUF model: {path}")
        version = _u32(handle)
        if version not in {2, 3}:
            raise ValueError(f"unsupported GGUF version {version}")
        tensor_count = _u64(handle)
        metadata_count = _u64(handle)
        for _ in range(metadata_count):
            key = _string(handle)
            metadata[key] = _value(handle, _u32(handle))
        for _ in range(tensor_count):
            name = _string(handle)
            shape = tuple(_u64(handle) for _ in range(_u32(handle)))
            tensor_type = _u32(handle)
            _u64(handle)   # data offset
            tensors.append((name, shape, tensor_type))
    return metadata, tensors


def gguf_fingerprint(metadata: dict[str, Any], tensors: list[tuple[str, tuple[int, ...], int]],
                     size: int) -> str:
    """What a model's quanta runtimes depend on: its tensors' names, shapes and
    types, its hyperparameters (metadata other than general.* and tokenizer.*) and the
    file size; not its path, file time or weights.  A copy or a move of the model keeps
    its cut plan; so does a fine-tune with the same layout."""
    import hashlib
    hyper = sorted((k, repr(v)) for k, v in metadata.items()
                   if not k.startswith(("general.", "tokenizer.")))
    digest = hashlib.sha256(repr((hyper, tensors, size)).encode())
    return digest.hexdigest()[:16]


def inspect_mmproj(path: Path, boundaries: tuple[str, ...] = (), size: int | None = None) -> VisionInfo:
    """The vision encoder of a VLM (llama.cpp mmproj GGUF); without
    boundaries, those of its projector type (VISION_BOUNDARIES_BY_PROJECTOR).
    size: as inspect_gguf's, when `path` holds only the file's head (a phone mmproj)."""
    stat = path.stat()
    metadata, tensors = gguf_layout(path)
    if metadata.get("general.architecture") != "clip" or not metadata.get("clip.has_vision_encoder"):
        raise ValueError(f"not a vision mmproj: {path}")
    layers = metadata.get("clip.vision.block_count")
    if not isinstance(layers, int) or layers <= 0:
        raise ValueError("missing positive GGUF metadata clip.vision.block_count")
    projector = str(metadata.get("clip.projector_type", ""))
    boundaries = tuple(boundaries) or VISION_BOUNDARIES_BY_PROJECTOR.get(projector, DEFAULT_VISION_BOUNDARIES)
    return VisionInfo(
        path=str(path.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
        projector=projector, layer_count=layers,
        image_size=int(metadata.get("clip.vision.image_size", 0)),
        patch_size=int(metadata.get("clip.vision.patch_size", 0)), boundaries=tuple(boundaries),
        fingerprint=gguf_fingerprint(metadata, tensors, stat.st_size if size is None else size),
    )


def inspect_gguf(path: Path, size: int | None = None) -> ModelInfo:
    """size: the model file's size when `path` holds only its head (a phone model,
    edgepace_android.model_info)."""
    stat = path.stat()
    metadata, tensors = gguf_layout(path)

    architecture = str(metadata.get("general.architecture", ""))
    if architecture not in SUPPORTED_DENSE_ARCHITECTURES:
        raise ValueError(
            f"unsupported model architecture {architecture!r}; "
            "Fine Plan accepts dense decoder-only text models"
        )
    layer_key = f"{architecture}.block_count"
    layer_count = metadata.get(layer_key)
    if not isinstance(layer_count, int) or layer_count <= 0:
        raise ValueError(f"missing positive GGUF metadata {layer_key}")
    expert_count = metadata.get(f"{architecture}.expert_count", 0)
    if not isinstance(expert_count, int) or expert_count < 0:
        raise ValueError(f"invalid GGUF expert count for architecture {architecture}")
    if expert_count > 0:
        raise ValueError(
            f"unsupported MoE model: architecture={architecture}, "
            f"expert_count={expert_count}"
        )
    file_type = metadata.get("general.file_type")
    return ModelInfo(
        path=str(path.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
        architecture=architecture, layer_count=layer_count,
        expert_count=expert_count,
        file_type=int(file_type) if isinstance(file_type, int) else None,
        fingerprint=gguf_fingerprint(metadata, tensors, stat.st_size if size is None else size),
    )


def candidate_count(layer_count: int, phase: str,
                    prefill_boundaries: tuple[str, ...] = LEGACY_PREFILL_BOUNDARIES) -> int:
    if phase == "prefill":
        return len(prefill_boundaries) * layer_count + 2
    if phase == "decode":
        return layer_count + 2
    raise ValueError(f"unknown phase {phase}")


def fine_map(layer_count: int, phase: str,
             prefill_boundaries: tuple[str, ...] = LEGACY_PREFILL_BOUNDARIES) -> str:
    return ",".join(
        f"{boundary}:1.0" for boundary in range(candidate_count(layer_count, phase, prefill_boundaries))
    )


def vision_fine_map(vision: VisionInfo) -> str:
    return ",".join(f"{boundary}:1.0" for boundary in range(vision.candidate_count))


def profile_from_trace(path: Path, layer_count: int, vision: VisionInfo | None = None,
                       prefill_boundaries: tuple[str, ...] = LEGACY_PREFILL_BOUNDARIES) -> dict[str, list[float]]:
    observed: dict[str, dict[int, list[float]]] = {"prefill": {}, "decode": {}, "vision": {}}
    samples: list[tuple[str, int, int, int]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            phase = row.get("phase", "")
            if row.get("event") != "complete" or phase not in observed:
                continue
            submit_ns = int(row["submit_ns"])
            complete_ns = int(row["complete_ns"])
            if complete_ns <= submit_ns:
                continue
            samples.append((phase, int(row["quantum_id"]), submit_ns, complete_ns))
    # a VLM runs its vision encoder on its own scheduler between two text
    # graphs, and the asynchronous text quantum before it is seen complete only at
    # the next text sync: a sample whose interval holds another phase's quantum is
    # not its own runtime.  (Kept if a quantum has no other sample.)
    starts = sorted((submit, phase) for phase, _, submit, _ in samples)
    keys = [submit for submit, _ in starts]
    by_quantum: dict[tuple[str, int], list[tuple[float, bool]]] = {}
    for phase, quantum_id, submit_ns, complete_ns in samples:
        lo, hi = bisect.bisect_right(keys, submit_ns), bisect.bisect_left(keys, complete_ns)
        overlapped = any(starts[i][1] != phase for i in range(lo, hi))
        by_quantum.setdefault((phase, quantum_id), []).append(((complete_ns - submit_ns) / 1e6, overlapped))
    for (phase, quantum_id), values in by_quantum.items():
        clean = [ms for ms, overlapped in values if not overlapped]
        observed[phase][quantum_id] = clean or [min(ms for ms, _ in values)]

    # per boundary: the maximum (the plan's runtime estimate) and the median (its SLO
    # check, one slow sample of the three requests must not reject a plan)
    result: dict[str, list[float]] = {}
    for phase in ("prefill", "decode"):
        expected = candidate_count(layer_count, phase, prefill_boundaries)
        missing = [index for index in range(expected) if not observed[phase].get(index)]
        if missing:
            raise ValueError(f"{phase} graph did not expose expected boundaries: {missing}"
                             + (f" (prefill boundaries {list(prefill_boundaries)})" if phase == "prefill" else ""))
        result[phase] = [max(observed[phase][index]) for index in range(expected)]
        result[f"{phase}_median"] = [statistics.median(observed[phase][index]) for index in range(expected)]
    if vision is not None:
        expected = vision.candidate_count
        missing = [index for index in range(expected) if not observed["vision"].get(index)]
        if missing:
            k = len(vision.boundaries)
            names = sorted({vision.boundaries[i % k] for i in missing if i < expected - 1})
            raise ValueError(f"vision graph did not expose {len(missing)} of {expected} boundaries "
                             f"(names {names or ['terminal']}); check --vision-boundaries against the "
                             f"clip graph of projector {vision.projector!r}")
        result["vision"] = [max(observed["vision"][index]) for index in range(expected)]
        result["vision_median"] = [statistics.median(observed["vision"][index]) for index in range(expected)]
    return result


def merge_to_range(runtimes_ms: list[float], low_ms: float, high_ms: float) -> list[CutChunk]:
    """Consecutive legal boundaries merged into quanta of low..high ms: a
    quantum closes before the boundary that would take it past high; one shorter
    than low then joins its shorter neighbour if the pair stays within high.  A
    single boundary longer than high stays a quantum of its own.  The terminal
    (post-norm and projector) stays isolated, as for the text phases."""
    if len(runtimes_ms) < 2:
        raise ValueError("a phase must expose at least one body and one terminal boundary")
    groups: list[list[int]] = []
    current: list[int] = []
    total = 0.0
    for index, runtime in enumerate(runtimes_ms[:-1]):
        if current and total + runtime > high_ms:
            groups.append(current)
            current, total = [], 0.0
        current.append(index)
        total += runtime
    if current:
        groups.append(current)
    span = lambda group: sum(runtimes_ms[i] for i in group)
    merged = True
    while merged:
        merged = False
        for g, group in enumerate(groups):
            if span(group) >= low_ms or len(groups) == 1:
                continue
            neighbours = [n for n in (g - 1, g + 1) if 0 <= n < len(groups)
                          and span(groups[n]) + span(group) <= high_ms]
            if neighbours:
                n = min(neighbours, key=lambda n: span(groups[n]))
                lo, hi = min(g, n), max(g, n)
                groups[lo:hi + 1] = [groups[lo] + groups[hi]]
                merged = True
                break
    terminal = len(runtimes_ms) - 1
    groups.append([terminal])
    return [CutChunk(i, group[0], group[-1], span(group), group[-1] == terminal)
            for i, group in enumerate(groups)]


def preserve_legal_boundaries(runtimes_ms: list[float]) -> list[CutChunk]:
    if len(runtimes_ms) < 2:
        raise ValueError("a phase must expose at least one body and one terminal boundary")
    if any(runtime <= 0.0 for runtime in runtimes_ms):
        raise ValueError("all measured legal quanta must be positive")
    terminal_id = len(runtimes_ms) - 1
    return [
        CutChunk(index, index, index, runtime, index == terminal_id)
        for index, runtime in enumerate(runtimes_ms)
    ]


def encode_chunks(chunks: list[CutChunk]) -> str:
    return ",".join(
        f"{chunk.end_boundary}:{chunk.estimated_runtime_ms:.6f}" for chunk in chunks
    )


def automatic_reference_ms(deadline_us: int, freshness_ms: float) -> float:
    if deadline_us <= 0 or freshness_ms <= 0:
        raise ValueError("deadline and freshness must be positive")
    return min(deadline_us / 3000.0, freshness_ms / 50.0)


def cache_name(info: ModelInfo, context: int, batch: int, ubatch: int,
               vision: VisionInfo | None = None,
               prefill_boundaries: tuple[str, ...] = LEGACY_PREFILL_BOUNDARIES,
               platform: str = "") -> str:
    """The plan cache key: model and vision encoder by content (fingerprint),
    graph shape (context, batch, ubatch), cut points, and the device it was timed on."""
    parts = [f"plan_{info.architecture}_{info.layer_count}_m{info.fingerprint}",
             f"c{context}_b{batch}_u{ubatch}"]
    if vision is not None:
        parts.append("v" + vision.fingerprint + "{:08x}".format(
            zlib.crc32(",".join(vision.boundaries).encode()) & 0xFFFFFFFF))
    if tuple(prefill_boundaries) != LEGACY_PREFILL_BOUNDARIES:
        parts.append("p{:08x}".format(zlib.crc32(",".join(prefill_boundaries).encode()) & 0xFFFFFFFF))
    if platform:
        parts.append("d{:08x}".format(zlib.crc32(platform.encode()) & 0xFFFFFFFF))
    return "_".join(parts) + ".json"


def legacy_cache_name(info: ModelInfo, context: int, batch: int, ubatch: int,
                      vision: VisionInfo | None = None,
                      prefill_boundaries: tuple[str, ...] = LEGACY_PREFILL_BOUNDARIES) -> str:
    """The key before (model path, size and file time), to adopt old entries."""
    path_key = zlib.crc32(info.path.encode("utf-8")) & 0xFFFFFFFF
    file_type = "none" if info.file_type is None else str(info.file_type)
    vision_key = "" if vision is None else "_v{:08x}".format(zlib.crc32(
        f"{vision.path}|{vision.size}|{vision.mtime_ns}|{','.join(vision.boundaries)}".encode()) & 0xFFFFFFFF)
    if tuple(prefill_boundaries) != LEGACY_PREFILL_BOUNDARIES:
        vision_key += "_p{:08x}".format(zlib.crc32(",".join(prefill_boundaries).encode()) & 0xFFFFFFFF)
    return (
        f"plan_{info.architecture}_{info.layer_count}_p{path_key:08x}_"
        f"{info.size}_{info.mtime_ns}_f{file_type}_c{context}_b{batch}_u{ubatch}{vision_key}.json"
    )


def vision_phase(vision: VisionInfo, runtimes: list[float],
                 quantum_ms: tuple[float, float] = VISION_QUANTUM_MS,
                 medians: list[float] | None = None) -> dict[str, Any]:
    """The vision plan: legal boundaries merged into quantum_ms quanta."""
    chunks = merge_to_range(runtimes, *quantum_ms)
    body = [chunk.estimated_runtime_ms for chunk in chunks[:-1]]
    return {
        "candidate_count": len(runtimes),
        "candidate_runtime_ms": runtimes,
        **({"candidate_median_ms": medians} if medians else {}),
        "boundaries": list(vision.boundaries),
        "quantum_range_ms": list(quantum_ms),
        "body_boundary_count": len(runtimes) - 1,
        "body_max_runtime_ms": max(body),
        "body_min_runtime_ms": min(body),
        "body_quanta_in_range": sum(quantum_ms[0] <= value <= quantum_ms[1] for value in body),
        "body_quanta": len(body),
        "terminal_boundary": len(runtimes) - 1,
        "terminal_runtime_ms": runtimes[-1],
        "chunks": [asdict(chunk) for chunk in chunks],
        "encoded_map": encode_chunks(chunks),
    }


def make_manifest(
    info: ModelInfo, profile: dict[str, list[float]],
    context: int, batch: int, ubatch: int,
    vision: VisionInfo | None = None,
    prefill_boundaries: tuple[str, ...] = LEGACY_PREFILL_BOUNDARIES,
) -> dict[str, Any]:
    phases: dict[str, Any] = {}
    fine_prefill = tuple(prefill_boundaries) != LEGACY_PREFILL_BOUNDARIES
    for phase in ("prefill", "decode"):
        runtimes = profile[phase]
        merged = phase == "prefill" and fine_prefill
        chunks = merge_to_range(runtimes, *PREFILL_QUANTUM_MS) if merged else preserve_legal_boundaries(runtimes)
        body = [chunk.estimated_runtime_ms for chunk in chunks[:-1]]
        extra = {"boundaries": list(prefill_boundaries), "quantum_range_ms": list(PREFILL_QUANTUM_MS),
                 "body_min_runtime_ms": min(body), "body_quanta": len(body),
                 "body_quanta_in_range": sum(PREFILL_QUANTUM_MS[0] <= v <= PREFILL_QUANTUM_MS[1] for v in body)} \
            if merged else {}
        medians = profile.get(f"{phase}_median")
        phases[phase] = {
            **extra,
            "candidate_count": len(runtimes),
            "candidate_runtime_ms": runtimes,
            **({"candidate_median_ms": medians} if medians else {}),
            "body_boundary_count": len(runtimes) - 1,
            "body_max_runtime_ms": max(body),
            "terminal_boundary": len(runtimes) - 1,
            "terminal_runtime_ms": runtimes[-1],
            "chunks": [asdict(chunk) for chunk in chunks],
            "encoded_map": encode_chunks(chunks),
        }
    if vision is not None:
        phases["vision"] = vision_phase(vision, profile["vision"], medians=profile.get("vision_median"))
    return {
        "schema_version": SCHEMA_VERSION,
        "prototype": PROTOTYPE_ID,
        "policy": "tail_aware_slo_online",
        "parent_prototype": "G1-adaptive-static-slo",
        "model": asdict(info),
        **({"vision_model": asdict(vision)} if vision is not None else {}),
        "context": context,
        "batch": batch,
        "ubatch": ubatch,
        "selection": {
            "strategy": "preserve_every_legal_boundary",
            "terminal_policy": "isolate_final_output_tail",
            "priority_independent": True,
            "slo_independent_geometry": True,
        },
        "profile_statistic": (
            "maximum submit-to-complete duration over three startup requests "
            "(the SLO check uses the median)"
        ),
        "phases": phases,
    }


def assess_slo(
    manifest: dict[str, Any], deadline_us: int, freshness_ms: float,
    max_tokens: int, uncuttable: bool = False,
) -> dict[str, Any]:
    """uncuttable: a quantum of a single legal span (the cut cannot shorten it)
    over its limit is an advisory, not a blocker: on a phone, the tied Q6_K LM head
    (terminal) and the vision patch embedding (IM2COL) the NPU cannot run stay on the CPU."""
    if deadline_us <= 0 or freshness_ms <= 0 or max_tokens <= 0:
        raise ValueError("deadline, freshness, and max tokens must be positive")
    reference_ms = automatic_reference_ms(deadline_us, freshness_ms)
    deadline_ms = deadline_us / 1000.0
    phase_result: dict[str, Any] = {}
    blockers: list[str] = []
    advisories: list[str] = []
    for phase in ("vision", "prefill", "decode"):
        if phase not in manifest["phases"]:
            continue
        value = manifest["phases"][phase]
        # a quantum is checked at its median runtime (the sum of its boundaries'
        # medians); the plan's estimates stay the maxima.  Plans cached before keep the maxima.
        medians = value.get("candidate_median_ms")
        checked = [(c, sum(medians[c["start_boundary"]:c["end_boundary"] + 1]) if medians
                    else float(c["estimated_runtime_ms"])) for c in value["chunks"]]
        body = [(c, ms) for c, ms in checked if not c["terminal"]]
        body_max = max(ms for _, ms in body)
        terminal = next(ms for c, ms in checked if c["terminal"])
        body_fit = body_max <= reference_ms
        terminal_fit = terminal <= deadline_ms
        if uncuttable and not body_fit:
            single = [ms for c, ms in body if c["start_boundary"] == c["end_boundary"] and ms > reference_ms]
            cuttable = [ms for c, ms in body if c["start_boundary"] != c["end_boundary"]]
            if not cuttable or max(cuttable) <= reference_ms:
                body_fit = True
                advisories.append(f"{phase}: {len(single)} single-span quanta up to {max(single):.3f}ms "
                                  f"exceed control reference {reference_ms:.3f}ms (not cuttable)")
        phase_result[phase] = {
            "checked_statistic": "median" if medians else "maximum",
            "body_max_estimate_ms": float(value["body_max_runtime_ms"]),
            "body_max_runtime_ms": body_max,
            "body_within_control_reference": body_fit,
            "terminal_runtime_ms": terminal,
            "terminal_within_control_reference": terminal <= reference_ms,
            "terminal_within_hard_deadline": terminal_fit,
        }
        if not body_fit:
            blockers.append(
                f"{phase} body legal quantum {body_max:.3f}ms exceeds "
                f"control reference {reference_ms:.3f}ms"
            )
        if not terminal_fit:
            (advisories if uncuttable else blockers).append(
                f"{phase} terminal quantum {terminal:.3f}ms exceeds "
                f"foreground deadline {deadline_ms:.3f}ms"
            )

    prefill_total = sum(
        float(value["estimated_runtime_ms"])
        for value in manifest["phases"]["prefill"]["chunks"]
    )
    decode_total = sum(
        float(value["estimated_runtime_ms"])
        for value in manifest["phases"]["decode"]["chunks"]
    )
    vision_total = sum(float(value["estimated_runtime_ms"])
                       for value in manifest["phases"].get("vision", {}).get("chunks", []))
    request_service_ms = vision_total + prefill_total + max_tokens * decode_total
    if request_service_ms > freshness_ms:
        blockers.append(
            f"profiled request service {request_service_ms:.3f}ms exceeds "
            f"freshness {freshness_ms:.3f}ms"
        )
    prefill_body = float(manifest["phases"]["prefill"]["body_max_runtime_ms"])
    if prefill_body > PREFILL_BODY_ADVISORY_MS:
        advisories.append(
            f"prefill body quantum {prefill_body:.2f}ms exceeds {PREFILL_BODY_ADVISORY_MS:g}ms: lower "
            f"--ubatch-size or cut inside the FFN (--prefill-boundaries ffn_inp,ffn_gate,ffn_up,l_out)")
    return {
        "advisories": advisories,
        "deadline_ms": deadline_ms,
        "freshness_ms": freshness_ms,
        "control_reference_ms": reference_ms,
        "reference_role": "body feasibility check only; never a coarsening target",
        "max_tokens": max_tokens,
        "profiled_request_service_ms": request_service_ms,
        "request_service_within_freshness": request_service_ms <= freshness_ms,
        "phases": phase_result,
        "admissible": not blockers,
        "blockers": blockers,
    }


def validate_slo(assessment: dict[str, Any]) -> None:
    blockers = assessment.get("blockers", [])
    if blockers:
        raise ValueError("; ".join(str(value) for value in blockers))


def session_manifest(
    base_manifest: dict[str, Any], assessment: dict[str, Any],
) -> dict[str, Any]:
    result = dict(base_manifest)
    result["slo_assessment"] = assessment
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--deadline-us", type=int, default=33333)
    parser.add_argument("--freshness-ms", type=float, default=1800.0)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--ubatch", type=int, default=256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    info = inspect_gguf(args.model)
    if args.inspect_only:
        value: dict[str, Any] = {"model": asdict(info)}
    else:
        if args.trace is None:
            parser.error("--trace is required unless --inspect-only is used")
        base = make_manifest(
            info, profile_from_trace(args.trace, info.layer_count),
            args.context, args.batch, args.ubatch,
        )
        assessment = assess_slo(base, args.deadline_us, args.freshness_ms, args.max_tokens)
        validate_slo(assessment)
        value = session_manifest(base, assessment)
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
