#!/usr/bin/env python3
"""#238 G2: determine whether the tiny Qwen3.5 adaptation has a Cactus path.

Question: can the adapted weights run behind a Cactus runtime, through a
*supported upstream packaging path* -- or is custom deployment BLOCKED?

What this checks (all read-only, CPU-only, no model load):

1. adapted-name coverage in the stock CQ conversion manifest: every LoRA-
   adapted HF tensor (from the merge-stage ``changed_params``) must resolve
   to a converter output file with a known quant policy. This shows the
   converter's name-mapping covers the adapted set -- mapping only, not proof
   the runtime uses changed bytes.
2. hosted bundle runnability: the pinned Cactus-Compute/Qwen3.5-2B artifact
   must contain a runnable bundle (components/manifest.json + graph), or it
   is weights-only and there is nothing to deploy adapted weights *into*.
3. supported packaging path: upstream ``cactus convert --lora`` merges the
   adapter and builds weights + graph. The graph half is the binding #380
   failure (no qwen3_5 export profile; torch.export guard on
   ``_update_linear_attn_mask``), and upstream docs state local bundle
   generation is unavailable while the graph builder is rewritten.
4. changed-weight checksums at HF level (sha256 per changed tensor, base vs
   merged) -- so "the file changed" is measured, separately from proof of
   runtime use.
5. proof-of-use criterion: a runnable adapted artifact generating text with a
   matched HF-vs-Cactus adaptation-sensitive delta. Recorded as NOT MET when
   no runnable artifact exists on either side of the comparison.

A changed CQ weights file is NOT proof the runtime uses it. PASS requires a
runnable adapted artifact; otherwise the verdict is custom deployment BLOCKED
with the exact upstream requirement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from g2_hf_weight_audit import (  # noqa: E402
    _DTYPES,
    load_headers,
    read_header,
    safetensors_files,
)

RUNNABLE_MARKERS = ("components/manifest.json", "graph.cactus",
                    "runtime_plan.json")
ADAPTER_SENSITIVE_PROBE_NOTE = (
    "HF-side adaptation-sensitive probes are in outputs/g2_roundtrip/ "
    "compare.json (base-vs-adapted worst logit delta 9.05, all 4 greedy "
    "texts changed). The Cactus side of the matched comparison needs a "
    "runnable stock bundle first (G1 BLOCKED) and a runnable adapted bundle "
    "(this check)."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merge-json",
                        default="outputs/g2_roundtrip/merge.json")
    parser.add_argument("--audit-json",
                        default="outputs/g2_hf_weight_audit.json")
    parser.add_argument("--base-manifest", default=None,
                        help="stock CQ conversion_manifest.json (retry "
                             "weights-only evidence)")
    parser.add_argument("--hosted-dir", default=None,
                        help="extracted hosted flat-weights bundle dir "
                             "(G1 evidence)")
    parser.add_argument("--sdk-dir", default=None,
                        help="upstream cactus-sdk checkout (read-only "
                             "source quotes)")
    parser.add_argument("--base-dir", default=None,
                        help="base HF checkpoint dir for checksums")
    parser.add_argument("--merged-dir",
                        default="outputs/merged-model-g2-qwen35-tiny")
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def manifest_coverage(changed: list[str], manifest: list[dict]) -> dict[str, Any]:
    by_hf = {entry.get("hf_name"): entry for entry in manifest}
    mapped, unmapped = [], []
    precisions: dict[str, int] = {}
    statuses: dict[str, int] = {}
    outputs = []
    for name in changed:
        entry = by_hf.get(name)
        if entry is None:
            unmapped.append(name)
            continue
        mapped.append(name)
        precisions[entry.get("precision", "?")] = \
            precisions.get(entry.get("precision", "?"), 0) + 1
        statuses[entry.get("status", "?")] = \
            statuses.get(entry.get("status", "?"), 0) + 1
        outputs.append({"hf_name": name,
                        "cq_file": entry.get("output_file"),
                        "status": entry.get("status"),
                        "precision": entry.get("precision"),
                        "fallback_reason": entry.get("fallback_reason")})
    return {
        "changed_total": len(changed),
        "mapped": len(mapped),
        "unmapped_names": unmapped,
        "precision_breakdown": precisions,
        "status_breakdown": statuses,
        "per_tensor": outputs,
    }


def bundle_structure(hosted_dir: Path | None) -> dict[str, Any]:
    report: dict[str, Any] = {"hosted_dir": str(hosted_dir) if hosted_dir
                              else None}
    if hosted_dir is None or not hosted_dir.is_dir():
        report["inspected"] = False
        report["reason"] = "hosted dir not provided or absent"
        return report
    files = [p for p in hosted_dir.iterdir()]
    weights = sorted(p.name for p in files if p.suffix == ".weights")
    markers = {}
    for marker in RUNNABLE_MARKERS:
        markers[marker] = (hosted_dir / marker).exists()
    report.update({
        "inspected": True,
        "total_files": len(files),
        "weights_files": len(weights),
        "runnable_markers": markers,
        "runnable": all(markers.values()),
        "conclusion": ("runnable bundle" if all(markers.values())
                       else "weights-only (no runnable bundle)"),
    })
    return report


def sdk_facts(sdk_dir: Path | None) -> dict[str, Any]:
    report: dict[str, Any] = {"sdk_dir": str(sdk_dir) if sdk_dir else None}
    if sdk_dir is None or not sdk_dir.is_dir():
        report["inspected"] = False
        return report
    report["inspected"] = True

    finetune = sdk_dir / "docs" / "finetuning.md"
    weights_only_quote = None
    if finetune.is_file():
        text = finetune.read_text(encoding="utf-8")
        match = re.search(r"[^\n]*CQ weights only[^\n]*\n[^\n]*graph builder"
                          r"[^\n]*", text)
        weights_only_quote = match.group(0).strip() if match else None
    report["finetuning_weights_only_quote"] = weights_only_quote

    utils_py = sdk_dir / "python" / "cactus" / "cli" / "utils.py"
    cq_re = None
    if utils_py.is_file():
        for lineno, line in enumerate(utils_py.read_text(
                encoding="utf-8").splitlines(), 1):
            if line.startswith("CQ_VARIANT_RE"):
                cq_re = f"utils.py:{lineno}: {line.strip()}"
    report["cq_variant_resolver"] = cq_re
    hosted_names = ["weights/qwen3.5-2b-int4.zip",
                    "weights/qwen3.5-2b-int8.zip"]
    pattern = re.compile(r"-cq(\d+(?:\.\d+)?)$", re.IGNORECASE)
    report["hosted_archives_match_cq_resolver"] = {
        name: bool(pattern.search(Path(name).stem)) for name in hosted_names}

    profiles_py = (sdk_dir / "python" / "cactus" / "transpiler"
                   / "ModelProfiles" / "profiles.py")
    report["qwen3_5_profile_present"] = (
        "qwen3_5" in profiles_py.read_text(encoding="utf-8")
        if profiles_py.is_file() else None)

    convert_py = sdk_dir / "python" / "cactus" / "cli" / "convert.py"
    report["lora_merge_path_present"] = (
        "_merge_lora_adapter" in convert_py.read_text(encoding="utf-8")
        if convert_py.is_file() else None)
    return report


def tensor_sha256(directory: Path, names: list[str]) -> dict[str, str]:
    """Raw-byte sha256 per named tensor (single- or multi-file safetensors)."""
    locations: dict[str, tuple[Path, int, int, int]] = {}
    for file_path in safetensors_files(directory):
        header, data_start = read_header(file_path)
        for name, spec in header.items():
            if name in names:
                dtype, width = _DTYPES[spec["dtype"]]
                begin, end = spec["data_offsets"]
                locations[name] = (file_path, data_start + begin,
                                   data_start + end)
    digests: dict[str, str] = {}
    open_handles: dict[Path, Any] = {}
    try:
        for name in names:
            if name not in locations:
                continue
            file_path, begin, end = locations[name]
            handle = open_handles.get(file_path)
            if handle is None:
                handle = file_path.open("rb")
                open_handles[file_path] = handle
            handle.seek(begin)
            digest = hashlib.sha256()
            remaining = end - begin
            while remaining:
                chunk = handle.read(min(1 << 20, remaining))
                if not chunk:
                    break
                digest.update(chunk)
                remaining -= len(chunk)
            digests[name] = digest.hexdigest()
    finally:
        for handle in open_handles.values():
            handle.close()
    return digests


def main(argv: list[str] | None = None) -> int:
    args = parse_args() if argv is None else parse_args(argv)
    merge = json.loads(Path(args.merge_json).read_text(encoding="utf-8"))
    changed = list(merge.get("changed_params", []))

    coverage: dict[str, Any] = {"inspected": False}
    if args.base_manifest and Path(args.base_manifest).is_file():
        manifest = json.loads(Path(args.base_manifest).read_text(
            encoding="utf-8"))
        entries = manifest if isinstance(manifest, list) \
            else manifest.get("weights", [])
        coverage = {"inspected": True, "manifest": args.base_manifest,
                    **manifest_coverage(changed, entries)}

    hosted = bundle_structure(Path(args.hosted_dir)
                              if args.hosted_dir else None)
    sdk = sdk_facts(Path(args.sdk_dir) if args.sdk_dir else None)

    checksums: dict[str, Any] = {"inspected": False}
    if args.base_dir and Path(args.base_dir).is_dir():
        base_headers = load_headers(Path(args.base_dir))
        merged_headers = load_headers(Path(args.merged_dir))
        changed_existing = [n for n in changed
                            if n in base_headers and n in merged_headers]
        base_sums = tensor_sha256(Path(args.base_dir), changed_existing)
        merged_sums = tensor_sha256(Path(args.merged_dir), changed_existing)
        differing = sorted(n for n in changed_existing
                           if base_sums.get(n) != merged_sums.get(n))
        checksums = {
            "inspected": True,
            "changed_tensors_hashed": len(changed_existing),
            "byte_identical": len(changed_existing) - len(differing),
            "byte_differing": len(differing),
            "differing_names": differing,
            "merged_sha256": {n: merged_sums[n] for n in differing},
        }

    audit = {}
    if Path(args.audit_json).is_file():
        audit = json.loads(Path(args.audit_json).read_text(encoding="utf-8"))

    runnable_adapted_exists = False  # no builder output; see verdict
    proof_of_use_met = False
    blocked = (not runnable_adapted_exists) or (not proof_of_use_met)

    payload = {
        "issue": 238,
        "stage": "g2_cactus_deploy_check",
        "adapter": args.merged_dir.replace("outputs/merged-model-",
                                           "outputs/lora-adapter-"),
        "merged_dir": args.merged_dir,
        "manifest_mapping": coverage,
        "hosted_bundle": hosted,
        "upstream_sdk": sdk,
        "changed_weight_checksums": checksums,
        "hf_audit_verdict": audit.get("verdict"),
        "hf_audit_changed": audit.get("changed_names", [])[:5],
        "proof_of_use_criterion": (
            "a runnable adapted Cactus artifact that generates text, with "
            "matched HF-vs-Cactus adaptation-sensitive probes "
            "(same prompts, same greedy decoding) showing the adaptation "
            "delta on both runtimes"),
        "proof_of_use_note": ADAPTER_SENSITIVE_PROBE_NOTE,
        "proof_of_use_met": proof_of_use_met,
        "verdict": "BLOCKED" if blocked else "PASS",
        "custom_deployment": ("BLOCKED" if blocked
                              else "available via supported path"),
        "upstream_requirements_ordered": [
            "R1: Cactus-Compute publishes a -cqN runnable bundle for "
            "Qwen3.5-2B (the v2.2.1 CLI resolver only matches -cqN; the "
            "hosted -int4/-int8 zips are legacy flat weights with no "
            "components/manifest.json, which the engine requires). Then: "
            "re-run G1 stock baseline, convert the G2 adapter with "
            "`cactus convert <base> --lora <adapter>`, and run matched "
            "HF-vs-Cactus probes.",
            "R2 (alternative): upstream qwen3_5 export-patch profile in "
            "the Crassus transpiler fixing the torch.export guard on "
            "_update_linear_attn_mask (transformers modeling_qwen3_5.py) "
            "so a local `cactus convert` graph build succeeds; the "
            "supported packaging path is then `cactus convert "
            "Qwen/Qwen3.5-2B --lora <adapter>`. Comment #380 local-graph "
            "failure remains binding until new evidence.",
        ],
        "weight_replacement_note": (
            "Splicing adapted bytes into hosted flat *.weights files would "
            "change files without a runtime that consumes them: the hosted "
            "layout has no graph/manifest, the v2.2.1 CLI cannot fetch it, "
            "and the engine refuses it (G1 evidence). A changed file is not "
            "proof of runtime use; no replacement was attempted or assumed."),
        "stock_benchmarking": (
            "stands independently on the HF runtime (transformers 5.5.0); "
            "the deployment BLOCKED verdict gates only the on-device "
            "comparison leg."),
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    # BLOCKED is a recorded verdict, not a script error: always exit 0 once
    # the evidence payload is written.
    return 0


if __name__ == "__main__":
    sys.exit(main())
