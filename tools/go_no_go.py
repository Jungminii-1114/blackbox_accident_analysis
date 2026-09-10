"""Conservative submission gate based on external local-holdout metrics.

This does not predict hidden leaderboard scores.  It converts a single-stage
local improvement into a transparent, deliberately wide transfer range and
blocks in-sample/smoke metrics from being used as a submit signal.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


OFFICIAL_BSS = {"stage1": 0.5339789355, "stage2": 0.11721, "stage3": 0.14581}
WEIGHT = {"stage1": 0.2, "stage2": 0.4, "stage3": 0.4}
LOCAL_KEY = {"stage1": "macro_f1", "stage2": "score", "stage3": "score"}
MIN_DELTA = {"stage1": 0.015, "stage2": 0.030, "stage3": 0.030}


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    """Return None for an absent report so the notebook ends in a clear NO_GO."""
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def metric(report: Optional[Dict[str, Any]], stage: str) -> Optional[float]:
    if report is None:
        return None
    value = report.get(stage, {}).get(LOCAL_KEY[stage])
    return float(value) if isinstance(value, (int, float)) else None


def print_number(label: str, value: Optional[float]) -> None:
    print(f"{label}: {'N/A' if value is None else f'{value:.5f}'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Print conservative GO/NO_GO for one DACON 236753 Stage change.")
    parser.add_argument("--stage", choices=tuple(WEIGHT), required=True)
    parser.add_argument("--baseline-local", type=Path, required=True, help="local_validate JSON for frozen baseline")
    parser.add_argument("--candidate-local", type=Path, required=True, help="local_validate JSON for candidate")
    parser.add_argument("--evaluation-kind", choices=("external_group_holdout", "smoke", "in_sample"), required=True)
    parser.add_argument("--smoke-pass", action="store_true", help="set only after ZIP extraction/re-inference checks pass")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--strict-exit", action="store_true", help="return exit code 2 for NO_GO (useful in CI)")
    args = parser.parse_args()
    baseline, candidate = read_json(args.baseline_local), read_json(args.candidate_local)
    stage = args.stage
    local_baseline, local_candidate = metric(baseline, stage), metric(candidate, stage)
    local_delta = None if local_baseline is None or local_candidate is None else local_candidate - local_baseline
    official_total = sum(WEIGHT[name] * OFFICIAL_BSS[name] for name in WEIGHT)
    projected = None
    if local_delta is not None:
        # Before two or more historical local→public pairs exist, transfer is unknown.
        # Report a broad 25–75% transfer band rather than a fabricated point forecast.
        low = official_total + WEIGHT[stage] * local_delta * 0.25
        high = official_total + WEIGHT[stage] * local_delta * 0.75
        projected = {"low": low, "high": high, "weighted_lift_low": low - official_total, "weighted_lift_high": high - official_total}

    failures = []
    if args.evaluation_kind != "external_group_holdout":
        failures.append("Local metric is not an external group/route holdout.")
    if not args.smoke_pass:
        failures.append("ZIP extraction + second smoke inference has not passed.")
    if local_delta is None:
        missing = []
        if baseline is None:
            missing.append(str(args.baseline_local))
        if candidate is None:
            missing.append(str(args.candidate_local))
        suffix = f" Missing/invalid report: {', '.join(missing)}." if missing else ""
        failures.append("Stage score is unavailable in one of the local metric reports." + suffix)
    elif local_delta < MIN_DELTA[stage]:
        failures.append(f"Local {stage} lift {local_delta:+.5f} is below required +{MIN_DELTA[stage]:.3f}.")
    verdict = "NO_GO" if failures else "GO"
    confidence = "low" if verdict == "NO_GO" else "moderate"

    print("\n=== Submission gate ===")
    print(f"Target: {stage} only (other stages frozen)")
    print("\n[LOCAL BSS — external holdout required]")
    print_number("baseline", local_baseline)
    print_number("candidate", local_candidate)
    print_number("delta", local_delta)
    print("\n[OFFICIAL BSS — observed baseline]")
    for name in ("stage1", "stage2", "stage3"):
        print(f"{name}: {OFFICIAL_BSS[name]:.5f} (weight {WEIGHT[name]:.1f})")
    print(f"weighted baseline: {official_total:.5f}")
    if projected:
        print("\n[Projected official weighted score — heuristic, not observed]")
        print(f"range: {projected['low']:.5f}–{projected['high']:.5f}")
        print(f"weighted lift: {projected['weighted_lift_low']:+.5f}–{projected['weighted_lift_high']:+.5f}")
    print(f"\n{verdict} (confidence: {confidence})")
    if failures:
        for failure in failures:
            print(f"- {failure}")
    else:
        print(f"- Local lift clears the {MIN_DELTA[stage]:.3f} minimum and the ZIP smoke gate passed.")
        print("- Submit exactly this one Stage change; record the resulting official Stage score before another change.")
    result = {"verdict": verdict, "confidence": confidence, "stage": stage, "local_bss": {"baseline": local_baseline, "candidate": local_candidate, "delta": local_delta}, "official_bss": {**OFFICIAL_BSS, "weighted_baseline": official_total, "projected": projected}, "failures": failures}
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # A notebook's final information cell must display NO_GO without looking like
    # a crashed run. CI can opt into a non-zero result with --strict-exit.
    raise SystemExit(2 if verdict == "NO_GO" and args.strict_exit else 0)


if __name__ == "__main__":
    main()
