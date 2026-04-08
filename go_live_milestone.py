import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from go_live_weekly_report import build_weekly_report


ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"


def _campaign_name(args_campaign_name: str) -> str:
    raw = args_campaign_name.strip() if args_campaign_name else os.getenv("QUANT_PAPER_CAMPAIGN_NAME", "default")
    raw = raw.strip().lower()
    return raw or "default"


def _campaign_paths(campaign_name: str) -> tuple[Path, Path, Path, Path]:
    if campaign_name == "default":
        return (
            LOGS / "go_live_phase_state.json",
            LOGS / "go_live_phase_timeline.jsonl",
            LOGS / "go_live_final_decision.json",
            LOGS / "go_live_weekly_report.json",
        )

    return (
        LOGS / f"go_live_phase_state_{campaign_name}.json",
        LOGS / f"go_live_phase_timeline_{campaign_name}.jsonl",
        LOGS / f"go_live_final_decision_{campaign_name}.json",
        LOGS / f"go_live_weekly_report_{campaign_name}.json",
    )


def _append_timeline(timeline_path: Path, entry: dict) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    with timeline_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def _save_state(state_path: Path, entry: dict) -> None:
    state_path.write_text(json.dumps(entry, indent=2), encoding="utf-8")


def _utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def handle_phase_start(phase: str, label: str, campaign_name: str) -> dict:
    state_path, timeline_path, _, _ = _campaign_paths(campaign_name)
    entry = {
        "kind": "phase-start",
        "campaign_name": campaign_name,
        "phase": phase,
        "label": label,
        "timestamp_utc": _utc_now(),
        "state": "started",
    }
    _save_state(state_path, entry)
    _append_timeline(timeline_path, entry)
    return entry


def handle_review(label: str, campaign_name: str, final_decision: bool = False) -> dict:
    _, timeline_path, final_decision_path, report_path = _campaign_paths(campaign_name)
    report = build_weekly_report(campaign_name=campaign_name)
    entry = {
        "kind": "final-decision" if final_decision else "review",
        "campaign_name": campaign_name,
        "label": label,
        "timestamp_utc": _utc_now(),
        "report_path": str(report_path.resolve()),
        "weekly_pass": report["checks"]["weekly_pass"],
        "ready_for_phase_b": report["checks"]["ready_for_phase_b"],
        "consecutive_weekly_passes": report["checks"]["consecutive_weekly_passes"],
        "required_for_phase_b": report["checks"]["required_for_phase_b"],
        "model_weekly_equity": report["metrics"]["model_weekly_equity"],
        "model_vs_random_edge": report["metrics"]["model_vs_random_edge"],
        "model_hit_rate": report["metrics"]["model_hit_rate"],
        "max_weekly_drawdown": report["metrics"]["max_weekly_drawdown"],
        "max_consecutive_loss_days": report["metrics"]["max_consecutive_loss_days"],
    }
    if final_decision:
        final_decision_path.write_text(json.dumps(entry, indent=2), encoding="utf-8")
        entry["final_decision_path"] = str(final_decision_path.resolve())
    _append_timeline(timeline_path, entry)
    return entry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a go-live milestone or run a scheduled review.")
    parser.add_argument("--kind", choices=["phase-start", "review", "final-decision"], required=True)
    parser.add_argument("--phase", default="")
    parser.add_argument("--label", default="")
    parser.add_argument("--campaign-name", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    campaign_name = _campaign_name(args.campaign_name)
    state_path, _, final_decision_path, _ = _campaign_paths(campaign_name)

    if args.kind == "phase-start":
        if not args.phase:
            raise ValueError("--phase is required for phase-start milestones")
        entry = handle_phase_start(args.phase, args.label or args.phase, campaign_name=campaign_name)
        print(f"Recorded phase start: {entry['phase']} -> {entry['label']}")
        print(f"Campaign: {campaign_name}")
        print(f"State: {state_path}")
        return

    entry = handle_review(args.label or args.kind, campaign_name=campaign_name, final_decision=args.kind == "final-decision")
    print(f"Milestone completed: {entry['label']}")
    print(f"Campaign: {campaign_name}")
    print(f"Weekly pass: {entry['weekly_pass']}")
    print(f"Ready for Phase B: {entry['ready_for_phase_b']}")
    if args.kind == "final-decision":
        print(f"Final decision snapshot: {final_decision_path}")


if __name__ == "__main__":
    main()
