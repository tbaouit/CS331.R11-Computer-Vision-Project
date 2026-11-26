"""Report comparable SD2 experiments (Baseline vs. Proposed) on a fixed manifest."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.profiles import load_evaluation, verified_jobs
from src.eval.visualize import comparison_grid
from src.evaluation.reporting import aggregate, write_json
from src.evaluation.runner import read_mask, read_rgb
import numpy as np


from src.pipelines.backend import validate_experiment_config

ROLES = ("baseline", "proposed")


def check_configs(configs):
    if set(configs) != set(ROLES):
        raise ValueError("Expected baseline and proposed")
    for role, config in configs.items():
        if validate_experiment_config(config) != role or config.get("seed") != 42:
            raise ValueError(f"Inconsistent configuration for {role}")


# Tách tỷ lệ PASS theo lượt và theo mẫu; Baseline để trống vì không dùng critic.
def critic_summary(selected):
    counts, invalid = Counter(), 0
    reviewed = passed = 0
    for _, result, _ in selected.values():
        history = result.get("critique_history", [])
        if history:
            reviewed += 1
            passed += int(bool(result.get("critic_accepted")))
        for attempt in history:
            decision = attempt["final_decision"]
            counts["REJECT" if decision == "FAIL" else decision] += 1
            invalid += int(attempt.get("valid_json") is False)
    judged = counts["PASS"] + counts["REJECT"]
    return dict(attempt_decisions=dict(counts), invalid_json=invalid, reviewed_samples=reviewed,
                sample_pass_rate=passed / reviewed if reviewed else None,
                attempt_pass_rate=counts["PASS"] / judged if judged else None,
                attempt_reject_rate=counts["REJECT"] / judged if judged else None)


# Kiểm tra hồ sơ đầu vào, ghép metric từng vật thể, xuất CSV/JSON và lưới ảnh.
def compare_versions(records, profiles, output, fid_reference=None, device="cuda"):
    if not records or len({r["sample_id"] for r in records}) != len(records):
        raise ValueError("Expected a nonempty manifest with unique sample IDs")
    if set(profiles) != set(ROLES) or len(set(profiles.values())) != len(profiles):
        raise ValueError("Provide distinct profiles for baseline and proposed")
    jobs, reports, configs = {}, {}, {}
    protocol = None
    for key, profile in profiles.items():
        directory, rows, settings = load_evaluation(profile, records)
        if protocol is not None and protocol != settings:
            raise ValueError("Metric protocols differ; rerun evaluation in the same runtime")
        protocol = settings
        if any(row.get("status") != "completed" for row in rows.values()):
            raise ValueError(f"Incomplete metrics for {key}")
        jobs[key] = verified_jobs(profile, records, directory, rows)
        reports[key] = rows
        configurations = [result["config"] for _, result, _ in jobs[key].values()]
        configs[key] = configurations[0]
        if any(cfg != configs[key] for cfg in configurations):
            raise ValueError(f"Mixed configs in {key}")
    check_configs(configs)
    paired = []
    for record in records:
        sid = record["sample_id"]
        base, _, hashes = jobs["baseline"][sid]
        entry = {"sample_id": sid, "category": record.get("name"), "hidden_ratio": record.get("hidden_ratio")}
        for key in profiles:
            job, result, current = jobs[key][sid]
            auto_mode = configs[key].get("visible_mask_source") == "auto" or configs["baseline"].get("visible_mask_source") == "auto"
            check_names = ("input.png", "visible.png", "occluded_input.png") if auto_mode else (
                "input.png", "visible.png", "occluded_input.png", "amodal.png"
            )
            for name in check_names:
                read = read_rgb if name == "input.png" else read_mask
                if current[name] != hashes[name] and not np.array_equal(read(job / name), read(base / name)):
                    raise ValueError(f"Input or mask differs: {key}/{sid}/{name}")
            entry.update({f"{key}_{metric}": value for metric, value in reports[key][sid]["metrics"].items()})
            entry[f"{key}_seconds"] = result.get("seconds")
        paired.append(entry)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_sample.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in paired for k in row)))
        writer.writeheader()
        writer.writerows(paired)
    summary = dict(profiles={k: p.name for k, p in profiles.items()}, configs=configs,
                   requested=len(records), protocol=protocol,
                   metrics={k: aggregate(list(rows.values()))["metrics"] for k, rows in reports.items()},
                   critic={k: critic_summary(jobs[k]) for k in profiles},
                   previews=comparison_grid(output, records, jobs),
                   fid=None)
    if fid_reference:
        from src.eval.fid import evaluate_fid
        summary["fid"] = evaluate_fid(records, jobs, fid_reference, device)
    write_json(output / "comparison.json", summary)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "metric", "mean", "std", "n"])
        for key, metrics in summary["metrics"].items():
            for metric, stat in metrics.items():
                writer.writerow([key, metric, stat["mean"], stat["std"], stat["n"]])
            critic = summary["critic"][key]
            for metric in ("attempt_pass_rate", "attempt_reject_rate", "sample_pass_rate"):
                n = critic["reviewed_samples"] if metric.startswith("sample") else sum(
                    critic["attempt_decisions"].get(decision, 0) for decision in ("PASS", "REJECT"))
                writer.writerow([key, metric, critic[metric], "", n])
            if summary["fid"]:
                writer.writerow([key, "FID_192", summary["fid"]["scores"][key], "", len(records)])
    return summary


# Đọc đường dẫn manifest, profile và thư mục ảnh sạch nếu bật FID.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    for role in ROLES:
        parser.add_argument("--" + role.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fid-reference", type=Path, help="Folder of clean, isolated, unoccluded RGB crops")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    records = json.loads(args.manifest.read_text(encoding="utf-8"))
    profiles = {key: getattr(args, key) for key in ROLES if getattr(args, key)}
    compare_versions(records, profiles, args.output, args.fid_reference, args.device)
    print("Comparison:", args.output)


if __name__ == "__main__":
    main()
