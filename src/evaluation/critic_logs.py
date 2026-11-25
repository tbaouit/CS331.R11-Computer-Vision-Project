# Phạm vi: proposed: tổng hợp lượt thử critic theo mẫu và signature; Baseline không có critic.
"""Resume-safe, compact batch critic log; raw Qwen responses remain per sample."""
import json
import os
from pathlib import Path


# Cập nhật log tổng hợp theo chữ ký mẫu; chạy lại không cộng trùng lượt cũ.
def update_critic_log(root, result):
    if result.get("config", {}).get("pipeline_mode") != "proposed":
        return
    path = Path(root) / "critic_logs.jsonl"
    existing = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.is_file() else []
    # Replace rows for this sample/signature on resume, rather than duplicating attempts.
    existing = [r for r in existing if not (r["sample_id"] == result["sample_id"] and r["signature"] == result["signature"])]
    for attempt in result.get("critique_history", []):
        row = {key: attempt.get(key) for key in ("attempt", "retries", "seed", "guidance_scale", "strength",
               "score", "color_match", "structure_continuity", "valid_json", "final_decision", "reason",
               "repair_instruction", "attempt_seconds", "synthesis_seconds")}
        row.update(sample_id=result["sample_id"], signature=result["signature"],
                   selected=attempt["attempt"] == result.get("trace", {}).get("selected_attempt"),
                   sample_outcome=result.get("trace", {}).get("outcome"))
        existing.append(row)
    partial = path.with_suffix(".jsonl.part")
    partial.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in existing), encoding="utf-8")
    os.replace(partial, path)
