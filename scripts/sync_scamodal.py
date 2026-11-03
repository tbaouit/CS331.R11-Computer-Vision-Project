"""Snapshot the author repo and build explicitly patched runtime copies."""
import argparse, ast, hashlib, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
CORE = ["amodal_completer.py", "amodal_shape_predictor.py", "vlm_reasoner.py", "metrics_utils.py", "segmenter.py"]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"Patch anchor changed: {old[:80]}")
    return text.replace(old, new, 1)


def patched(name, text):
    if name == "amodal_shape_predictor.py":
        tree = ast.parse(text)
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_heuristic_fallback")
        lines = text.splitlines(keepends=True)
        lines[method.lineno-1:method.end_lineno] = [
            "    def _heuristic_fallback(self, visible_mask):\n",
            "        raise RuntimeError('Strict SCAmodal: Pix2Gestalt failed; heuristic fallback is disabled. See pipeline.log.')\n"]
        text = "".join(lines)
        text = replace_once(text, "from PIL import Image", "from PIL import Image\nfrom pathlib import Path")
        text = replace_once(text, "            result_pil = Image.fromarray(result_np)",
            "            if os.environ.get('SCAMODAL_RUN_DIR'):\n"
            "                Image.fromarray(result_np).save(Path(os.environ['SCAMODAL_RUN_DIR']) / 'pix2gestalt_rgb_256.png')\n"
            "            result_pil = Image.fromarray(result_np)")
        text = replace_once(text, "            alpha_channel = amodal_rgba[:, :, 3]",
            "            alpha_channel = amodal_rgba[:, :, 3]\n"
            "            if os.environ.get('SCAMODAL_RUN_DIR'):\n"
            "                Image.fromarray(alpha_channel).save(Path(os.environ['SCAMODAL_RUN_DIR']) / 'carvekit_alpha_256.png')")
    if name == "amodal_completer.py":
        text = replace_once(text,
            "            AmodalCompleter._shape_predictor = Pix2GestaltPredictor(device=self.device)",
            "            AmodalCompleter._shape_predictor = Pix2GestaltPredictor(device=self.device)\n"
            "            if AmodalCompleter._shape_predictor.model is None:\n"
            "                raise RuntimeError('Strict SCAmodal: Pix2Gestalt initialization failed.')")
        text = replace_once(text, "            AmodalCompleter._vlm = VLMReasoner(device=self.device)",
            "            AmodalCompleter._vlm = VLMReasoner(device=self.device)\n"
            "            if AmodalCompleter._vlm.model is None:\n"
            "                raise RuntimeError('Strict SCAmodal: Qwen initialization failed.')")
    if name == "metrics_utils.py":
        text = replace_once(text, "ssim(img1, img2, full=True, channel_axis=2)",
                            "ssim(img1, img2, full=True, channel_axis=2, data_range=255)")
    if name == "vlm_reasoner.py":
        text = replace_once(text, '            return "ERROR: Model not loaded."',
                            '            raise RuntimeError("Qwen model is not loaded")')
        start = text.index("        def _parse(label, default=5.0):")
        end = text.index('        structural = _parse("STRUCTURAL")', start)
        text = text[:start] + (
            "        def _parse(label):\n"
            "            match = re.search(rf'{label}\\s*:\\s*([\\d.]+)', output_text, re.IGNORECASE)\n"
            "            if not match:\n"
            "                raise ValueError(f'Critic response missing {label}: {output_text}')\n"
            "            value = float(match.group(1))\n"
            "            if not 0 <= value <= 10:\n"
            "                raise ValueError(f'Critic score out of range: {label}={value}')\n"
            "            return value\n\n") + text[end:]
    compile(text, name, "exec")
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("D:/SCAmodal"))
    args = parser.parse_args()
    source = args.source.resolve()
    snapshot = ROOT / "upstream/SCAmodal"
    runtime = ROOT / "runtime/SCAmodal"
    ref = source / ".git/refs/heads/main"
    commit = ref.read_text().strip() if ref.exists() else "unknown"
    entries = []
    # Preserve all source/docs/subset lists, not .git or any model weights.
    for path in sorted(source.rglob("*")):
        if not path.is_file() or ".git" in path.relative_to(source).parts:
            continue
        if path.suffix not in (".py", ".md", ".txt", ".json", ".html") and path.name != ".gitignore":
            continue
        rel = path.relative_to(source)
        target = snapshot / rel
        original = path.read_bytes()
        if target.exists() and target.read_bytes() != original:
            raise RuntimeError(f"Snapshot already differs; preserve history rather than overwrite: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(original)
        entries.append(dict(path=rel.as_posix(), sha256=sha(original)))
    runtime.mkdir(parents=True, exist_ok=True)
    patched_files = []
    for name in CORE:
        original = (snapshot/name).read_text(encoding="utf-8")
        result = patched(name, original)
        (runtime/name).write_text(result, encoding="utf-8")
        patched_files.append(dict(path=name, original_sha256=sha((snapshot/name).read_bytes()),
                                  runtime_sha256=sha((runtime/name).read_bytes())))
    record = dict(repo="https://github.com/dpduy123/SCAmodal", commit=commit,
                  source_files=entries, runtime_files=patched_files,
                  changes=["disable heuristic fallback; reject missing Pix2Gestalt/Qwen",
                           "save Pix2Gestalt RGB and CarveKit alpha",
                           "reject missing/out-of-range critic scores rather than default 5",
                           "no change to generation prompt, mask processing, inpainting or critic loop"])
    (ROOT/"upstream/provenance.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
    print("Snapshot/runtime prepared:", commit, len(entries), "source files")


if __name__ == "__main__":
    main()
