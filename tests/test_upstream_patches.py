import ast
import hashlib
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]

class PatchTests(unittest.TestCase):
    def test_snapshot_checksums_and_runtime(self):
        provenance=json.loads((ROOT/"upstream/provenance.json").read_text())
        self.assertEqual(provenance["commit"],"92e5b392cba1c767f4ddc08497f6b4450743929a")
        for row in provenance["source_files"]:
            self.assertEqual(hashlib.sha256((ROOT/"upstream/SCAmodal"/row["path"]).read_bytes()).hexdigest(),row["sha256"])
        for row in provenance["runtime_files"]:
            self.assertEqual(hashlib.sha256((ROOT/"runtime/SCAmodal"/row["path"]).read_bytes()).hexdigest(),row["runtime_sha256"])
    def test_heuristic_fallback_raises(self):
        tree=ast.parse((ROOT/"runtime/SCAmodal/amodal_shape_predictor.py").read_text(encoding="utf-8"))
        method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=="_heuristic_fallback")
        function=ast.FunctionDef(name="fallback",args=method.args,body=method.body,decorator_list=[])
        module=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))
        ns={};exec(compile(module,"fallback","exec"),ns)
        with self.assertRaisesRegex(RuntimeError,"disabled"):ns["fallback"](None,None)
    def test_constructor_checks_precede_vlm(self):
        text=(ROOT/"runtime/SCAmodal/amodal_completer.py").read_text(encoding="utf-8")
        self.assertLess(text.index("Pix2Gestalt initialization failed"),text.index("VLMReasoner(device"))
    def test_critic_has_no_default_five(self):
        text=(ROOT/"runtime/SCAmodal/vlm_reasoner.py").read_text(encoding="utf-8")
        self.assertNotIn("default=5.0",text)
        self.assertIn("Critic response missing",text)

if __name__=="__main__":unittest.main()
