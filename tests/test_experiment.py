import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from src.evaluation.runner import run_one, crop_original

class FakeBackend:
    """Synthetic outputs for orchestration tests only; never used by the production CLI."""
    def __init__(self, fail=False):
        self.calls=0
        self.fail=fail
    def begin_sample(self,job): self.job=job
    def complete(self,image,visible,label):
        self.calls+=1
        if self.fail: raise RuntimeError("synthetic stage failure")
        assert isinstance(label,str) # No GT argument is available in this interface.
        mask=visible.copy();mask[2:8,3:9]=True
        rgb=np.full_like(image,170);rgb[visible]=image[visible]
        rgba=np.dstack([rgb,mask.astype(np.uint8)*255])
        return dict(inpainted_rgba=rgba,amodal_mask=mask,critique_history=[{"score":8.0}],
                    final_score=8.0,_trace={"pad_info":{},"backend":"synthetic unit test"})

class ExperimentTests(unittest.TestCase):
    def sample(self,root):
        image=np.full((12,15,3),90,np.uint8)
        visible=np.zeros((12,15),np.uint8);visible[3:6,4:7]=255
        gt=visible.copy();gt[2:8,3:9]=255
        for name,array in [("image.png",image),("visible.png",visible),("amodal.png",gt)]:
            Image.fromarray(array).save(root/name)
        return dict(sample_id="train_1",split="train",image="image.png",visible_mask="visible.png",
                    gt_amodal="amodal.png",name="object")
    def test_outputs_resume_and_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);record=self.sample(root);backend=FakeBackend()
            cfg={"critique_threshold":7,"appearance_metrics":False}
            result,skipped=run_one(record,root,root/"out",cfg,backend,"profile")
            self.assertFalse(skipped);self.assertEqual(result["metrics"]["amodal_iou"],1)
            again,skipped=run_one(record,root,root/"out",cfg,backend,"profile")
            self.assertTrue(skipped);self.assertEqual(backend.calls,1)
            job=next((root/"out/train_1").iterdir())
            (job/"completion.png").write_bytes(b"corrupt")
            run_one(record,root,root/"out",cfg,backend,"profile")
            self.assertEqual(backend.calls,2)
    def test_failure_does_not_mark_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);record=self.sample(root)
            with self.assertRaisesRegex(RuntimeError,"synthetic"):
                run_one(record,root,root/"out",{"critique_threshold":7},FakeBackend(True),"p")
            self.assertEqual(len(list((root/"out").rglob("failure.json"))),1)
            self.assertFalse(list((root/"out").rglob("result.json")))
    def test_gt_is_opened_after_backend(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);record=self.sample(root)
            (root/"amodal.png").unlink();backend=FakeBackend()
            with self.assertRaises(FileNotFoundError):
                run_one(record,root,root/"out",{"critique_threshold":7},backend,"p")
            self.assertEqual(backend.calls,1)
    def test_padding_crop(self):
        array=np.arange(6*9).reshape(6,9)
        expected=array[1:5,2:7]
        np.testing.assert_array_equal(crop_original(array,dict(pad_top=1,pad_bottom=1,pad_left=2,pad_right=2),(4,5)),expected)
        with self.assertRaises(ValueError):crop_original(array,{},(4,5))

if __name__=="__main__":unittest.main()
