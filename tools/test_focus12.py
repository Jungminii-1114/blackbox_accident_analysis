"""CPU tests for group exclusion, time metrics, source preservation and export."""
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

import focus12_lab as focus
import portfolio_lab as lab
import local_validate


class FocusTests(unittest.TestCase):
    def test_group_exclusion(self):
        groups = np.repeat(np.arange(5).astype(str),2)
        y = np.tile([-1.,1.],5)
        x = np.random.default_rng(5).normal(size=(10,12))
        first = focus.s1_nested({'A':x},{'A':x},y,groups,('A',))
        altered = y.copy(); altered[:2] *= -1
        second = focus.s1_nested({'A':x},{'A':x},altered,groups,('A',))
        np.testing.assert_allclose(first['margins'][:2], second['margins'][:2])
        self.assertEqual(first['jpeg90_agreement'],1.)
        self.assertNotIn('0',first['folds'][0]['train_sources'])

    def test_missing_time_never_means_frame_seconds(self):
        self.assertIsNone(local_validate._as_time('x',30,{},None))
        result = local_validate._event_accuracy({('x',):{'collision_frame':'30'}}, {('x',):{'collision_frame':'30'}},
                                               ['collision_frame'],'collision_frame',{},None)
        self.assertIsNone(result['accuracy_at_0_3s'])

    def test_same_accuracy_different_mae(self):
        target = {('x',):{'collision_frame':'30'}}
        a = local_validate._event_accuracy(target,{('x',):{'collision_frame':'30'}},['collision_frame'],'collision_frame',{},10.)
        b = local_validate._event_accuracy(target,{('x',):{'collision_frame':'33'}},['collision_frame'],'collision_frame',{},10.)
        self.assertEqual(a['accuracy_at_0_3s'],b['accuracy_at_0_3s'])
        self.assertLess(a['mae_seconds'],b['mae_seconds'])

    def test_export_and_head_equality(self):
        old_parent = focus.PARENT
        config = {'stage1_slots':5,'stage2_family':'B','stage3':focus.HISTORY['stage3_frozen']}
        parent = lab.make_source(config)
        self.assertEqual(hashlib.sha256(parent.encode()).hexdigest(),focus.HISTORY['parent_source_sha256'])
        try:
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);focus.PARENT=root
                (root/'inference.py').write_text(parent)
                x=np.random.default_rng(1).normal(size=(6,12)); y=np.array([-1,1]*3)
                head=focus.ridge_fit(x,y,.1)
                candidate=focus.source({'stage1':{'family':'A','head':head},'stage2':'C'})
                self.assertTrue(candidate.startswith(parent))
                self.assertEqual(candidate.count('predict_stage3 = pf_predict_stage3'),1)
                (root/'candidate.py').write_text(candidate)
                m=lab.load_module(root/'candidate.py')
                np.testing.assert_allclose(m.f12_head_predict(x,head),focus.ridge_predict(x,head))
                self.assertEqual(m.FOCUS12_CONFIG['stage1']['family'],'A')
                np.testing.assert_allclose(m.f12_rank([2.,2.,1.]),[.75,.75,0.])
                self.assertEqual(m.f12_event_index(np.zeros((1,3)),'C'),0)
                self.assertEqual(m.f12_refine(np.array([42]),np.zeros((1,3)),42),0)
        finally:
            focus.PARENT=old_parent

    def test_unlabeled_dense_rows_not_errors(self):
        import csv
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            columns=['ID','sample_index','accel_label','steer_label']
            for name,rows in [('labels', [['x',0,'CONSTANT','STRAIGHT']]),
                              ('prediction',[['x',0,'CONSTANT','STRAIGHT'],['x',1,'CONSTANT','STRAIGHT']])]:
                with (root/(name+'.csv')).open('w',newline='') as handle:
                    writer=csv.writer(handle);writer.writerow(columns);writer.writerows(rows)
            result=local_validate.score_stage3(root/'labels.csv',root/'prediction.csv')
            self.assertEqual(result['unexpected_predictions'],0)
            self.assertEqual(result['unlabeled_predictions_same_video'],1)


if __name__=='__main__':
    unittest.main()
