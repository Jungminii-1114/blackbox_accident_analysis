"""Real-video CPU QA; no claim to reproduce the CUDA evaluator."""
import argparse
from pathlib import Path
import tempfile
import time
import zipfile

import numpy as np
import pandas as pd
import torch

import focus12_lab as focus
import portfolio_lab as lab


def main():
    p=argparse.ArgumentParser();p.add_argument('--fixture',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();start=time.monotonic();torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix='focus12_cpu_') as directory:
        root=Path(directory);focus.PARENT=root
        config={'stage1_slots':5,'stage2_family':'B','stage3':focus.HISTORY['stage3_frozen']}
        (root/'inference.py').write_text(lab.make_source(config))
        (root/'candidate.py').write_text(focus.source({'stage1':{'family':'baseline3'},'stage2':'parent_B'}))
        m=lab.load_module(root/'candidate.py');m._device=lambda:torch.device('cpu')
        model=m.f12_backbone(lab.ROOT/'model/stage1')
        labels=pd.read_csv(lab.ROOT/'fixtures/baseline_data/stage1/labels.csv')
        groups=labels.path.map(lambda v:Path(v).stem).to_numpy()
        y=np.where(labels.label=='RERECORDED',1.,-1.)
        clean={'A':[],'B':[]};noise={'A':[],'B':[]}
        for row in labels.itertuples():
            print('CPU S1 features',row.ID,flush=True)
            path=args.fixture/'input/stage1/videos'/(row.ID+Path(row.path).suffix)
            a=m.f12_features(path,model);b=m.f12_features(path,model,90)
            for family in clean:
                clean[family].append(a[family]);noise[family].append(b[family])
        del model
        clean={k:np.stack(v) for k,v in clean.items()};noise={k:np.stack(v) for k,v in noise.items()}
        result=focus.s1_nested(clean,noise,y,groups)
        _,family,penalty=focus.choose_head(clean,y,groups)
        head=focus.ridge_fit(clean[family],y,penalty)
        np.testing.assert_allclose(m.f12_head_predict(clean[family],head),focus.ridge_predict(clean[family],head))
        m.FOCUS12_CONFIG['stage1']={'family':family,'head':head}
        exported=m.predict_stage1(args.fixture/'input/stage1',lab.ROOT/'model/stage1')
        reference=dict(zip(labels.ID,focus.as_labels(focus.ridge_predict(clean[family],head))))
        assert all(reference[r.ID]==r.answer for r in exported.itertuples())
        s2labels=pd.read_csv(args.fixture/'labels/stage2.csv')
        times=pd.read_csv(args.fixture/'labels/stage2_frame_time.csv')
        mapping={(r.ID,int(r.frame)):r.time_seconds for r in times.itertuples()}
        predictions={f:{q:{} for q in ('clean','95','90')} for f in ('E','C','D')}
        for row in s2labels.itertuples():
            for q in ('clean','95','90'):
                numbers,features=m.f12_event_features(args.fixture/'input/stage2/images'/row.ID,None if q=='clean' else int(q))
                parent_numbers,parent_features=m.f12_parent_event_features(args.fixture/'input/stage2/images'/row.ID,None if q=='clean' else int(q))
                if q=='clean':
                    n,f=m.pf_event_features(args.fixture/'input/stage2/images'/row.ID)
                    np.testing.assert_array_equal(n,parent_numbers);np.testing.assert_allclose(f,parent_features)
                anchor=int(parent_numbers[np.argmax(m.pf_event_scores(parent_features,'B'))])
                for family2 in predictions:
                    index=m.f12_refine(numbers,features,anchor) if family2=='E' else m.f12_event_index(features,family2)
                    predictions[family2][q][row.ID]=int(numbers[index])
        metrics={f:{q:focus.event_metrics(v,s2labels,mapping) for q,v in versions.items()} for f,versions in predictions.items()}
        # Stage3 function behavior is preserved on all real fixture videos.
        original=lab.load_module(root/'inference.py')
        pd.testing.assert_frame_equal(original.predict_stage3(args.fixture/'input/stage3',lab.ROOT/'model/stage3'),
                                      m.predict_stage3(args.fixture/'input/stage3',lab.ROOT/'model/stage3'))
        (root/'export.py').write_text(focus.source({'stage1':{'family':family,'head':head},'stage2':'E'}))
        with zipfile.ZipFile(root/'submit.zip','w',zipfile.ZIP_STORED) as archive:
            archive.write(root/'export.py','inference.py')
            archive.write(lab.ROOT/'requirements.txt','requirements.txt')
            for name in lab.MODEL_FILES:
                archive.write(lab.ROOT/name,name)
        lab.run([lab.sys.executable,lab.ROOT/'tools/validate_submit_package.py','--zip',root/'submit.zip','--import-check'])
        lab.save(args.output,{'status':'CPU_QA_PASS','cuda_full_package':'NOT_RUN', 'seconds':time.monotonic()-start,
            'stage1_oof':result,'stage1_final_family':family,'stage2':metrics,'source':focus.code_hashes(),
            'checks':['real ten S1 videos clean/JPEG90','exported S1 predictions match fitted features',
                      'source-group nested CV','S2 clean/JPEG95/JPEG90 on five videos','Stage3 exact parent equality on all fixture rows',
                      'fresh extracted ZIP subprocess import/layout']})
        print('QA saved:',args.output,'S1 OOF:',result['macro_f1'],flush=True)


if __name__=='__main__':
    main()
