"""Open held-out test labels only after the validation-selected run is complete.

No thresholds, checkpoints, hyperparameters or parsers are selected here.
"""
import copy,hashlib,json,time
from pathlib import Path
import torch
from room_model import load_model
from data import predict,score

ROOT=Path('training/private')
protocol=json.loads(Path('training/protocol.json').read_text(encoding='utf-8'))
trained=json.loads((ROOT/'training-result.json').read_text(encoding='utf-8'))
assert trained['protocolSha256']==hashlib.sha256(Path('training/protocol.json').read_bytes()).hexdigest()
assert trained['checkpointSha256']==hashlib.sha256((ROOT/'best.pth').read_bytes()).hexdigest()
torch.set_num_threads(2);device=torch.device('cuda:0')
test=json.loads((ROOT/'test.json').read_text(encoding='utf-8'))
train=json.loads((ROOT/'train.json').read_text(encoding='utf-8'))
assert set(e['source'] for e in test)==set(protocol['testImages'])
assert not {e['roiSha256'] for e in train}&{e['roiSha256'] for e in test if e.get('roiSha256')}
before_model=load_model(ROOT/'english_g2.pth',device)
after_model=load_model(ROOT/'best.pth',device)
start=time.perf_counter();before=predict(before_model,test,device);before_ms=(time.perf_counter()-start)*1000
start=time.perf_counter();after=predict(after_model,test,device);after_ms=(time.perf_counter()-start)*1000
old=[e['tesseract'].upper() for e in test]
train_labels={e['label'] for e in train}
novel=[i for i,e in enumerate(test) if e['label'] not in train_labels]
metrics={name:score(test,preds) for name,preds in [('tesseractBaseline',old),('sameModelBeforeTraining',before),('validationSelectedAfterTraining',after)]}
novel_metrics={name:score([test[i] for i in novel],[preds[i] for i in novel]) for name,preds in [('tesseractBaseline',old),('sameModelBeforeTraining',before),('validationSelectedAfterTraining',after)]}
report={'experiment':protocol['experiment'],'scope':'Room-code recognizer only; course/geometry/axis unchanged. Same automatically extracted inputs for pretrained vs fine-tuned weights.','split':{'trainImages':protocol['trainImages'],'validationImages':protocol['validationImages'],'testImages':protocol['testImages'],'trainingUniqueCrops':len(train),'validationRecords':9,'testRecords':len(test),'testNovelCodeRecords':len(novel),'testDistinctCodes':len({e['label'] for e in test})},'selectedEpoch':trained['selectedEpoch'],'lastEpoch':trained['lastEpoch'],'metrics':metrics,'novelCodeMetrics':novel_metrics,'speedGpuMs':{'before':before_ms,'after':after_ms,'note':'Model loading and crop/Tesseract time excluded; GPU kernels may include first-run initialization. Not a browser benchmark.'},'training':{k:v for k,v in trained.items() if k!='history'},'limitations':['Only two final-test screenshots, correlated courses; not a generalization guarantee.','All supplied images were previously visible during baseline program development. Held out only from this fine-tuning/model-selection run.','Validation is a single screenshot with nine codes.','English model is public pretrained; only new fine-tuning pixels/labels come from supplied screenshots.','Removing duplicate crops and augmenting training only does not create more independent data.','Exact-code accuracy, not full classroom description accuracy.'],'rows':[{**e,'before':b,'after':a}for e,b,a in zip(test,before,after)]}
# End-to-end comparison on the same test images. Only the roomCode field changes.
evaluation=json.loads(Path('samples/private/evaluation.json').read_text(encoding='utf-8'))
end_to_end={name:{'nameCorrect':0,'roomCorrect':0,'roomTotal':0,'allCorrect':0,'allTotal':0,'total':0}for name in ['tesseractBaseline','sameModelBeforeTraining','validationSelectedAfterTraining']}
replacement={name:{(e['source'],e['baselineIndex']):p for e,p in zip(test,preds)}for name,preds in [('sameModelBeforeTraining',before),('validationSelectedAfterTraining',after)]}
for sample in evaluation['samples']:
    if sample['id'] not in protocol['testImages']:continue
    baseline=json.loads(Path(f"samples/private/results/{sample['id']}.json").read_text(encoding='utf-8'))
    for row in sample['rows']:
        expected=row['expected'];pred=row['predicted']
        index=next((i for i,c in enumerate(baseline['classes'])if pred and c['day']==pred['day'] and c['startMinutes']==pred['startMinutes']),None)
        for name,totals in end_to_end.items():
            totals['total']+=1;totals['nameCorrect']+=bool(row['correct']['name'])
            room=pred['roomCode'] if pred else ''
            if name!='tesseractBaseline':room=replacement[name].get((sample['id'],index),room)
            room_ok=expected['roomCode'] is not None and room.upper()==expected['roomCode'].upper()
            if expected['roomCode'] is not None:totals['roomTotal']+=1;totals['roomCorrect']+=room_ok
            if all(v is not None for v in expected.values()):
                totals['allTotal']+=1
                totals['allCorrect']+=bool(room_ok and all(row['correct'][f] for f in ['name','day','startMinutes','endMinutes']))
report['endToEndOnTest']=end_to_end
(ROOT/'test-comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
public={k:v for k,v in report.items() if k!='rows'}
Path('evaluation/finetune-comparison.json').write_text(json.dumps(public,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
Path('evaluation/finetune-history.json').write_text(json.dumps(trained['history'],indent=2)+'\n',encoding='utf-8')
print(json.dumps(public,ensure_ascii=False,indent=2),flush=True)
