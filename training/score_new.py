"""Evaluation of additional supplied images with frozen models and rules."""
from pathlib import Path
import hashlib,json,time
import torch
import onnxruntime as ort
from room_model import load_model,decode
from data import image_tensor,predict

root=Path('training/private/new-check')
metadata=json.loads(Path('trained/room-code-v2.json').read_text(encoding='utf-8'))
model_path=Path('trained/room-code-v2.onnx')
before_hash=hashlib.sha256(model_path.read_bytes()).hexdigest()
assert before_hash==metadata['sha256'],'Frozen trained model has changed'
torch.set_num_threads(2)
inputs=json.loads((root/'inputs.json').read_text(encoding='utf-8'))
truth=json.loads((root/'truth.json').read_text(encoding='utf-8'))
entries=json.loads((root/'inference-inputs.json').read_text(encoding='utf-8'))
before_model=load_model('training/private/english_g2.pth','cpu')
before_predictions=predict(before_model,entries,'cpu')
options=ort.SessionOptions();options.intra_op_num_threads=2
session=ort.InferenceSession(str(model_path),options,providers=['CPUExecutionProvider'])
started=time.perf_counter();after_predictions=[]
for entry in entries:
    image=image_tensor(entry).unsqueeze(0).numpy()
    logits=session.run(None,{'image':image})[0]
    after_predictions+=decode(torch.from_numpy(logits)) if entry['roi'] else ['']
room_ms=(time.perf_counter()-started)*1000
room_lookup={(e['source'],e['baselineIndex']):{'before':b,'after':a,'roi':e['roi']}for e,b,a in zip(entries,before_predictions,after_predictions)}
normalize=lambda s:''.join(str(s or '').upper().split())
training_labels={e['label'] for e in json.loads(Path('training/private/train.json').read_text(encoding='utf-8'))}
results=[]
for sample in truth['samples']:
    baseline=json.loads((root/f"{sample['id']}-baseline.json").read_text(encoding='utf-8'))
    used=set();rows=[]
    metrics={engine:{f:{'correct':0,'total':0,'excluded':0}for f in ['name','roomCode','novelRoomCode','startMinutes','endMinutes','allFields']}for engine in ['tesseract','before','after']}
    for expected in sample['classes']:
        candidates=[(abs((p['startMinutes'] if p['startMinutes'] is not None else -10000)-expected['startMinutes']),i,p)for i,p in enumerate(baseline['classes'])if i not in used and p['day']==expected['day']]
        candidates.sort(key=lambda v:v[0]);pair=candidates[0] if candidates and candidates[0][0]<=45 else None
        if pair:_,i,p=pair;used.add(i)
        else:i=None;p=None
        room=room_lookup.get((sample['id'],i),{'before':'','after':'','roi':None})
        engines={engine:({**p,'roomCode':p['roomCode'] if engine=='tesseract' else room[engine]}if p else None)for engine in metrics}
        correctness={}
        for engine,pred in engines.items():
            checked={}
            for field in ['name','roomCode','startMinutes','endMinutes']:
                metric=metrics[engine][field]
                if expected[field] is None:metric['excluded']+=1;checked[field]=None;continue
                metric['total']+=1
                checked[field]=bool(pred and (pred[field] is not None and abs(pred[field]-expected[field])<=5 if field.endswith('Minutes') else normalize(pred[field])==normalize(expected[field])))
                metric['correct']+=checked[field]
            full=metrics[engine]['allFields']
            novel=metrics[engine]['novelRoomCode']
            if expected['roomCode'] and expected['roomCode'] not in training_labels:
                novel['total']+=1;novel['correct']+=bool(checked['roomCode'])
            else:novel['excluded']+=1
            if any(v is None for v in expected.values()):full['excluded']+=1
            else:full['total']+=1;full['correct']+=bool(pred and all(checked.values()) and pred['day']==expected['day'])
            correctness[engine]=checked
        rows.append({'expected':expected,'prediction':engines['after'],'originalTesseractRoom':p['roomCode'] if p else None,'beforeTrainingRoom':room['before'],'correctness':correctness})
    for engine in metrics.values():
        for m in engine.values():m['accuracy']=m['correct']/m['total'] if m['total'] else None
    info=next(i for i in inputs if i['id']==sample['id'])
    results.append({'id':sample['id'],'input':info,'expectedBlocks':len(sample['classes']),'predictedBlocks':len(baseline['classes']),'matchedBlocks':len(used),'spuriousBlocks':len(baseline['classes'])-len(used),'untimedSubjectsNotHandled':sample['untimedSubjects'],'baselineMs':baseline['elapsedMs'],'metrics':metrics,'rows':rows})
assert hashlib.sha256(model_path.read_bytes()).hexdigest()==before_hash
report={'modelSha256':before_hash,'trainingPerformed':False,'parserOrRoiRulesChanged':False,'timeToleranceMinutes':5,'roomInferenceCpuMsAll':room_ms,'note':'Image 1 is new to previous inputs. Image 2 repeats previous evaluation content with minor pixel differences; report separately. Unknown room/end excluded, and untimed subjects tracked separately.','images':results}
(root/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
summary={**report,'images':[{k:v for k,v in r.items() if k!='rows'}for r in results]}
(root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
