"""GPU fine-tuning. This script never opens test.json or test predictions."""
import copy,hashlib,json,os,random,time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from torch import nn
from room_model import load_model
from data import image_tensor,encode,predict,score

ROOT=Path('training/private')
protocol=json.loads(Path('training/protocol.json').read_text(encoding='utf-8'))
assert (ROOT/'frozen-protocol.json').read_bytes()==Path('training/protocol.json').read_bytes(),'Protocol changed after preparation'
assert torch.cuda.is_available(),'CUDA GPU required for this experiment'
seed=protocol['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
torch.set_num_threads(2)
torch.backends.cudnn.benchmark=False
device=torch.device('cuda:0')
train=json.loads((ROOT/'train.json').read_text(encoding='utf-8'))
validation=json.loads((ROOT/'validation.json').read_text(encoding='utf-8'))
assert not set(e['source'] for e in train)&set(e['source'] for e in validation)
model=load_model(ROOT/'english_g2.pth',device)
for name,param in model.named_parameters():
    param.requires_grad=not(name.startswith('FeatureExtraction.') or name.startswith('SequenceModeling.0.'))
optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=protocol['learningRate'],weight_decay=protocol['weightDecay'])
criterion=nn.CTCLoss(blank=0,zero_infinity=True)
metrics=score(validation,predict(model,validation,device))
@torch.inference_mode()
def validation_loss():
    model.eval();losses=[]
    for start in range(0,len(validation),protocol['batchSize']):
        records=validation[start:start+protocol['batchSize']]
        x=torch.stack([image_tensor(e) for e in records]).to(device)
        targets,lengths=encode([e['label'] for e in records])
        logits=model(x).log_softmax(-1).transpose(0,1)
        losses.append(float(criterion(logits,targets.to(device),torch.full((len(records),),logits.shape[0],dtype=torch.long),lengths)))
    return sum(losses)/len(losses)
metrics['ctcLoss']=validation_loss()
best=(metrics['accuracy'],-metrics['cer'],-metrics['ctcLoss']);best_epoch=0;bad=0
torch.save({'model':model.cpu().state_dict(),'epoch':0},ROOT/'best.pth');model.to(device)
started=time.perf_counter();history=[{'epoch':0,'validation':metrics}]
print(json.dumps({'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda,'trainRecords':len(train),'validationRecords':len(validation),'trainableParameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'totalParameters':sum(p.numel() for p in model.parameters()),'initialValidation':metrics}),flush=True)
for epoch in range(1,protocol['maxEpochs']+1):
    model.train();model.FeatureExtraction.eval();model.SequenceModeling[0].eval()
    order=list(range(len(train)))*protocol['augmentationRepeats'];random.shuffle(order);losses=[]
    for start in range(0,len(order),protocol['batchSize']):
        records=[train[i] for i in order[start:start+protocol['batchSize']]]
        x=torch.stack([image_tensor(e,True) for e in records]).to(device)
        targets,lengths=encode([e['label'] for e in records]);targets=targets.to(device)
        optimizer.zero_grad(set_to_none=True);logits=model(x).log_softmax(-1).transpose(0,1)
        input_lengths=torch.full((len(records),),logits.shape[0],dtype=torch.long)
        loss=criterion(logits,targets,input_lengths,lengths)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();losses.append(float(loss.detach()))
    val_predictions=predict(model,validation,device);val_score=score(validation,val_predictions);val_score['ctcLoss']=validation_loss()
    train_score=score(train,predict(model,train,device))
    row={'epoch':epoch,'loss':sum(losses)/len(losses),'train':train_score,'validation':val_score,'elapsedSeconds':time.perf_counter()-started,'maxGpuAllocatedBytes':torch.cuda.max_memory_allocated()}
    history.append(row);print(json.dumps(row),flush=True)
    candidate=(val_score['accuracy'],-val_score['cer'],-val_score['ctcLoss'])
    if candidate>best:
        best=candidate;best_epoch=epoch;bad=0
        torch.save({'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'epoch':epoch},ROOT/'best.pth')
    else:bad+=1
    (ROOT/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    if bad>=protocol['patience']:break
result={'selectedEpoch':best_epoch,'lastEpoch':history[-1]['epoch'],'selectedBy':'validation exact accuracy, CER, then CTC loss; no test access','seconds':time.perf_counter()-started,'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda,'checkpointSha256':hashlib.sha256((ROOT/'best.pth').read_bytes()).hexdigest(),'protocolSha256':hashlib.sha256(Path('training/protocol.json').read_bytes()).hexdigest(),'history':history}
(ROOT/'training-result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print('TRAINING_COMPLETE '+json.dumps({k:v for k,v in result.items() if k!='history'}),flush=True)
