"""Export the frozen selected model and verify CPU ONNX predictions, not retrain."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
import onnx
import onnxruntime as ort
from room_model import load_model,decode,CHARACTERS
from data import image_tensor,score

root=Path('training/private');out=Path('trained');out.mkdir(exist_ok=True)
training=json.loads((root/'training-result.json').read_text(encoding='utf-8'))
model=load_model(root/'best.pth','cpu').eval()
dummy=torch.ones(1,1,64,320)
path=out/'room-code-v2.onnx'
torch.onnx.export(model,dummy,path,input_names=['image'],output_names=['logits'],opset_version=17,dynamo=False)
onnx.checker.check_model(onnx.load(path))
options=ort.SessionOptions();options.intra_op_num_threads=2
session=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
entries=json.loads((root/'test.json').read_text(encoding='utf-8'))
predictions=[];torch_predictions=[];max_error=0
with torch.inference_mode():
    for entry in entries:
        image=image_tensor(entry).unsqueeze(0)
        expected=model(image)
        actual=session.run(None,{'image':image.numpy()})[0]
        max_error=max(max_error,float(np.max(np.abs(actual-expected.numpy()))))
        predictions+=decode(torch.from_numpy(actual));torch_predictions+=decode(expected)
assert predictions==torch_predictions,'ONNX and PyTorch decoded predictions differ'
metadata={'model':'EasyOCR english_g2, fine-tuned last BiLSTM/linear layers on supplied screenshot room crops','parameters':sum(p.numel() for p in model.parameters()),'input':{'name':'image','shape':[1,1,64,320],'dtype':'float32','normalization':'grayscale / 127.5 - 1; preserve aspect, white/right pad, image padded by 4 pixels before resize'},'output':{'name':'logits','ctcBlank':0,'characters':CHARACTERS,'decode':'greedy CTC; ASCII alphanumeric and hyphen only, uppercase result; no room dictionary'},'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'selectedEpoch':training['selectedEpoch'],'protocolSha256':training['protocolSha256'],'onnxCpuParity':{'matchingPredictions':len(predictions),'total':len(entries),'maxAbsLogitError':max_error,'score':score(entries,predictions)},'deploymentStatus':'Export validated with ONNX Runtime CPU. Browser and mobile inference have not yet been wired or benchmarked. Existing browser UI still uses the original baseline.','license':'Based on EasyOCR pretrained english_g2, Apache-2.0; see training/README.md'}
(out/'room-code-v2.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(metadata,ensure_ascii=False,indent=2))
