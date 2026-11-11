from pathlib import Path
import math, random
import numpy as np
import torch
from PIL import Image,ImageEnhance,ImageFilter,ImageOps
from room_model import CHARACTERS

ROOT=Path('training/private')
HEIGHT=64
WIDTH=320

def image_tensor(entry, augment=False):
    if entry['roi'] is None:
        image=Image.new('L',(100,32),255)
    else:
        image=Image.open(ROOT/entry['roi']).convert('L')
    image=ImageOps.expand(image,border=4,fill=255)
    if augment:
        image=image.rotate(random.uniform(-1.2,1.2),resample=Image.Resampling.BICUBIC,expand=False,fillcolor=255)
        if random.random()<.25:image=image.filter(ImageFilter.GaussianBlur(random.uniform(.15,.65)))
        if random.random()<.2:image=image.filter(random.choice([ImageFilter.MinFilter(3),ImageFilter.MaxFilter(3)]))
        image=ImageEnhance.Contrast(image).enhance(random.uniform(.8,1.2))
    width=min(WIDTH,math.ceil(HEIGHT*image.width/image.height))
    if augment:width=max(8,min(WIDTH,round(width*random.uniform(.92,1.08))))
    image=image.resize((width,HEIGHT),Image.Resampling.BICUBIC)
    arr=np.array(image,dtype=np.float32)/127.5-1
    if augment:arr=np.clip(arr+np.random.normal(0,random.uniform(0,.025),arr.shape),-1,1).astype(np.float32)
    out=np.ones((1,HEIGHT,WIDTH),dtype=np.float32);out[:,:,:width]=arr
    return torch.from_numpy(out)

def edit_distance(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        nxt=[i]
        for j,y in enumerate(b,1):nxt.append(min(nxt[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=nxt
    return row[-1]

def encode(labels):
    mapping={c:i+1 for i,c in enumerate(CHARACTERS)}
    return torch.tensor([mapping[c] for label in labels for c in label],dtype=torch.long),torch.tensor([len(label) for label in labels],dtype=torch.long)

def score(entries,predictions):
    correct=sum(e['label']==p for e,p in zip(entries,predictions))
    edits=sum(edit_distance(e['label'],p) for e,p in zip(entries,predictions))
    chars=sum(len(e['label']) for e in entries)
    return {'correct':correct,'total':len(entries),'accuracy':correct/len(entries) if entries else 0,'characterEdits':edits,'characters':chars,'cer':edits/chars if chars else 0}

@torch.inference_mode()
def predict(model,entries,device,batch=8):
    from room_model import decode
    model.eval();predictions=[]
    for start in range(0,len(entries),batch):
        chunk=entries[start:start+batch]
        logits=model(torch.stack([image_tensor(e) for e in chunk]).to(device))
        predictions.extend(decode(logits))
    # Failed extraction is scored as failure regardless of a model's blank-image guess.
    return [p if e['roi'] else '' for e,p in zip(entries,predictions)]
