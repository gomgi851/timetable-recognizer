import * as ort from 'onnxruntime-web/wasm';
import {flattenLines} from './recognizer.js';
export function decodeCTC(data,steps,characters){
 const count=characters.length+1,allowed=[0,...Array.from(characters,(c,i)=>/^[A-Za-z0-9-]$/.test(c)?i+1:-1).filter(i=>i>0)];
 let last=-1,text='';for(let t=0;t<steps;t++){let best=0;for(const i of allowed)if(data[t*count+i]>data[t*count+best])best=i;if(best&&best!==last)text+=characters[best-1];last=best;}return text.toUpperCase();
}
// Frozen validation room-line and symbol boundaries; no room dictionary.
export async function roomROI({crop,lines,parsed,worker,makeCanvas}){
 const norm=s=>s.replace(/\s/g,'').toUpperCase(),roomLine=lines.find(l=>parsed.room&&norm(parsed.room).startsWith(norm(l.text.trim())));if(!roomLine)return null;
 const y=roomLine.bbox.y0,h=roomLine.bbox.y1-y,line=makeCanvas(crop.width,h+20),ctx=line.getContext('2d');ctx.fillStyle='white';ctx.fillRect(0,0,line.width,line.height);ctx.drawImage(crop,0,y,crop.width,h,0,10,crop.width,h);
 await worker.setParameters({tessedit_pageseg_mode:'7'});
 const input=line.toBuffer?line.toBuffer('image/png'):new Uint8Array(await(await line.convertToBlob()).arrayBuffer()),result=await worker.recognize(input,{},{blocks:true});
 const symbols=flattenLines(result.data).flatMap(l=>l.words.flatMap(w=>w.symbols||[])).filter(s=>s.text.trim()),digit=symbols.findIndex(s=>/\d/.test(s.text));if(digit<0)return null;
 let start=0;if(symbols.slice(0,digit).filter(s=>/[가-힣]/.test(s.text)).length>=2){start=digit;while(start>0&&/^[A-Za-z]$/.test(symbols[start-1].text))start--;}
 let end=symbols.findIndex((s,i)=>i>digit&&/[([{]/.test(s.text));if(end<0)end=symbols.findIndex((s,i)=>i>digit&&/[가-힣]/.test(s.text));if(end<0)end=symbols.length;if(end<=start)return null;
 const sx=Math.max(0,symbols[start].bbox.x0-4),ex=Math.min(line.width,symbols[end-1].bbox.x1+4),roi=makeCanvas(Math.max(1,ex-sx),h+4);roi.getContext('2d').drawImage(line,sx,8,roi.width,roi.height,0,0,roi.width,roi.height);return roi;
}
export function roomInput(roi,makeCanvas){
 const padded=makeCanvas(roi.width+8,roi.height+8),p=padded.getContext('2d');p.fillStyle='white';p.fillRect(0,0,padded.width,padded.height);p.drawImage(roi,4,4);
 const width=Math.min(320,Math.ceil(64*padded.width/padded.height)),resized=makeCanvas(width,64),ctx=resized.getContext('2d');ctx.imageSmoothingEnabled=true;ctx.imageSmoothingQuality='high';ctx.drawImage(padded,0,0,width,64);
 const pixels=ctx.getImageData(0,0,width,64).data,values=new Float32Array(64*320).fill(1);for(let y=0;y<64;y++)for(let x=0;x<width;x++){const i=(y*width+x)*4;values[y*320+x]=(pixels[i]*.299+pixels[i+1]*.587+pixels[i+2]*.114)/127.5-1;}return values;
}
export async function loadRoomModel(){
 ort.env.wasm.numThreads=1;ort.env.wasm.wasmPaths=new URL('/runtime/',self.location.origin).href;
 const response=await fetch('/models/room-code-v2.json');if(!response.ok)throw new Error('강의실 모델 메타데이터 로딩 실패');const metadata=await response.json();
 const session=await ort.InferenceSession.create('/models/room-code-v2.onnx',{executionProviders:['wasm']});
 return {metadata,async recognize(roi,makeCanvas){const input=new ort.Tensor('float32',roomInput(roi,makeCanvas),[1,1,64,320]),result=await session.run({image:input}),logits=result.logits,text=decodeCTC(logits.data,logits.dims[1],metadata.output.characters);input.dispose();Object.values(result).forEach(t=>t.dispose());return text;}};
}
