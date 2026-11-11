import fs from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {createCanvas,loadImage} from '@napi-rs/canvas';
import {createWorker} from 'tesseract.js';
import {textCrop,recognizeLines,parseCourse,flattenLines} from '../src/recognizer.js';
const root='training/private';await fs.mkdir(root+'/rois',{recursive:true});
const protocol=JSON.parse(await fs.readFile('training/protocol.json','utf8'));
await fs.copyFile('training/protocol.json',root+'/frozen-protocol.json');
const truth=JSON.parse(await fs.readFile('samples/private/truth.json','utf8'));
const worker=await createWorker('kor+eng',1,{langPath:'public/models',gzip:false,cacheMethod:'none'});
const partitions={train:[],validation:[],test:[]};
const norm=s=>s.replace(/\s/g,'').toUpperCase();
try{for(const sample of truth.samples){
 const split=protocol.trainImages.includes(sample.id)?'train':protocol.validationImages.includes(sample.id)?'validation':'test';
 const baseline=JSON.parse(await fs.readFile(`samples/private/results/${sample.id}.json`,'utf8'));
 const image=await loadImage(`samples/private/${sample.id}.png`),source=createCanvas(image.width,image.height);source.getContext('2d').drawImage(image,0,0);
 const used=new Set();
 for(let n=0;n<sample.classes.length;n++){
  const expected=sample.classes[n];if(!expected.roomCode)continue;
  const candidates=baseline.classes.map((p,i)=>({p,i,d:expected.startMinutes==null?norm(p.name).includes(norm(expected.name))?0:1e6:Math.abs((p.startMinutes??-1e6)-expected.startMinutes)})).filter(c=>!used.has(c.i)&&c.p.day===expected.day&&c.d<=45).sort((a,b)=>a.d-b.d);
  const found=candidates[0];const entry={id:`${sample.id}-${n}`,source:sample.id,sourceSha256:sample.sha256,split,label:expected.roomCode.toUpperCase(),baselineIndex:found?.i??null,tesseract:found?.p.roomCode??'',roi:null};
  if(!found){partitions[split].push(entry);continue;}used.add(found.i);
  const b=found.p.bounds;const color=baseline.geometry.blocks[found.i]?.color;
  const crop=textCrop(source,{...b,x:b.x+3,y:b.y+3,width:b.width-6,height:b.height-6},createCanvas,color,2);
  await worker.setParameters({tessedit_char_whitelist:'',user_defined_dpi:'150'});
  const lines=await recognizeLines(crop,worker,createCanvas),parsed=parseCourse(lines);
  const roomLine=lines.find(l=>norm(parsed.room).startsWith(norm(l.text.trim()))&&parsed.room);
  if(roomLine){
   const y=roomLine.bbox.y0,h=roomLine.bbox.y1-y;
   const line=createCanvas(crop.width,h+20),ctx=line.getContext('2d');ctx.fillStyle='white';ctx.fillRect(0,0,line.width,line.height);ctx.drawImage(crop,0,y,crop.width,h,0,10,crop.width,h);
   await worker.setParameters({tessedit_pageseg_mode:'7'});
   const result=await worker.recognize(line.toBuffer('image/png'),{},{blocks:true});
   const symbols=flattenLines(result.data).flatMap(l=>l.words.flatMap(w=>w.symbols||[])).filter(s=>s.text.trim());
   const digit=symbols.findIndex(s=>/\d/.test(s.text));
   if(digit>=0){
    let start=0;if(symbols.slice(0,digit).filter(s=>/[가-힣]/.test(s.text)).length>=2){start=digit;while(start>0&&/^[A-Za-z]$/.test(symbols[start-1].text))start--;}
    let end=symbols.findIndex((s,i)=>i>digit&&/[([{]/.test(s.text));
    if(end<0)end=symbols.findIndex((s,i)=>i>digit&&/[가-힣]/.test(s.text));
    if(end<0)end=symbols.length;
    if(end>start){const sx=Math.max(0,symbols[start].bbox.x0-4),ex=Math.min(line.width,symbols[end-1].bbox.x1+4),sy=8,roi=createCanvas(Math.max(1,ex-sx),h+4);roi.getContext('2d').drawImage(line,sx,sy,roi.width,roi.height,0,0,roi.width,roi.height);const buffer=roi.toBuffer('image/png');entry.roi=`rois/${entry.id}.png`;entry.roiSha256=createHash('sha256').update(buffer).digest('hex');entry.roiText=result.data.text.trim();await fs.writeFile(root+'/'+entry.roi,buffer);}
   }
  }
  partitions[split].push(entry);
 }
 console.log(`Prepared screenshot ${sample.id} (${split})`);
}}finally{await worker.terminate();}
// Deduplicate crops within training, and drop exact pixel copies from training if
// they also occur outside train. Labels are never used to manufacture input crops.
const heldoutHashes=new Set([...partitions.validation,...partitions.test].map(e=>e.roiSha256).filter(Boolean));
const seen=new Set();let droppedDuplicates=0;
partitions.train=partitions.train.filter(e=>{if(!e.roi||protocol.trainingQualityExcludedIds?.includes(e.id))return false;if(seen.has(e.roiSha256)||heldoutHashes.has(e.roiSha256)){droppedDuplicates++;return false;}seen.add(e.roiSha256);return true;});
for(const [split,entries] of Object.entries(partitions))await fs.writeFile(`${root}/${split}.json`,JSON.stringify(entries,null,2));
await fs.writeFile(`${root}/preparation-audit.json`,JSON.stringify({protocolSha256:createHash('sha256').update(await fs.readFile('training/protocol.json')).digest('hex'),counts:Object.fromEntries(Object.entries(partitions).map(([s,e])=>[s,{total:e.length,withROI:e.filter(r=>r.roi).length}])),droppedDuplicates,noTrainHeldoutPixelDuplicate:true},null,2));
console.log(await fs.readFile(`${root}/preparation-audit.json`,'utf8'));
