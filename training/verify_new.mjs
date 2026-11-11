// Evaluation-only runner: no model training or parser/ROI rule changes.
import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {createCanvas,loadImage} from '@napi-rs/canvas';
import {createWorker} from 'tesseract.js';
import {recognizeTimetable,textCrop,recognizeLines,parseCourse,flattenLines} from '../src/recognizer.js';
const root=process.argv[2]||'training/private/new-check';
const inputs=JSON.parse(await fs.readFile(root+'/inputs.json','utf8'));
await fs.mkdir(root+'/rois',{recursive:true});
const worker=await createWorker('kor+eng',1,{langPath:'public/models',gzip:false,cacheMethod:'none'});
const norm=s=>s.replace(/\s/g,'').toUpperCase(),records=[];
try{for(const input of inputs){
 const image=await loadImage(input.file),source=createCanvas(image.width,image.height);source.getContext('2d').drawImage(image,0,0);
 const baseline=await recognizeTimetable(source,{worker,makeCanvas:createCanvas});
 await fs.writeFile(`${root}/${input.id}-baseline.json`,JSON.stringify(baseline,null,2));
 for(let i=0;i<baseline.classes.length;i++){
  const found=baseline.classes[i];
  const entry={id:`${input.id}-${i}`,source:input.id,baselineIndex:i,tesseract:found.roomCode??'',roi:null};
  const b=found.bounds,color=baseline.geometry.blocks[i]?.color;
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
    if(end>start){const sx=Math.max(0,symbols[start].bbox.x0-4),ex=Math.min(line.width,symbols[end-1].bbox.x1+4),sy=8,roi=createCanvas(Math.max(1,ex-sx),h+4);roi.getContext('2d').drawImage(line,sx,sy,roi.width,roi.height,0,0,roi.width,roi.height);const buffer=roi.toBuffer('image/png');const dest=`${root}/rois/${entry.id}.png`;entry.roi=path.relative('training/private',dest).replaceAll('\\','/');entry.roiSha256=createHash('sha256').update(buffer).digest('hex');entry.roiText=result.data.text.trim();await fs.writeFile(dest,buffer);}
   }
  }
  records.push(entry);
 }
 console.log(`${input.id}: ${baseline.classes.length} blocks, ${Math.round(baseline.elapsedMs)}ms baseline`);
}}finally{await worker.terminate();}
await fs.writeFile(root+'/inference-inputs.json',JSON.stringify(records,null,2));
