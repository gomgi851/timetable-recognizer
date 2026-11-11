import fs from 'node:fs/promises';
import {loadImage,createCanvas} from '@napi-rs/canvas';
import {createWorker} from 'tesseract.js';
import {recognizeTimetable} from '../src/recognizer.js';
await fs.mkdir('samples/private/results',{recursive:true});
const init=performance.now();
const worker=await createWorker('kor+eng',1,{langPath:'public/models',gzip:false,cacheMethod:'none'});
console.log('worker init ms',Math.round(performance.now()-init));
try {for(const id of process.argv.slice(2).length?process.argv.slice(2):['1','2','3','4','5','6','7','8']) {
 const im=await loadImage(`samples/private/${id}.png`),c=createCanvas(im.width,im.height);c.getContext('2d').drawImage(im,0,0);
 try{const result=await recognizeTimetable(c,{worker,makeCanvas:createCanvas});await fs.writeFile(`samples/private/results/${id}.json`,JSON.stringify(result,null,2));console.log(id,Math.round(result.elapsedMs),JSON.stringify({calibration:result.calibration,classes:result.classes.map(c=>({day:c.day,name:c.name,roomCode:c.roomCode,start:c.startMinutes,end:c.endMinutes,raw:c.rawText}))}));}
 catch(e){console.log(id,'ERROR',e.message);await fs.writeFile(`samples/private/results/${id}.json`,JSON.stringify({error:e.message,classes:[]}));}
}}finally{await worker.terminate();}
