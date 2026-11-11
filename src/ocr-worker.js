import {createWorker} from 'tesseract.js';
import {recognizeTimetable} from './recognizer.js';
let worker;
const send=progress=>postMessage({type:'progress',...progress});
self.onmessage=async({data})=>{
  const totalStart=performance.now();
  try{
    const init=performance.now();
    if(!worker)worker=await createWorker('kor+eng',1,{workerPath:new URL('/ocr/worker.min.js',self.location.origin).href,corePath:new URL('/ocr/',self.location.origin).href,langPath:new URL('/models',self.location.origin).href,gzip:false,cachePath:'timetable-best-v1',logger:m=>send({stage:m.status,progress:m.progress})});
    const initMs=performance.now()-init;
    const bitmap=await createImageBitmap(data.file);
    if(bitmap.width*bitmap.height>16_000_000){bitmap.close();throw new Error('이미지가 너무 큽니다. 1,600만 픽셀 이하로 줄여주세요.');}
    const source=new OffscreenCanvas(bitmap.width,bitmap.height);source.getContext('2d').drawImage(bitmap,0,0);bitmap.close();
    const result=await recognizeTimetable(source,{worker,makeCanvas:(w,h)=>new OffscreenCanvas(w,h),onProgress:send,startHour:data.startHour});
    postMessage({type:'result',result:{...result,initMs,totalMs:performance.now()-totalStart}});
  }catch(error){postMessage({type:'error',message:error.message});}
};
