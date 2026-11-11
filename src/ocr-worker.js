import {createWorker} from 'tesseract.js';
import {recognizeTimetable} from './recognizer.js';
import {loadRoomModel,roomROI} from './room-model.js';
let worker,roomModel;
const send=progress=>postMessage({type:'progress',...progress});
self.onmessage=async({data})=>{
 const totalStart=performance.now(),timings={},cold=!worker||!roomModel;
 try{
  send({stage:'OCR 모델 준비 (첫 실행은 다운로드 필요)'});let t=performance.now();
  if(!worker)worker=await createWorker('kor+eng',1,{workerPath:new URL('/ocr/worker.min.js',self.location.origin).href,corePath:new URL('/ocr/',self.location.origin).href,langPath:new URL('/models',self.location.origin).href,gzip:false,cachePath:'timetable-best-v1',logger:m=>send({stage:m.status,progress:m.progress})});
  timings.ocrInitMs=performance.now()-t;send({stage:'학습한 강의실 AI 준비'});t=performance.now();
  if(!roomModel)roomModel=await loadRoomModel();timings.roomInitMs=performance.now()-t;t=performance.now();
  const bitmap=await createImageBitmap(data.file);if(bitmap.width*bitmap.height>16_000_000){bitmap.close();throw new Error('1,600만 픽셀 이하 사진을 선택해주세요.');}
  const makeCanvas=(w,h)=>new OffscreenCanvas(w,h),source=makeCanvas(bitmap.width,bitmap.height);source.getContext('2d').drawImage(bitmap,0,0);bitmap.close();timings.decodeMs=performance.now()-t;
  timings.roomRoiMs=0;timings.roomInferenceMs=0;let inferred=0;t=performance.now();
  const result=await recognizeTimetable(source,{worker,makeCanvas,onProgress:send,startHour:data.startHour,roomRecognizer:async context=>{
   let started=performance.now();const roi=await roomROI(context);timings.roomRoiMs+=performance.now()-started;
   if(!roi)return {baselineRoomCode:context.parsed.roomCode,roomModelStatus:'강의실 영역 미검출 · OCR 유지'};
   started=performance.now();const code=await roomModel.recognize(roi,makeCanvas);timings.roomInferenceMs+=performance.now()-started;inferred++;
   return {baselineRoomCode:context.parsed.roomCode,roomCode:code||context.parsed.roomCode,roomModelStatus:code?'학습 모델 적용':'모델 결과 없음 · OCR 유지'};
  }});
  timings.extractionMs=performance.now()-t;timings.workerTotalMs=performance.now()-totalStart;
  postMessage({type:'result',result:{...result,timings,cold,inferred,engine:'Tesseract kor+eng + fine-tuned room-code-v2 ONNX / WASM CPU',modelSha256:roomModel.metadata.sha256,exclusions:['격자 밖 시간 미지정 사이버강의']}});
 }catch(error){postMessage({type:'error',message:String(error.message||error)});}
};
