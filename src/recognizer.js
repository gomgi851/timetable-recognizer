import {detectGeometry,calibrateTime,blockTimes,pixel} from './geometry.js';
export const weekdays=['월','화','수','목','금'];
export const flattenLines = data => (data.blocks||[]).flatMap(b=>b.paragraphs.flatMap(p=>p.lines));
export async function recognizeLines(crop,worker,makeCanvas) {
  const {width,height}=crop,ctx=crop.getContext('2d'),d=ctx.getImageData(0,0,width,height).data,groups=[];
  for(let y=0;y<height;y++){let count=0;for(let x=0;x<width;x++)if(d[(y*width+x)*4]<100)count++;if(count>2){if(!groups.length||y-groups.at(-1).at(-1)>12)groups.push([]);groups.at(-1).push(y);}}
  await worker.setParameters({tessedit_pageseg_mode:'7'});
  const lines=[];for(const g of groups){const y=g[0],h=g.at(-1)-y+1;if(h<8)continue;const runs=[];for(let row=y;row<y+h;row++){let run=0;for(let x=0;x<=width;x++){if(x<width&&d[(row*width+x)*4]<100)run++;else if(run){runs.push(run);run=0;}}}runs.sort((a,b)=>a-b);const strokeRatio=(runs[Math.floor(runs.length/2)]||0)/h;const line=makeCanvas(width,h+20),lc=line.getContext('2d');lc.fillStyle='white';lc.fillRect(0,0,width,h+20);lc.drawImage(crop,0,y,width,h,0,10,width,h);const {data}=await worker.recognize(await canvasInput(line));lines.push({text:data.text.trim(),confidence:data.confidence,strokeRatio,bbox:{x0:0,x1:width,y0:y,y1:y+h}});}
  return lines;
}

// Text is binarized relative to its own cell color, independently of theme.
export function textCrop(source,rect,makeCanvas,background=null,scale=2) {
  const x=Math.max(0,Math.round(rect.x)),y=Math.max(0,Math.round(rect.y));
  const width=Math.max(1,Math.min(source.width-x,Math.round(rect.width))),height=Math.max(1,Math.min(source.height-y,Math.round(rect.height)));
  const canvas=makeCanvas(width*scale+24,height*scale+24),ctx=canvas.getContext('2d');
  ctx.fillStyle='white';ctx.fillRect(0,0,canvas.width,canvas.height);
  const input=source.getContext('2d').getImageData(x,y,width,height),out=ctx.createImageData(width,height);
  const bg=background||pixel(input,width/2,height/2);
  const bgLum=bg.reduce((a,b)=>a+b)/3;
  // White and black text both differ substantially from flat block fill.
  for(let i=0;i<input.data.length;i+=4){const c=[...input.data.slice(i,i+3)];const lum=c.reduce((a,b)=>a+b)/3;
    const ink=background ? Math.abs(lum-bgLum)>Math.max(30,Math.min(bgLum,255-bgLum)*.38) : (bgLum<128?lum>bgLum+45:lum<bgLum-45);
    out.data[i]=out.data[i+1]=out.data[i+2]=ink?0:255;out.data[i+3]=255;
  }
  const small=makeCanvas(width,height);small.getContext('2d').putImageData(out,0,0);ctx.imageSmoothingEnabled=false;ctx.drawImage(small,12,12,width*scale,height*scale);
  return canvas;
}

export function parseCourse(lines) {
  lines=lines.filter(l=>/[A-Za-z가-힣0-9)]/.test(l.text)).sort((a,b)=>a.bbox.y0-b.bbox.y0);
  if(!lines.length)return {name:'',room:'',roomCode:'',rawText:'',warnings:['글자를 읽지 못했습니다.']};
  const texts=lines.map(l=>l.text.trim()),rawText=texts.join('\n');
  const codePattern=/(?:[A-Za-z]\s*)?\d{3,4}(?:\s*-\s*[A-Za-z]?\d{2,4})?/;
  const roomIndex=texts.findIndex(t=>codePattern.test(t)&&(t.includes('(')||t.includes('[')||t.includes('{')||/^[^가-힣]{0,4}\d{3,5}(?:-[A-Za-z]?\d{2,4})?$/.test(t)));
  const firstH=lines[0].bbox.y1-lines[0].bbox.y0;
  let nameCount=1;
  while(nameCount<Math.min(roomIndex<0?lines.length:roomIndex,3)) {
    const prev=lines[nameCount-1],cur=lines[nameCount];
    const currentH=cur.bbox.y1-cur.bbox.y0;
    if(/^[A-Z]{2,}\d*$/.test(texts[0])&&/[가-힣]/.test(cur.text))break;
    if(currentH<firstH*.88 || cur.bbox.y0-prev.bbox.y1>firstH*.55)break;
    nameCount++;
  }
  if(roomIndex>0)nameCount=Math.min(nameCount,roomIndex);
  const name=texts.slice(0,nameCount).join('').replace(/\s+/g,' ').trim();
  const room=roomIndex<0?'':texts.slice(roomIndex).join('').replace(/\s+/g,' ').trim();
  const roomCode=roomIndex<0?'':texts[roomIndex].match(codePattern)?.[0]?.replace(/\s/g,'')||'';
  const warnings=[];if(!name)warnings.push('과목명을 확인해주세요.');if(!roomCode)warnings.push('강의실을 확인해주세요.');
  return {name,room,roomCode,rawText,warnings};
}

export async function recognizeTimetable(source,{worker,makeCanvas,onProgress=()=>{},startHour=null}) {
  const started=performance.now();
  const geometry=detectGeometry(source.getContext('2d').getImageData(0,0,source.width,source.height));
  onProgress({stage:'시간 눈금 인식',progress:0});
  const axisRect={x:Math.max(0,geometry.left-source.width*.065),y:geometry.top,width:source.width*.06,height:Math.min(source.height-geometry.top,geometry.bottom-geometry.top+geometry.hourHeight*.7)};
  const axis=textCrop(source,axisRect,makeCanvas,null,2);
  await worker.setParameters({tessedit_pageseg_mode:'11',tessedit_char_whitelist:'0123456789',user_defined_dpi:'150'});
  const axisResult=await worker.recognize(await canvasInput(axis),{},{blocks:true});
  const axisLines=flattenLines(axisResult.data).map(l=>({...l,bbox:{...l.bbox,y0:(l.bbox.y0-12)/2+axisRect.y,y1:(l.bbox.y1-12)/2+axisRect.y}}));
  const calibration=calibrateTime(axisLines,geometry);
  if(startHour!=null)calibration.startHour=Number(startHour);
  await worker.setParameters({tessedit_pageseg_mode:'6',tessedit_char_whitelist:'',preserve_interword_spaces:'1'});
  const classes=[];
  for(let i=0;i<geometry.blocks.length;i++) {
    const b=geometry.blocks[i];onProgress({stage:`수업 인식 ${i+1}/${geometry.blocks.length}`,progress:i/geometry.blocks.length});
    const crop=textCrop(source,{...b,x:b.x+3,y:b.y+3,width:b.width-6,height:b.height-6},makeCanvas,b.color,2);
    const lines=await recognizeLines(crop,worker,makeCanvas);
    const parsed=parseCourse(lines);
    const endY=b.y+b.height;
    const outerX=Math.max(1,Math.round(source.width*.005));
    const endPixels=source.getContext('2d').getImageData(outerX,Math.min(source.height-11,Math.max(0,endY-5)),1,11).data;
    const panelBoundary=Math.max(...[0,1,2].map(i=>Math.abs(endPixels[i]-endPixels[40+i])))>8;
    const clipped=b.clipped||endY>=source.height-2||panelBoundary;
    const times=blockTimes(b,geometry,calibration.startHour);
    if(clipped)parsed.warnings.push('수업 끝이 잘렸을 수 있습니다. 종료 시각을 확인해주세요.');
    classes.push({id:`course-${i+1}`,day:weekdays[b.day],...parsed,...times,...(clipped?{endMinutes:null}:{}),confidence:lines.length?lines.reduce((s,l)=>s+l.confidence,0)/lines.length:0,bounds:{x:b.x,y:b.y,width:b.width,height:b.height}});
  }
  return {classes,geometry,calibration,elapsedMs:performance.now()-started,engine:'Tesseract.js kor+eng LSTM',warnings:calibration.startHour==null?calibration.warnings:[]};
}
async function canvasInput(canvas){if(typeof canvas.toBuffer==='function')return canvas.toBuffer('image/png');if(canvas.convertToBlob)return new Uint8Array(await (await canvas.convertToBlob()).arrayBuffer());return canvas;}
