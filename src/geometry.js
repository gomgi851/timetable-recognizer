const median = a => [...a].sort((a,b)=>a-b)[Math.floor(a.length/2)];
const distance = (a,b) => Math.max(...a.map((v,i)=>Math.abs(v-b[i])));
export function pixel(image,x,y) { const i=(Math.max(0,Math.min(image.height-1,y|0))*image.width+Math.max(0,Math.min(image.width-1,x|0)))*4; return [...image.data.slice(i,i+3)]; }
const luminance = c => c.reduce((a,b)=>a+b,0)/3;
function courseColor(c) { const l=luminance(c); return l>70 && l<240 && (Math.max(...c)-Math.min(...c)>12 || l<215); }
function groups(values,gap=3) { const out=[]; for(const v of values) { if(!out.length||v-out.at(-1).at(-1)>gap)out.push([]); out.at(-1).push(v); }return out.map(median); }
export function detectGeometry(image) {
  const {width:w,height:h}=image, runs=[];
  for(let y=0;y<h;y+=3) {
    let start=0, color=pixel(image,0,y);
    for(let x=1;x<=w;x++) { const c=x<w?pixel(image,x,y):[0,0,0];
      if(x===w||distance(c,color)>4) { if(x-start>w*.14&&x-start<w*.23&&courseColor(color))runs.push({x:start,y,width:x-start,color}); start=x;color=c; }
    }
  }
  if(runs.length<5)throw new Error('수업 칸을 찾지 못했습니다. 시간표 전체가 나온 원본 사진을 선택해주세요.');
  const bins=new Map();for(const r of runs){const k=Math.round(r.width/3);bins.set(k,(bins.get(k)||0)+1);}
  const mode=[...bins].sort((a,b)=>b[1]-a[1])[0][0]*3;
  const full=runs.filter(r=>Math.abs(r.width-mode)<6);
  let colWidth=median(full.map(r=>r.width))+2;
  const anchors=full.map(r=>{let x=r.x;while(x>w*.13)x-=colWidth;return x;}).filter(x=>x>w*.025&&x<w*.13);
  let left=median(anchors),right=left+5*colWidth;
  const palette=[];for(const r of full)if(!palette.some(c=>distance(c,r.color)<8))palette.push(r.color);
  let blocks=[];
  for(let day=0;day<5;day++) {
    const x=left+day*colWidth;let active=null;
    for(let y=0;y<h;y++) {
      const colors=Array.from({length:37},(_,i)=>pixel(image,x+colWidth*(.04+i*.025),y));
      const candidates=palette.map(c=>({c,n:colors.filter(v=>distance(v,c)<10).length})).sort((a,b)=>b.n-a.n);
      const c=candidates[0].c;
      const filled=candidates[0].n>=12;
      if(active&&(!filled||distance(active.color,c)>18)) { if(y-active.y>h*.016) blocks.push({...active,height:y-active.y});active=null; }
      if(filled&&!active)active={day,x:x+2,y,width:colWidth-4,color:c};
    }
    if(active&&h-active.y>h*.016)blocks.push({...active,height:h-active.y,clipped:true});
  }
  // Connected flat-color regions preserve the outer cell boundary despite text holes.
  let components=colorComponents(image).filter(b=>b.width>w*.13&&b.width<w*.25&&b.height>h*.015);
  if(components.length>=4){
    const xs=groups(components.map(b=>b.x).sort((a,b)=>a-b),8),spacings=[];
    for(let i=0;i<xs.length;i++)for(let j=i+1;j<xs.length;j++){const d=xs[j]-xs[i],steps=Math.round(d/colWidth);if(steps>0&&Math.abs(d/steps-colWidth)<colWidth*.12)spacings.push(d/steps);}
    if(spacings.length)colWidth=median(spacings);
    left=median(components.map(b=>b.x-Math.round((b.x-w*.075)/colWidth)*colWidth));right=left+5*colWidth;
    components=colorComponents(image,{left,colWidth}).filter(b=>b.width>w*.13&&b.width<w*.25&&b.height>h*.015);
    blocks=components.flatMap(b=>splitRegion(image,b)).map(b=>({...b,day:Math.max(0,Math.min(4,Math.round((b.x-left)/colWidth)))}));
  }
  const firstY=Math.min(...blocks.map(b=>b.y)),lastY=Math.max(...blocks.map(b=>b.y+b.height));
  const lineRows=[];
  for(let y=Math.max(4,firstY-h*.12)|0;y<Math.min(h-4,lastY+h*.03);y++) {
    const xs=[.008,.016,.024,.032].map(t=>left-w*t);
    const cs=xs.map(x=>pixel(image,x,y));const c=cs[0];
    if(cs.every(v=>distance(v,c)<6)&&Math.max(...c)-Math.min(...c)<12&&xs.every(x=>Math.abs(luminance(pixel(image,x,y))-luminance(pixel(image,x,y+4)))>6 || Math.abs(luminance(pixel(image,x,y))-luminance(pixel(image,x,y-4)))>6))lineRows.push(y);
  }
  const lines=groups(lineRows,4);
  const diffs=lines.slice(1).map((v,i)=>v-lines[i]).filter(v=>v>w*.07&&v<w*.19);
  if(diffs.length<2)throw new Error('시간 눈금 간격을 찾지 못했습니다. 표가 잘리지 않은 사진을 사용해주세요.');
  const hourHeight=median(diffs);
  const origin=lines.find(v=>v>=firstY-hourHeight*.15) ?? firstY;
  const alignedLines=lines.filter(y=>Math.abs((y-origin)/hourHeight-Math.round((y-origin)/hourHeight))<.08);
  const top=Math.min(...alignedLines),bottom=Math.max(...alignedLines);
  return {left,right,colWidth,top,bottom,hourHeight,lines:alignedLines,blocks:blocks.filter(b=>b.y>=top-5&&b.y<bottom+hourHeight),imageWidth:w,imageHeight:h};
}

function colorComponents(image,grid=null) {
  const {width:w,height:h,data:d}=image,n=w*h,seen=new Uint8Array(n),queue=new Int32Array(n),out=[];
  const valid=p=>{const i=p*4,r=d[i],g=d[i+1],b=d[i+2],l=(r+g+b)/3;return l>70&&l<240&&(Math.max(r,g,b)-Math.min(r,g,b)>12||l<215);};
  for(let p=0;p<n;p++) {
    if(seen[p]||!valid(p))continue;
    let head=0,tail=1;queue[0]=p;seen[p]=1;let x0=w,x1=0,y0=h,y1=0,rs=0,gs=0,bs=0;
    while(head<tail){const q=queue[head++],x=q%w,y=(q/w)|0;rs+=d[q*4];gs+=d[q*4+1];bs+=d[q*4+2];x0=Math.min(x0,x);x1=Math.max(x1,x);y0=Math.min(y0,y);y1=Math.max(y1,y);
      const neighbors=[x>0?q-1:-1,x<w-1?q+1:-1,y>0?q-w:-1,y<h-1?q+w:-1];
      for(const t of neighbors){if(t<0||seen[t]||!valid(t))continue;if(grid&&Math.floor((q%w-grid.left)/grid.colWidth)!==Math.floor((t%w-grid.left)/grid.colWidth))continue;const a=q*4,b=t*4;if(Math.max(Math.abs(d[a]-d[b]),Math.abs(d[a+1]-d[b+1]),Math.abs(d[a+2]-d[b+2]))>24)continue;seen[t]=1;queue[tail++]=t;}
    }
    if(tail>w*h*.0015){let width=x1-x0+1,height=y1-y0+1;
      if(width>w*.1){const counts=new Uint32Array(height);for(let j=0;j<tail;j++)counts[((queue[j]/w)|0)-y0]++;let lo=0,hi=height-1;while(lo<hi&&counts[lo]<width*.3)lo++;while(hi>lo&&counts[hi]<width*.3)hi--;y0+=lo;y1=y0+hi-lo;height=y1-y0+1;}
      if(tail/(width*height)>.35)out.push({x:x0,y:y0,width,height,color:[rs/tail,gs/tail,bs/tail],area:tail});}
  }
  return out;
}
function splitRegion(image,b) {
  const colors=[];
  for(let y=b.y;y<b.y+b.height;y++){const cs=[];for(let i=0;i<30;i++){const c=pixel(image,b.x+b.width*(.03+i*.032),y);if(courseColor(c))cs.push(c);}colors.push(cs.length?[0,1,2].map(k=>median(cs.map(c=>c[k]))):b.color);}
  const cuts=[];
  for(let i=12;i<colors.length-12;i++){const before=[0,1,2].map(k=>median(colors.slice(i-8,i-2).map(c=>c[k]))),after=[0,1,2].map(k=>median(colors.slice(i+2,i+8).map(c=>c[k])));if(distance(before,after)>42)cuts.push(i);}
  const boundaries=[0,...groups(cuts,12).filter(v=>v>image.height*.02&&colors.length-v>image.height*.02),b.height];
  return boundaries.slice(1).map((end,i)=>({...b,y:b.y+boundaries[i],height:end-boundaries[i],color:[0,1,2].map(k=>median(colors.slice(boundaries[i],end).map(c=>c[k])))}));
}

export function calibrateTime(lines, geometry) {
  const labels=[];
  for(const line of lines) {
    const text=line.text.trim();if(!/^\d{1,2}$/.test(text))continue;
    const hour=Number(text);if(hour<1||hour>23)continue;
    const index=Math.round((line.bbox.y0-geometry.top)/geometry.hourHeight);
    labels.push({hour,index});
  }
  const votes=new Map();
  for(const {hour,index} of labels)for(const h of (hour<=11?[hour,hour+12]:[hour])) {
    const start=h-index;if(start>=6&&start<=14)votes.set(start,(votes.get(start)||0)+1);
  }
  const ranked=[...votes].sort((a,b)=>b[1]-a[1]);
  if(!ranked.length||ranked[0][1]<2 || ranked[1]?.[1]===ranked[0][1])return {startHour:null,labels,warnings:['시간 눈금 인식이 불확실합니다. 시작 시각을 직접 지정해주세요.']};
  return {startHour:ranked[0][0],labels,warnings:[]};
}

export function blockTimes(block,g,startHour) {
  if(startHour==null)return {startMinutes:null,endMinutes:null};
  const start=startHour*60+(block.y-g.top)/g.hourHeight*60;
  const end=startHour*60+(block.y+block.height-g.top)/g.hourHeight*60;
  return {startMinutes:Math.round(start/5)*5,endMinutes:Math.round(end/5)*5};
}
