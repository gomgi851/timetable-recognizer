import fs from 'node:fs/promises';
import {createHash} from 'node:crypto';
const root=process.argv[2]||'samples/private';
const truth=JSON.parse(await fs.readFile(`${root}/truth.json`,'utf8'));
const normalize=s=>String(s??'').normalize('NFKC').replace(/\s/g,'').toUpperCase();
const fields=['name','roomCode','day','startMinutes','endMinutes'];
const metrics=Object.fromEntries(fields.map(f=>[f,{correct:0,total:0,excluded:0}]));
const report={method:truth.method,timeToleranceMinutes:5,matching:'Greedy one-to-one, same weekday and start time within 45 min; unknown start matched by normalized name containment. Missing predictions count as incorrect. Name whitespace and case ignored; room code only.',samples:[],metrics,expected:0,predicted:0,matched:0,fullyCorrect:0,fullyScorable:0};
const seen=new Set();
for(const sample of truth.samples){
 if(seen.has(sample.sha256))throw new Error('Duplicate input in truth');seen.add(sample.sha256);
 const bytes=await fs.readFile(`${root}/${sample.id}.png`);if(createHash('sha256').update(bytes).digest('hex')!==sample.sha256)throw new Error('Image hash mismatch');
 const result=JSON.parse(await fs.readFile(`${root}/results/${sample.id}.json`,'utf8')),preds=result.classes||[],used=new Set(),rows=[];
 for(const expected of sample.classes){
  let best=-1,bestCost=Infinity;
  for(let i=0;i<preds.length;i++){if(used.has(i)||preds[i].day!==expected.day)continue;const p=preds[i];let cost=expected.startMinutes==null?(normalize(p.name).includes(normalize(expected.name))?0:Infinity):Math.abs((p.startMinutes??-10000)-expected.startMinutes);if(cost<=45&&cost<bestCost){best=i;bestCost=cost;}}
  const p=best<0?null:preds[best];if(p){used.add(best);report.matched++;}
  const correct={};for(const field of fields){const m=metrics[field];if(expected[field]==null){m.excluded++;correct[field]=null;continue;}m.total++;const ok=!!p&&(field.endsWith('Minutes')?p[field]!=null&&Math.abs(p[field]-expected[field])<=report.timeToleranceMinutes:normalize(p[field])===normalize(expected[field]));if(ok)m.correct++;correct[field]=ok;}
  if(fields.every(f=>expected[f]!=null)){report.fullyScorable++;if(fields.every(f=>correct[f]))report.fullyCorrect++;}
  rows.push({expected,predicted:p?Object.fromEntries(fields.map(f=>[f,p[f]])):null,correct});
 }
 report.expected+=sample.classes.length;report.predicted+=preds.length;
 report.samples.push({id:sample.id,sha256:sample.sha256,expected:sample.classes.length,predicted:preds.length,matched:used.size,elapsedMs:result.elapsedMs??null,error:result.error,rows});
}
for(const m of Object.values(metrics))m.accuracy=m.total?m.correct/m.total:null;
const rows=report.samples.flatMap(s=>s.rows);
const roomsPresent=rows.filter(r=>r.expected.roomCode);
report.roomCodePresentOnly={correct:roomsPresent.filter(r=>r.correct.roomCode).length,total:roomsPresent.length,accuracy:roomsPresent.filter(r=>r.correct.roomCode).length/roomsPresent.length};
report.limitations=['Development-set evaluation, not held-out or general accuracy.','Ground truth transcribed visually by one annotator; no independent second annotation.','Time values inferred visually from grid; ±5 minute tolerance, not exact-to-minute ground truth.','Weekday is part of matching and is not an independent weekday classification benchmark.','Full classroom/building descriptions were not scored; room code only.','One obscured room, one uncertain start and four uncertain/obscured ends excluded per field.','Duplicate screenshots removed by SHA-256.','Mobile browser speed and peak memory have not been measured.'];
report.detectionRecall=report.matched/report.expected;report.detectionPrecision=report.matched/report.predicted;
report.fullAccuracy=report.fullyCorrect/report.fullyScorable;
const times=report.samples.filter(s=>s.elapsedMs!=null).map(s=>s.elapsedMs/1000).sort((a,b)=>a-b);
report.speed={environment:`Node ${process.version}; same WASM recognition code as browser, not a mobile/browser speed measurement`,count:times.length,minSeconds:times[0],medianSeconds:times[Math.floor(times.length/2)],maxSeconds:times.at(-1),meanSeconds:times.reduce((a,b)=>a+b,0)/times.length,excludes:'Model loading, page loading, image decoding; includes geometry and OCR.'};
await fs.writeFile(`${root}/evaluation.json`,JSON.stringify(report,null,2));
const summary={...report,samples:report.samples.map(({rows,...s})=>s)};
await fs.mkdir('evaluation',{recursive:true});await fs.writeFile('evaluation/summary.json',JSON.stringify(summary,null,2)+'\n');
console.log(JSON.stringify(summary,null,2));
