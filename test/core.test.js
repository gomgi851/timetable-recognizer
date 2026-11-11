import {test} from 'node:test';import assert from 'node:assert/strict';
import {blockTimes,calibrateTime,detectGeometry} from '../src/geometry.js';import {parseCourse} from '../src/recognizer.js';
test('time calibration unwraps afternoon labels and preserves half-hour block boundaries',()=>{const g={top:100,hourHeight:120};const labels=[11,12,1,2].map((hour,i)=>({text:String(hour),bbox:{y0:110+i*120}}));assert.equal(calibrateTime(labels,g).startHour,11);assert.deepEqual(blockTimes({y:400,height:180},g,11),{startMinutes:810,endMinutes:900});});
test('ambiguous or missing time axis yields unknown times',()=>{assert.equal(calibrateTime([],{top:0,hourHeight:100}).startHour,null);assert.equal(blockTimes({y:50,height:100},{top:0,hourHeight:100},null).endMinutes,null);});
test('wrapped course and compound room code are retained, professor is excluded',()=>{const values=[['기초프로그래',40],['밍',40],['김명준',27],['401-3183(신공학관)',27]];const lines=values.map(([text,h],i)=>({text,bbox:{y0:i*55,y1:i*55+h}}));const out=parseCourse(lines);assert.equal(out.name,'기초프로그래밍');assert.equal(out.roomCode,'401-3183');});
test('no classroom does not invent one',()=>assert.equal(parseCourse([{text:'검도',bbox:{y0:0,y1:30}}]).roomCode,''));
test('unrelated blank image fails without fabricating classes',()=>assert.throws(()=>detectGeometry({width:100,height:100,data:new Uint8ClampedArray(40000).fill(255)})));
