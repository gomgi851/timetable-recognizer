import test from 'node:test';
import assert from 'node:assert/strict';
import {createCanvas} from '@napi-rs/canvas';
import {decodeCTC,roomInput} from '../src/room-model.js';
test('CTC collapses repeats but permits repeated characters separated by blank; masks non-room symbols',()=>{
 const chars='A1(-',labels=[1,1,0,1,2,2,3,0,4],data=new Float32Array(labels.length*5).fill(-10);
 labels.forEach((label,t)=>{data[t*5+label]=10;data[t*5]=0;});
 assert.equal(decodeCTC(data,labels.length,chars),'AA1-');
});
test('room input has fixed shape, normalized range and white padding',()=>{
 const roi=createCanvas(20,40),ctx=roi.getContext('2d');ctx.fillStyle='black';ctx.fillRect(0,0,20,40);
 const input=roomInput(roi,createCanvas);assert.equal(input.length,64*320);assert.equal(input[319],1);assert.ok(input.some(v=>v<-.9));assert.ok(input.every(v=>v>=-1&&v<=1));
});
