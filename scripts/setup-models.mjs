import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
await fs.mkdir('public/models',{recursive:true});await fs.mkdir('public/ocr',{recursive:true});
for(const lang of ['kor','eng']) {
 const dest=`public/models/${lang}.traineddata`;
 try{await fs.access(dest);}catch{
  const url=`https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/${lang}.traineddata`;
  console.log(`Downloading ${lang} from official tessdata_best...`);
  const response=await fetch(url);if(!response.ok)throw new Error(`${url}: ${response.status}`);
  await fs.writeFile(dest,new Uint8Array(await response.arrayBuffer()));
 }
}
await fs.copyFile('node_modules/tesseract.js/dist/worker.min.js','public/ocr/worker.min.js');
for(const name of ['tesseract-core-relaxedsimd-lstm.wasm','tesseract-core-relaxedsimd-lstm.wasm.js'])await fs.rm(path.join('public/ocr',name),{force:true});
for(const name of await fs.readdir('node_modules/tesseract.js-core'))if(/-lstm\.wasm(\.js)?$/.test(name))await fs.copyFile(path.join('node_modules/tesseract.js-core',name),path.join('public/ocr',name));
const assets=[];for(const dir of ['public/models','public/ocr'])for(const name of await fs.readdir(dir)){const b=await fs.readFile(path.join(dir,name));assets.push({file:`${dir}/${name}`,bytes:b.length,sha256:createHash('sha256').update(b).digest('hex')});}
await fs.mkdir('public/licenses',{recursive:true});
await fs.copyFile('node_modules/tesseract.js/LICENSE.md','public/licenses/Tesseract.js-APACHE-2.0.txt');
const licenseResponse=await fetch('https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/LICENSE');
if(!licenseResponse.ok)throw new Error('Could not retrieve tessdata_best license');
await fs.writeFile('public/licenses/tessdata_best-APACHE-2.0.txt',await licenseResponse.text());
await fs.writeFile('public/licenses/NOTICE.txt','OCR: Tesseract.js (https://github.com/naptha/tesseract.js), Tesseract.js-core and Tesseract OCR. Models: tessdata_best (https://github.com/tesseract-ocr/tessdata_best). These components are distributed under the Apache License 2.0; see accompanying license files.');
await fs.writeFile('model-assets.json',JSON.stringify({source:'https://github.com/tesseract-ocr/tessdata_best',license:'Apache-2.0',assets},null,2)+'\n');
console.log(JSON.stringify(assets.map(({file,bytes})=>({file,bytes}))));
