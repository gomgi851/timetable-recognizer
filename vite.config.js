import {defineConfig} from 'vite';
import fs from 'node:fs/promises';
import path from 'node:path';
export default defineConfig({plugins:[{name:'local-evaluation-samples',configureServer(server){server.middlewares.use(async(req,res,next)=>{try{if(req.url==='/__local-samples'){const names=(await fs.readdir('samples/private')).filter(n=>/^\d+\.png$/.test(n));res.setHeader('Content-Type','application/json');res.end(JSON.stringify(names));return;}const match=req.url?.match(/^\/__local-samples\/(\d+\.png)$/);if(match){res.setHeader('Content-Type','image/png');res.end(await fs.readFile(path.join('samples/private',match[1])));return;}next();}catch{res.statusCode=404;res.end('Local sample unavailable');}});}}]});
