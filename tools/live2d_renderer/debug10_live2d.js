/**
 * Debug: render only drawables with dro <= threshold, to find the layer that breaks
 */
const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');

const scriptDir = __dirname;
const modelDir = process.argv[2] || '/Volumes/TAMWoolff/data/live2d/005';
const threshold = parseInt(process.argv[3] || '400');
const modelFiles = fs.readdirSync(modelDir).filter(f => f.endsWith('.model3.json') && !f.startsWith('.'));
const modelJson = JSON.parse(fs.readFileSync(path.join(modelDir, modelFiles[0]), 'utf8'));
const texturePaths = modelJson.FileReferences.Textures.map(t => path.join(modelDir, t));
const mocPath = path.join(modelDir, modelJson.FileReferences.Moc);
const mocData = fs.readFileSync(mocPath);
const cubismCoreCode = fs.readFileSync(path.join(scriptDir, 'live2dcubismcore.min.js'), 'utf8');
const mocBase64 = mocData.toString('base64');
const texDataArr = texturePaths.filter(t => !path.basename(t).startsWith('.')).map(t => {
  try { return { b64: fs.readFileSync(t).toString('base64'), mime: 'image/png' }; } catch(e) { return null; }
}).filter(Boolean);

const VERT = `attribute vec2 aPos; attribute vec2 aUV; uniform vec2 uScale; uniform vec2 uOffset; varying vec2 vUV; void main(){vec2 p=(aPos-uOffset)*uScale; gl_Position=vec4(p.x,p.y,0.0,1.0); vUV=aUV;}`;
const FRAG = `precision mediump float; uniform sampler2D uTex; uniform float uOpacity; varying vec2 vUV; void main(){vec4 c=texture2D(uTex,vUV); gl_FragColor=vec4(c.rgb,c.a*uOpacity);}`;

const texLoadScript = texDataArr.map((t, i) =>
  `var img${i}=new Image(); await new Promise(function(r){img${i}.onload=r; img${i}.src="data:${t.mime};base64,${t.b64}";}); imgs.push(img${i});`
).join('\n');

const scriptBody = [
  'window.onerror=function(msg,s,l,c,e){window.__error=msg+" at "+s+":"+l;};',
  'function waitForSDK(){return new Promise(function(r){var chk=function(){try{Live2DCubismCore.Version.csmGetVersion();r();}catch(e){setTimeout(chk,50);}};chk();});}',
  'function compile(gl,vSrc,fSrc){',
  '  var vs=gl.createShader(gl.VERTEX_SHADER); gl.shaderSource(vs,vSrc); gl.compileShader(vs);',
  '  if(!gl.getShaderParameter(vs,gl.COMPILE_STATUS)) throw new Error("VS:"+gl.getShaderInfoLog(vs));',
  '  var fs=gl.createShader(gl.FRAGMENT_SHADER); gl.shaderSource(fs,fSrc); gl.compileShader(fs);',
  '  if(!gl.getShaderParameter(fs,gl.COMPILE_STATUS)) throw new Error("FS:"+gl.getShaderInfoLog(fs));',
  '  var p=gl.createProgram(); gl.attachShader(p,vs); gl.attachShader(p,fs); gl.linkProgram(p);',
  '  if(!gl.getProgramParameter(p,gl.LINK_STATUS)) throw new Error("Link:"+gl.getProgramInfoLog(p));',
  '  return p;',
  '}',
  '(async function(){',
  'await waitForSDK();',
  'var mocBase64="' + mocBase64 + '";',
  'var mocBytes=Uint8Array.from(atob(mocBase64),function(c){return c.charCodeAt(0);});',
  'var moc=Live2DCubismCore.Moc.fromArrayBuffer(mocBytes.buffer);',
  'var model=Live2DCubismCore.Model.fromMoc(moc);',
  'var params=model.parameters;',
  'for(var pi=0;pi<params.count;pi++) params.values[pi]=params.defaultValues[pi];',
  'model.update();',
  'var d=model.drawables;',
  'var imgs=[];',
  texLoadScript,
  'var canvas=document.getElementById("gl");',
  'var gl=canvas.getContext("webgl",{premultipliedAlpha:false,alpha:true});',
  'if(!gl){window.__error="WebGL not available"; return;}',
  'gl.viewport(0,0,512,512); gl.clearColor(0,0,0,0); gl.clear(gl.COLOR_BUFFER_BIT);',
  'gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);',
  // Upload textures
  'var textures=imgs.map(function(img){',
  '  var tex=gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D,tex);',
  '  gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,img);',
  '  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);',
  '  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);',
  '  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);',
  '  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);',
  '  return tex;',
  '});',
  'var VERT=' + JSON.stringify(VERT) + ';',
  'var FRAG=' + JSON.stringify(FRAG) + ';',
  'var prog=compile(gl,VERT,FRAG);',
  'gl.useProgram(prog);',
  // Full bbox
  'var mnX=1e9,mxX=-1e9,mnY=1e9,mxY=-1e9;',
  'for(var i=0;i<d.count;i++){',
  '  if(!(d.dynamicFlags[i]&1)) continue; if(d.opacities[i]<=0) continue;',
  '  var did=d.ids[i]; if(did.indexOf("Touch")>=0||did.indexOf("HitArea")>=0) continue;',
  '  var pos=d.vertexPositions[i];',
  '  for(var v=0;v<pos.length;v+=2){mnX=Math.min(mnX,pos[v]);mxX=Math.max(mxX,pos[v]);mnY=Math.min(mnY,pos[v+1]);mxY=Math.max(mxY,pos[v+1]);}',
  '}',
  'var cX=(mnX+mxX)/2,cY=(mnY+mxY)/2,sz=Math.max(mxX-mnX,mxY-mnY);',
  'gl.uniform2f(gl.getUniformLocation(prog,"uScale"),2/sz,2/sz);',
  'gl.uniform2f(gl.getUniformLocation(prog,"uOffset"),cX,cY);',
  'gl.uniform1i(gl.getUniformLocation(prog,"uTex"),0);',
  // Sort by drawOrders
  'var order=[]; for(var i=0;i<d.count;i++) order.push(i);',
  'order.sort(function(a,b){return d.drawOrders[a]-d.drawOrders[b];});',
  'var posBuf=gl.createBuffer(), uvBuf=gl.createBuffer(), idxBuf=gl.createBuffer();',
  'var aPos=gl.getAttribLocation(prog,"aPos"), aUV=gl.getAttribLocation(prog,"aUV"), uOp=gl.getUniformLocation(prog,"uOpacity");',
  'var threshold=' + threshold + ';',
  'var drawn=0;',
  'for(var oi=0;oi<order.length;oi++){',
  '  var idx=order[oi];',
  '  if(!(d.dynamicFlags[idx]&1)) continue; if(d.opacities[idx]<=0) continue;',
  '  if(d.drawOrders[idx]>threshold) continue;',  // THRESHOLD
  '  var texIdx=d.textureIndices[idx]; if(texIdx>=textures.length) continue;',
  '  gl.uniform1f(uOp,d.opacities[idx]);',
  '  var pos=d.vertexPositions[idx], uvs=d.vertexUvs[idx], indices=d.indices[idx];',
  '  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D,textures[texIdx]);',
  '  gl.bindBuffer(gl.ARRAY_BUFFER,posBuf); gl.bufferData(gl.ARRAY_BUFFER,pos,gl.DYNAMIC_DRAW);',
  '  gl.enableVertexAttribArray(aPos); gl.vertexAttribPointer(aPos,2,gl.FLOAT,false,0,0);',
  '  gl.bindBuffer(gl.ARRAY_BUFFER,uvBuf); gl.bufferData(gl.ARRAY_BUFFER,uvs,gl.DYNAMIC_DRAW);',
  '  gl.enableVertexAttribArray(aUV); gl.vertexAttribPointer(aUV,2,gl.FLOAT,false,0,0);',
  '  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,idxBuf); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,indices,gl.DYNAMIC_DRAW);',
  '  gl.drawElements(gl.TRIANGLES,indices.length,gl.UNSIGNED_SHORT,0);',
  '  drawn++;',
  '}',
  'window.__drawn=drawn;',
  'var out=document.getElementById("out"); var ctx2=out.getContext("2d"); ctx2.drawImage(canvas,0,0);',
  'window.__result=out.toDataURL("image/png"); window.__done=true;',
  '})();',
].join('\n');

const html = '<!DOCTYPE html><html><head><style>*{margin:0;padding:0;}</style></head><body>' +
  '<canvas id="gl" width="512" height="512"></canvas>' +
  '<canvas id="out" width="512" height="512" style="display:none"></canvas>' +
  '<script>' + cubismCoreCode + '</script>' +
  '<script>' + scriptBody + '</script>' +
  '</body></html>';

(async () => {
  const browser = await puppeteer.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--enable-webgl', '--use-gl=angle'],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 512, height: 512 });
  await page.setContent(html, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction('window.__done||window.__error', { timeout: 30000 });
  const err = await page.evaluate(() => window.__error);
  const drawn = await page.evaluate(() => window.__drawn);
  if (err) { console.error('Error:', err); process.exit(1); }
  console.log('Drawn:', drawn, 'drawables with dro <=', threshold);
  const dataUrl = await page.evaluate(() => window.__result);
  const out = '/tmp/debug10_dro' + threshold + '.png';
  fs.writeFileSync(out, Buffer.from(dataUrl.split(',')[1], 'base64'));
  console.log('Saved', out);
  await browser.close();
})();
