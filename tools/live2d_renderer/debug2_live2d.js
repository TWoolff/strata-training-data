/**
 * Debug script - check all visible drawable position ranges
 */
const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');

const scriptDir = __dirname;
const modelDir = process.argv[2] || '/Volumes/TAMWoolff/data/live2d/001';
const modelFiles = fs.readdirSync(modelDir).filter(f => f.endsWith('.model3.json') && !f.startsWith('.'));
const modelJson = JSON.parse(fs.readFileSync(path.join(modelDir, modelFiles[0]), 'utf8'));
const mocPath = path.join(modelDir, modelJson.FileReferences.Moc);
const mocData = fs.readFileSync(mocPath);
const cubismCoreCode = fs.readFileSync(path.join(scriptDir, 'live2dcubismcore.min.js'), 'utf8');
const mocBase64 = mocData.toString('base64');

const scriptBody = [
  'window.onerror=function(msg,s,l,c,e){window.__error=msg;};',
  'function waitForSDK(){return new Promise(function(r){var chk=function(){try{Live2DCubismCore.Version.csmGetVersion();r();}catch(e){setTimeout(chk,50);}};chk();});}',
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
  'var ci=model.canvasinfo;',
  'var maxCoord = Math.max(ci.CanvasWidth, ci.CanvasHeight) / ci.PixelsPerUnit;',
  'var info=[];',
  'var globalMinX=1e9,globalMaxX=-1e9,globalMinY=1e9,globalMaxY=-1e9;',
  'var outliersFiltered=0, totalVisible=0;',
  'for(var i=0;i<d.count;i++){',
  '  if(!(d.dynamicFlags[i]&1)) continue;',
  '  if(d.opacities[i]<=0) continue;',
  '  totalVisible++;',
  '  var pos=d.vertexPositions[i];',
  '  var mnX=1e9,mxX=-1e9,mnY=1e9,mxY=-1e9;',
  '  var hasOutlier=false;',
  '  for(var v=0;v<pos.length;v+=2){',
  '    mnX=Math.min(mnX,pos[v]);mxX=Math.max(mxX,pos[v]);',
  '    mnY=Math.min(mnY,pos[v+1]);mxY=Math.max(mxY,pos[v+1]);',
  '    if(Math.abs(pos[v])>maxCoord*1.5||Math.abs(pos[v+1])>maxCoord*1.5) hasOutlier=true;',
  '  }',
  '  if(hasOutlier){outliersFiltered++; continue;}',
  '  globalMinX=Math.min(globalMinX,mnX);globalMaxX=Math.max(globalMaxX,mxX);',
  '  globalMinY=Math.min(globalMinY,mnY);globalMaxY=Math.max(globalMaxY,mxY);',
  '}',
  'window.__info={',
  '  ci:{w:ci.CanvasWidth,h:ci.CanvasHeight,ppu:ci.PixelsPerUnit,maxCoord:maxCoord.toFixed(2)},',
  '  totalDrawables:d.count,',
  '  totalVisible:totalVisible,',
  '  outliersFiltered:outliersFiltered,',
  '  bbox:{minX:globalMinX.toFixed(3),maxX:globalMaxX.toFixed(3),minY:globalMinY.toFixed(3),maxY:globalMaxY.toFixed(3)},',
  '  bboxYflip:{minY:(-globalMaxY).toFixed(3),maxY:(-globalMinY).toFixed(3)},',
  '  bboxSize:{w:(globalMaxX-globalMinX).toFixed(3),h:(globalMaxY-globalMinY).toFixed(3)},',
  '};',
  'window.__done=true;',
  '})();',
].join('\n');

const html = '<!DOCTYPE html><html><body><script>' + cubismCoreCode + '</script><script>' + scriptBody + '</script></body></html>';

(async () => {
  const browser = await puppeteer.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });
  const page = await browser.newPage();
  await page.setContent(html, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction('window.__done||window.__error', { timeout: 30000 });
  const info = await page.evaluate(() => window.__info);
  const err = await page.evaluate(() => window.__error);
  if (err) console.error('Error:', err);
  else console.log(JSON.stringify(info, null, 2));
  await browser.close();
})();
