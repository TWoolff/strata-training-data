const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');
const modelDir = process.argv[2] || '/Volumes/TAMWoolff/data/live2d/047';
const modelFiles = fs.readdirSync(modelDir).filter(f => f.endsWith('.model3.json') && !f.startsWith('.'));
const modelJson = JSON.parse(fs.readFileSync(path.join(modelDir, modelFiles[0]), 'utf8'));
const mocPath = path.join(modelDir, modelJson.FileReferences.Moc);
const mocData = fs.readFileSync(mocPath);
const cubismCoreCode = fs.readFileSync('/Users/taw/code/strata-training-data/tools/live2d_renderer/live2dcubismcore.min.js', 'utf8');
const mocBase64 = mocData.toString('base64');

const scriptBody = [
  'window.onerror=function(m,s,l,c,e){window.__error=m;};',
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
  'var ci=model.canvasinfo;',
  'var d=model.drawables;',
  'var allX=[],allY=[];',
  'for(var i=0;i<d.count;i++){',
  '  if(!(d.dynamicFlags[i]&1)) continue;',
  '  var pos=d.vertexPositions[i];',
  '  for(var v=0;v<pos.length;v+=2){allX.push(pos[v]);allY.push(pos[v+1]);}',
  '}',
  'window.__info={ci:{w:ci.CanvasWidth,h:ci.CanvasHeight,ppu:ci.PixelsPerUnit,ox:ci.CanvasOriginX,oy:ci.CanvasOriginY},',
  '  xRange:[Math.min.apply(null,allX),Math.max.apply(null,allX)],',
  '  yRange:[Math.min.apply(null,allY),Math.max.apply(null,allY)]};',
  'window.__done=true;',
  '})();',
].join('\n');

const html = '<html><body><script>' + cubismCoreCode + '</script><script>' + scriptBody + '</script></body></html>';

(async() => {
  const browser = await puppeteer.launch({headless:true,args:['--no-sandbox','--disable-setuid-sandbox']});
  const page = await browser.newPage();
  await page.setContent(html, {waitUntil:'domcontentloaded'});
  await page.waitForFunction('window.__done||window.__error', {timeout:30000});
  const info = await page.evaluate(() => window.__info);
  const err = await page.evaluate(() => window.__error);
  if(err) console.error(err);
  else console.log(JSON.stringify(info, null, 2));
  await browser.close();
})();
