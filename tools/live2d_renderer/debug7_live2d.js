/**
 * Debug: check masking/clipping in the model
 */
const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');

const scriptDir = __dirname;
const modelDir = process.argv[2] || '/Volumes/TAMWoolff/data/live2d/005';
const modelFiles = fs.readdirSync(modelDir).filter(f => f.endsWith('.model3.json') && !f.startsWith('.'));
const modelJson = JSON.parse(fs.readFileSync(path.join(modelDir, modelFiles[0]), 'utf8'));
const mocPath = path.join(modelDir, modelJson.FileReferences.Moc);
const mocData = fs.readFileSync(mocPath);
const cubismCoreCode = fs.readFileSync(path.join(scriptDir, 'live2dcubismcore.min.js'), 'utf8');
const mocBase64 = mocData.toString('base64');

const scriptBody = [
  'window.onerror=function(msg,s,l,c,e){window.__error=msg+" at "+s+":"+l;};',
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
  // Check masking-related properties
  'var hasMasks=false;',
  'var maskedCount=0;',
  'var maskInfo=[];',
  'for(var i=0;i<d.count;i++){',
  '  var masks=d.masks?d.masks[i]:null;',
  '  if(masks&&masks.length>0){hasMasks=true; maskedCount++; if(maskInfo.length<10) maskInfo.push({id:d.ids[i],masks:Array.from(masks),ro:d.renderOrders[i]});}',
  '}',
  // Check all available properties on drawables
  'var propNames=Object.keys(d);',
  'window.__info={',
  '  hasMasks:hasMasks,',
  '  maskedCount:maskedCount,',
  '  totalDrawables:d.count,',
  '  drawableProps:propNames,',
  '  maskInfo:maskInfo,',
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
