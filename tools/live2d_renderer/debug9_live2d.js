/**
 * Debug: identify the mask drawables and masked drawables
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
  // Find mask source drawables (indices 77 and 78)
  'var info={};',
  'info.maskSrc77={i:77,id:d.ids[77],ro:d.renderOrders[77],dro:d.drawOrders[77],op:d.opacities[77],vis:!!(d.dynamicFlags[77]&1)};',
  'info.maskSrc78={i:78,id:d.ids[78],ro:d.renderOrders[78],dro:d.drawOrders[78],op:d.opacities[78],vis:!!(d.dynamicFlags[78]&1)};',
  // Find all drawables that are masked, and what their constantFlags say
  'info.maskedDrawables=[];',
  'for(var i=0;i<d.count;i++){',
  '  var m=d.masks[i];',
  '  if(m&&m.length>0){',
  '    var cf=d.constantFlags?d.constantFlags[i]:0;',
  '    info.maskedDrawables.push({i:i,id:d.ids[i],ro:d.renderOrders[i],dro:d.drawOrders[i],masks:Array.from(m),constantFlags:cf,isInverted:!!(cf&8)});',
  '  }',
  '}',
  // Also check what constantFlags tells us about the mask SOURCES
  'info.maskSrc77.cf=d.constantFlags?d.constantFlags[77]:0;',
  'info.maskSrc78.cf=d.constantFlags?d.constantFlags[78]:0;',
  'window.__info=info;',
  'window.__done=true;',
  '})();',
].join('\n');

const html = '<!DOCTYPE html><html><body><script>' + cubismCoreCode + '</script><script>' + scriptBody + '</script></body></html>';

(async () => {
  const browser = await puppeteer.launch({ headless: true, args: ['--no-sandbox', '--disable-setuid-sandbox'] });
  const page = await browser.newPage();
  await page.setContent(html, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction('window.__done||window.__error', { timeout: 30000 });
  const info = await page.evaluate(() => window.__info);
  const err = await page.evaluate(() => window.__error);
  if (err) console.error('Error:', err);
  else console.log(JSON.stringify(info, null, 2));
  await browser.close();
})();
