/**
 * Debug script - check render orders and drawable flags
 */
const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');

const scriptDir = __dirname;
const modelDir = process.argv[2] || '/Volumes/TAMWoolff/data/live2d/264';
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
  // Collect visible drawables sorted by renderOrder
  'var items=[];',
  'for(var i=0;i<d.count;i++){',
  '  if(!(d.dynamicFlags[i]&1)) continue;',
  '  if(d.opacities[i]<=0) continue;',
  '  var flags=d.constantFlags?d.constantFlags[i]:0;',
  // Check blend mode flags (bit 2 = multiply, bit 3 = additive in Cubism)
  '  items.push({i:i, id:d.ids[i], ro:d.renderOrders[i], op:d.opacities[i].toFixed(2), flags:flags});',
  '}',
  'items.sort(function(a,b){return a.ro-b.ro;});',
  // Show first 20 and last 20 by render order
  'var first=items.slice(0,20);',
  'var last=items.slice(-20);',
  'window.__info={',
  '  ci:{w:ci.CanvasWidth,h:ci.CanvasHeight,ppu:ci.PixelsPerUnit},',
  '  total:d.count, visible:items.length,',
  '  renderOrderRange:[items[0]&&items[0].ro, items[items.length-1]&&items[items.length-1].ro],',
  '  first20:first, last20:last,',
  '  hasConstantFlags:!!d.constantFlags,',
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
