/**
 * Debug: compare drawOrders vs renderOrders
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
  // Collect both order arrays for visible drawables, sorted by renderOrder
  'var items=[];',
  'for(var i=0;i<d.count;i++){',
  '  if(!(d.dynamicFlags[i]&1)) continue;',
  '  if(d.opacities[i]<=0) continue;',
  '  items.push({i:i,id:d.ids[i],ro:d.renderOrders[i],dro:d.drawOrders[i]});',
  '}',
  // Sort by renderOrder
  'items.sort(function(a,b){return a.ro-b.ro;});',
  // Show first 20 and last 20
  'var first=items.slice(0,20);',
  'var last=items.slice(-20);',
  // Check if drawOrders differs from renderOrders
  'var differ=items.filter(function(x){return x.ro!==x.dro;}).length;',
  'window.__info={',
  '  total:items.length,differ:differ,',
  '  sortedByRO_first:first,sortedByRO_last:last,',
  '};',
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
  else {
    console.log('total visible:', info.total, '  drawOrders != renderOrders:', info.differ);
    console.log('\nFirst 20 (back→front by renderOrder):');
    info.sortedByRO_first.forEach(x => console.log(`  i=${String(x.i).padStart(3)} ro=${String(x.ro).padStart(4)} dro=${String(x.dro).padStart(4)} id=${x.id}`));
    console.log('\nLast 20 (front):');
    info.sortedByRO_last.forEach(x => console.log(`  i=${String(x.i).padStart(3)} ro=${String(x.ro).padStart(4)} dro=${String(x.dro).padStart(4)} id=${x.id}`));
  }
  await browser.close();
})();
