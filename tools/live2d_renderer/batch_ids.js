const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');

const scriptDir = '/Users/taw/code/strata-training-data/tools/live2d_renderer';
const liveDir = '/Volumes/TAMWoolff/data/live2d';
const cubismCoreCode = fs.readFileSync(path.join(scriptDir, 'live2dcubismcore.min.js'), 'utf8');

const models = fs.readdirSync(liveDir)
  .filter(f => !f.startsWith('.') && f !== 'README.md' && f !== 'labels')
  .sort();

const allIds = {};

async function processModel(modelName) {
  const modelDir = path.join(liveDir, modelName);
  if (!fs.statSync(modelDir).isDirectory()) return;
  const modelFiles = fs.readdirSync(modelDir).filter(f => f.endsWith('.model3.json') && !f.startsWith('.'));
  if (!modelFiles.length) return;
  const modelJson = JSON.parse(fs.readFileSync(path.join(modelDir, modelFiles[0]), 'utf8'));
  const mocRel = modelJson.FileReferences.Moc;
  let mocData;
  try { mocData = fs.readFileSync(path.join(modelDir, mocRel)); } catch(e) { return; }
  const mocBase64 = mocData.toString('base64');

  const script = [
    'window.onerror=function(m){window.__error=m;};',
    'function waitForSDK(){return new Promise(function(r){var chk=function(){try{Live2DCubismCore.Version.csmGetVersion();r();}catch(e){setTimeout(chk,50);}};chk();});}',
    '(async function(){',
    'await waitForSDK();',
    'var mb="' + mocBase64 + '";',
    'var by=Uint8Array.from(atob(mb),function(c){return c.charCodeAt(0);});',
    'var moc=Live2DCubismCore.Moc.fromArrayBuffer(by.buffer);',
    'var model=Live2DCubismCore.Model.fromMoc(moc);',
    'var params=model.parameters;',
    'for(var i=0;i<params.count;i++) params.values[i]=params.defaultValues[i];',
    'model.update();',
    'var d=model.drawables;',
    'window.__ids=Array.from(d.ids);',
    'window.__done=true;',
    '})();',
  ].join('\n');

  const html = '<html><body><script>' + cubismCoreCode + '</script><script>' + script + '</script></body></html>';

  const browser = await puppeteer.launch({headless:true,args:['--no-sandbox','--disable-setuid-sandbox']});
  const page = await browser.newPage();
  await page.setContent(html, {waitUntil:'domcontentloaded'});
  try {
    await page.waitForFunction('window.__done||window.__error',{timeout:15000});
    const ids = await page.evaluate(() => window.__ids);
    if (ids) allIds[modelName] = ids;
  } catch(e) {}
  await browser.close();
}

(async () => {
  // Process in batches of 5 concurrently
  for (let i = 0; i < models.length; i += 5) {
    const batch = models.slice(i, i+5);
    await Promise.all(batch.map(m => processModel(m).catch(e => {})));
    process.stderr.write('Processed ' + Math.min(i+5, models.length) + '/' + models.length + '\r');
  }
  fs.writeFileSync('/tmp/all_drawable_ids.json', JSON.stringify(allIds));
  console.log('Done. Models processed:', Object.keys(allIds).length);
})();
