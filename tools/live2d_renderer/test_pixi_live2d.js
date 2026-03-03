/**
 * Test: use pixi-live2d-display with pixi v6 to render a Live2D model properly
 */
const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');

const scriptDir = __dirname;
const modelDir = process.argv[2] || '/Volumes/TAMWoolff/data/live2d/047';
const modelFiles = fs.readdirSync(modelDir).filter(f => f.endsWith('.model3.json') && !f.startsWith('.'));
const modelJsonPath = path.join(modelDir, modelFiles[0]);
const modelJson = JSON.parse(fs.readFileSync(modelJsonPath, 'utf8'));

const BASE_URL = 'https://live2d.model/';
const MODEL_URL = BASE_URL + modelFiles[0];

// Build resource map
const resMap = {};
resMap[MODEL_URL] = { data: Buffer.from(JSON.stringify(modelJson)), mime: 'application/json' };
const mocRel = modelJson.FileReferences.Moc;
resMap[BASE_URL + mocRel] = { data: fs.readFileSync(path.join(modelDir, mocRel)), mime: 'application/octet-stream' };
for (const texRel of modelJson.FileReferences.Textures) {
  try {
    resMap[BASE_URL + texRel] = { data: fs.readFileSync(path.join(modelDir, texRel)), mime: 'image/png' };
  } catch(e) {}
}

const pixiCode = fs.readFileSync(path.join(scriptDir, 'node_modules/pixi.js/dist/browser/pixi.js'), 'utf8');
const cubism4Code = fs.readFileSync(path.join(scriptDir, 'node_modules/pixi-live2d-display/dist/cubism4.js'), 'utf8');
const cubismCoreCode = fs.readFileSync(path.join(scriptDir, 'live2dcubismcore.min.js'), 'utf8');

const SIZE = 512;

// Build the HTML page — to be served at a proper URL to avoid CORS null-origin issues
// We'll serve it from the same fake origin
const HTML_URL = BASE_URL + 'index.html';
const htmlContent = `<!DOCTYPE html><html><head>
<style>*{margin:0;padding:0;}body{background:transparent;}</style>
</head><body>
<canvas id="canvas" width="${SIZE}" height="${SIZE}"></canvas>
<script src="${BASE_URL}cubismcore.js"></script>
<script src="${BASE_URL}pixi.js"></script>
<script src="${BASE_URL}cubism4.js"></script>
<script>
window.onerror = function(m,s,l,c,e){ window.__error = m + ' at ' + s + ':' + l; };
(async function(){
try {
  await new Promise(function(r){
    var chk=function(){try{Live2DCubismCore.Version.csmGetVersion();r();}catch(e){setTimeout(chk,50);}};
    chk();
  });
  
  var app = new PIXI.Application({
    view: document.getElementById('canvas'),
    width: ${SIZE}, height: ${SIZE},
    backgroundAlpha: 0,
    antialias: false,
  });
  
  var model = await PIXI.live2d.Live2DModel.from('${MODEL_URL}');
  
  app.stage.addChild(model);
  
  // Fit to canvas — use scale to fit
  var scaleX = ${SIZE} / model.width;
  var scaleY = ${SIZE} / model.height;
  var sc = Math.min(scaleX, scaleY) * 0.9;
  model.scale.set(sc);
  model.x = Math.round((${SIZE} - model.width) / 2);
  model.y = Math.round((${SIZE} - model.height) / 2);
  
  app.render();
  
  window.__result = document.getElementById('canvas').toDataURL('image/png');
  window.__modelW = model.width;
  window.__modelH = model.height;
  window.__done = true;
} catch(e) {
  window.__error = e.message + ' | ' + e.stack.split('\\n').slice(0,3).join(' | ');
}
})();
</script></body></html>`;

resMap[HTML_URL] = { data: Buffer.from(htmlContent), mime: 'text/html' };
resMap[BASE_URL + 'cubismcore.js'] = { data: Buffer.from(cubismCoreCode), mime: 'application/javascript' };
resMap[BASE_URL + 'pixi.js'] = { data: Buffer.from(pixiCode), mime: 'application/javascript' };
resMap[BASE_URL + 'cubism4.js'] = { data: Buffer.from(cubism4Code), mime: 'application/javascript' };

const corsHeaders = { 'Access-Control-Allow-Origin': '*' };

(async () => {
  const browser = await puppeteer.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--enable-webgl', '--use-gl=angle'],
  });
  const page = await browser.newPage();
  
  await page.setRequestInterception(true);
  page.on('request', req => {
    const url = req.url();
    if (resMap[url]) {
      const { data, mime } = resMap[url];
      req.respond({ status: 200, contentType: mime, body: data, headers: corsHeaders });
    } else if (url.startsWith(BASE_URL)) {
      // Fallback: try to serve as file
      const relPath = url.replace(BASE_URL, '');
      const fullPath = path.join(modelDir, relPath);
      try {
        const data = fs.readFileSync(fullPath);
        req.respond({ status: 200, contentType: 'application/octet-stream', body: data, headers: corsHeaders });
      } catch(e) {
        req.respond({ status: 404, body: 'Not found', headers: corsHeaders });
      }
    } else {
      req.continue();
    }
  });
  
  page.on('console', msg => {
    if (msg.type() === 'error') console.error('PAGE:', msg.text().split('\n')[0]);
  });
  
  await page.setViewport({ width: SIZE, height: SIZE });
  await page.goto(HTML_URL, { waitUntil: 'domcontentloaded' });
  
  try {
    await page.waitForFunction('window.__done || window.__error', { timeout: 30000 });
  } catch(e) {
    console.error('Timeout waiting for render');
    await browser.close();
    process.exit(1);
  }
  
  const err = await page.evaluate(() => window.__error);
  if (err) {
    console.error('Render error:', err.split(' | ')[0]);
    await browser.close();
    process.exit(1);
  }
  
  const mw = await page.evaluate(() => window.__modelW);
  const mh = await page.evaluate(() => window.__modelH);
  console.log('Model:', mw, 'x', mh);
  
  const dataUrl = await page.evaluate(() => window.__result);
  const outPath = '/tmp/test_pixi_live2d_' + path.basename(modelDir) + '.png';
  fs.writeFileSync(outPath, Buffer.from(dataUrl.split(',')[1], 'base64'));
  console.log('Saved:', outPath);
  
  await browser.close();
})();
