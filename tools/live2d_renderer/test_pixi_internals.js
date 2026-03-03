const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');

const scriptDir = __dirname;
const modelDir = process.argv[2] || '/Volumes/TAMWoolff/data/live2d/047';
const modelFiles = fs.readdirSync(modelDir).filter(f => f.endsWith('.model3.json') && !f.startsWith('.'));
const modelJson = JSON.parse(fs.readFileSync(path.join(modelDir, modelFiles[0]), 'utf8'));

const BASE_URL = 'https://live2d.model/';
const MODEL_URL = BASE_URL + modelFiles[0];

const resMap = {};
resMap[MODEL_URL] = { data: Buffer.from(JSON.stringify(modelJson)), mime: 'application/json' };
const mocRel = modelJson.FileReferences.Moc;
resMap[BASE_URL + mocRel] = { data: fs.readFileSync(path.join(modelDir, mocRel)), mime: 'application/octet-stream' };
for (const texRel of modelJson.FileReferences.Textures) {
  try { resMap[BASE_URL + texRel] = { data: fs.readFileSync(path.join(modelDir, texRel)), mime: 'image/png' }; } catch(e) {}
}

const pixiCode = fs.readFileSync(path.join(scriptDir, 'node_modules/pixi.js/dist/browser/pixi.js'), 'utf8');
const cubism4Code = fs.readFileSync(path.join(scriptDir, 'node_modules/pixi-live2d-display/dist/cubism4.js'), 'utf8');
const cubismCoreCode = fs.readFileSync(path.join(scriptDir, 'live2dcubismcore.min.js'), 'utf8');

const SIZE = 512;

const htmlContent = `<!DOCTYPE html><html><body>
<canvas id="canvas" width="${SIZE}" height="${SIZE}"></canvas>
<script src="${BASE_URL}cubismcore.js"></script>
<script src="${BASE_URL}pixi.js"></script>
<script src="${BASE_URL}cubism4.js"></script>
<script>
window.onerror = function(m,s,l,c,e){ window.__error = m; };
(async function(){
try {
  await new Promise(function(r){var chk=function(){try{Live2DCubismCore.Version.csmGetVersion();r();}catch(e){setTimeout(chk,50);}};chk();});
  var app = new PIXI.Application({view:document.getElementById('canvas'),width:${SIZE},height:${SIZE},backgroundAlpha:0});
  var model = await PIXI.live2d.Live2DModel.from('${MODEL_URL}');
  app.stage.addChild(model);
  app.render();
  
  var im = model.internalModel;
  var info = {
    imKeys: Object.keys(im),
  };
  
  // Try coreModel
  if(im.coreModel) {
    var cm = im.coreModel;
    info.coreModelType = cm.constructor.name;
    info.coreModelKeys = Object.keys(cm).slice(0,30);
    // Try _model (internal Live2DCubismCore model)
    if(cm._model) {
      var d = cm._model.drawables;
      info.drawableCount = d.count;
      // Get first 3 drawables' vertex positions
      info.samplePositions = [];
      for(var i=0;i<Math.min(3,d.count);i++) {
        var pos = d.vertexPositions[i];
        info.samplePositions.push({
          id: d.ids[i],
          visible: !!(d.dynamicFlags[i]&1),
          positions: pos ? Array.from(pos).slice(0,8) : null,
        });
      }
    }
  }
  
  window.__info = info;
  window.__done = true;
} catch(e) {
  window.__error = e.message + ' // ' + e.stack.split('\\n').slice(0,2).join(' // ');
}
})();
</script></body></html>`;

const HTML_URL = BASE_URL + 'index.html';
resMap[HTML_URL] = { data: Buffer.from(htmlContent), mime: 'text/html' };
resMap[BASE_URL + 'cubismcore.js'] = { data: Buffer.from(cubismCoreCode), mime: 'application/javascript' };
resMap[BASE_URL + 'pixi.js'] = { data: Buffer.from(pixiCode), mime: 'application/javascript' };
resMap[BASE_URL + 'cubism4.js'] = { data: Buffer.from(cubism4Code), mime: 'application/javascript' };

const corsHeaders = { 'Access-Control-Allow-Origin': '*' };

(async () => {
  const browser = await puppeteer.launch({headless:true,args:['--no-sandbox','--disable-setuid-sandbox','--enable-webgl','--use-gl=angle']});
  const page = await browser.newPage();
  await page.setRequestInterception(true);
  page.on('request', req => {
    const url = req.url();
    if(resMap[url]) req.respond({status:200,contentType:resMap[url].mime,body:resMap[url].data,headers:corsHeaders});
    else if(url.startsWith(BASE_URL)) req.respond({status:404,headers:corsHeaders});
    else req.continue();
  });
  await page.goto(HTML_URL, {waitUntil:'domcontentloaded'});
  await page.waitForFunction('window.__done||window.__error',{timeout:30000});
  const err = await page.evaluate(() => window.__error);
  const info = await page.evaluate(() => window.__info);
  if(err) console.error('Error:', err);
  else console.log(JSON.stringify(info, null, 2));
  await browser.close();
})();
