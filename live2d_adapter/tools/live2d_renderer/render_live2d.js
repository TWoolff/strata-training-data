/**
 * Live2D model renderer using Puppeteer + pixi-live2d-display (pixi v6).
 *
 * Uses pixi-live2d-display which correctly evaluates the full Cubism Framework
 * deformer hierarchy (warp deformers, rotation deformers). The previous approach
 * using Cubism Core only did not evaluate deformers, producing exploded vertices.
 *
 * Usage: node render_live2d.js <model_dir> <output_png> [resolution] [seg_map_json]
 *
 * seg_map_json: optional path to JSON file mapping mesh_id -> region_id (0-21).
 *   If provided, also outputs <output_png>.seg.png with flat region colors.
 */

const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');

const modelDir = process.argv[2];
const outputPng = process.argv[3];
const resolution = parseInt(process.argv[4] || '512');
const segMapPath = process.argv[5] || null;

if (!modelDir || !outputPng) {
  console.error('Usage: node render_live2d.js <model_dir> <output_png> [resolution] [seg_map_json]');
  process.exit(1);
}

// Find the .model3.json file
const modelFiles = fs.readdirSync(modelDir)
  .filter(f => f.endsWith('.model3.json') && !f.startsWith('.'));
if (modelFiles.length === 0) {
  console.error('No .model3.json found in', modelDir);
  process.exit(1);
}

const modelJsonPath = path.join(modelDir, modelFiles[0]);
const modelJson = JSON.parse(fs.readFileSync(modelJsonPath, 'utf8'));
const segMap = segMapPath ? JSON.parse(fs.readFileSync(segMapPath, 'utf8')) : null;

// Load JS bundles
const scriptDir = __dirname;
const cubismCoreCode = fs.readFileSync(path.join(scriptDir, 'live2dcubismcore.min.js'), 'utf8');
const pixiCode = fs.readFileSync(path.join(scriptDir, 'node_modules/pixi.js/dist/browser/pixi.js'), 'utf8');
const cubism4Code = fs.readFileSync(path.join(scriptDir, 'node_modules/pixi-live2d-display/dist/cubism4.js'), 'utf8');

// Fake base URL - all model files served via request interception from this origin
const BASE_URL = 'https://live2d.model/';
const MODEL_URL = BASE_URL + modelFiles[0];

// Build resource map for request interception
const resMap = {};
resMap[MODEL_URL] = { data: Buffer.from(JSON.stringify(modelJson)), mime: 'application/json' };

const mocRel = modelJson.FileReferences.Moc;
try {
  resMap[BASE_URL + mocRel] = { data: fs.readFileSync(path.join(modelDir, mocRel)), mime: 'application/octet-stream' };
} catch(e) {
  console.error('Cannot read moc:', mocRel);
  process.exit(1);
}

for (const texRel of modelJson.FileReferences.Textures) {
  const texPath = path.join(modelDir, texRel);
  try {
    resMap[BASE_URL + texRel] = { data: fs.readFileSync(texPath), mime: 'image/png' };
  } catch(e) {
    console.warn('Missing texture:', texRel);
  }
}

// ---------------------------------------------------------------------------
// Segmentation fragment shader: flat grayscale region ID masked by texture alpha.
// uRegionId is the raw region ID (1–21). We encode as pixel value = region_id * 10
// so region 1 → R=10, region 21 → R=210. Background (rid=0) drawables are skipped.
// Python decodes: region_id = round(R / 10).
const FRAG_SEG_SRC = `
  precision highp float;
  uniform sampler2D uTex;
  uniform float uRegionId;
  varying vec2 vUV;
  void main() {
    float a = texture2D(uTex, vUV).a;
    if (a < 0.05) discard;
    float v = uRegionId * 10.0 / 255.0;
    gl_FragColor = vec4(v, v, v, 1.0);
  }
`;

// Segmentation vertex shader: uses the same Cubism → pixi → NDC transform.
// Positions are Cubism Core model units (post-deform, Y-up, Live2D convention).
//
// Full transform chain:
//   1. Cubism model units → canvas pixels:
//      canvas_x = pos_x * ppu + originX
//      canvas_y = -pos_y * ppu + originY   (Y-flip: Cubism Y-up → canvas Y-down)
//   2. Canvas pixels → pixi screen pixels:
//      pixi_x = canvas_x * modelScale + modelX
//      pixi_y = canvas_y * modelScale + modelY
//   3. Pixi screen pixels → WebGL NDC (Y-up [-1,1]):
//      ndcX = pixi_x / halfSize - 1
//      ndcY = -(pixi_y / halfSize - 1)  = 1 - pixi_y / halfSize
//
// Uniforms:
//   uPPU: PixelsPerUnit (from canvasinfo)
//   uOriginX, uOriginY: CanvasOriginX, CanvasOriginY (from canvasinfo)
//   uModelScale: pixi model.scale.x
//   uModelOffsetX, uModelOffsetY: pixi model.x, model.y
//   uHalfSize: RENDER_SIZE / 2.0
const VERT_SEG_SRC = `
  attribute vec2 aPos;
  attribute vec2 aUV;
  uniform float uPPU;
  uniform float uOriginX;
  uniform float uOriginY;
  uniform float uModelScale;
  uniform float uModelOffsetX;
  uniform float uModelOffsetY;
  uniform float uHalfSize;
  varying vec2 vUV;
  void main() {
    float cx = aPos.x * uPPU + uOriginX;
    float cy = -aPos.y * uPPU + uOriginY;
    float px = cx * uModelScale + uModelOffsetX;
    float py = cy * uModelScale + uModelOffsetY;
    float ndcX = px / uHalfSize - 1.0;
    float ndcY = 1.0 - py / uHalfSize;
    gl_Position = vec4(ndcX, ndcY, 0.0, 1.0);
    vUV = aUV;
  }
`;

// ---------------------------------------------------------------------------
const SIZE = resolution;
const RENDER_SIZE = SIZE * 2;  // render at 2× for better pixel bbox accuracy

// Build the HTML page content
function buildPageHtml(modelUrl, segMapJson) {
  const parts = [];
  parts.push('<!DOCTYPE html><html><head>');
  parts.push('<style>*{margin:0;padding:0;}body{background:transparent;}</style>');
  parts.push('</head><body>');
  parts.push('<canvas id="canvas" width="' + SIZE + '" height="' + SIZE + '"></canvas>');
  parts.push('<script src="' + BASE_URL + 'cubismcore.js"></script>');
  parts.push('<script src="' + BASE_URL + 'pixi.js"></script>');
  parts.push('<script src="' + BASE_URL + 'cubism4.js"></script>');
  parts.push('<script>');
  parts.push('window.onerror = function(m,s,l,c,e){ window.__error = m + " at " + s + ":" + l; };');

  // Wait for Cubism Core WASM to initialize
  parts.push('function waitForSDK(){');
  parts.push('  return new Promise(function(r){');
  parts.push('    var chk=function(){try{Live2DCubismCore.Version.csmGetVersion();r();}catch(e){setTimeout(chk,50);}};');
  parts.push('    chk();');
  parts.push('  });');
  parts.push('}');

  // Compile a standalone WebGL program (for seg pass)
  parts.push('function compileProgram(gl, vSrc, fSrc){');
  parts.push('  var vs=gl.createShader(gl.VERTEX_SHADER); gl.shaderSource(vs,vSrc); gl.compileShader(vs);');
  parts.push('  if(!gl.getShaderParameter(vs,gl.COMPILE_STATUS)) throw new Error("VS:"+gl.getShaderInfoLog(vs));');
  parts.push('  var fs=gl.createShader(gl.FRAGMENT_SHADER); gl.shaderSource(fs,fSrc); gl.compileShader(fs);');
  parts.push('  if(!gl.getShaderParameter(fs,gl.COMPILE_STATUS)) throw new Error("FS:"+gl.getShaderInfoLog(fs));');
  parts.push('  var p=gl.createProgram(); gl.attachShader(p,vs); gl.attachShader(p,fs); gl.linkProgram(p);');
  parts.push('  if(!gl.getProgramParameter(p,gl.LINK_STATUS)) throw new Error("Link:"+gl.getProgramInfoLog(p));');
  parts.push('  return p;');
  parts.push('}');

  // Find pixel bounding box of non-transparent content
  parts.push('function pixelBbox(ctx2d, w, h, thresh){');
  parts.push('  var data=ctx2d.getImageData(0,0,w,h).data;');
  parts.push('  var minX=w,maxX=0,minY=h,maxY=0;');
  parts.push('  for(var y=0;y<h;y++){for(var x=0;x<w;x++){');
  parts.push('    if(data[(y*w+x)*4+3]>thresh){');
  parts.push('      if(x<minX)minX=x; if(x>maxX)maxX=x;');
  parts.push('      if(y<minY)minY=y; if(y>maxY)maxY=y;');
  parts.push('    }');
  parts.push('  }}');
  parts.push('  if(minX>maxX) return null;');
  parts.push('  return {x:minX,y:minY,w:maxX-minX+1,h:maxY-minY+1};');
  parts.push('}');

  // Crop canvas to bbox + padding, scale to outSize, return {canvas, params}
  // smoothing: set to false for seg masks (nearest-neighbor, no interpolation)
  parts.push('function cropAndScale(srcCanvas, bbox, pad, outSize, smoothing){');
  parts.push('  if(smoothing===undefined) smoothing=true;');
  parts.push('  var px=Math.max(0,bbox.x-pad), py=Math.max(0,bbox.y-pad);');
  parts.push('  var pw=Math.min(srcCanvas.width,bbox.x+bbox.w+pad)-px;');
  parts.push('  var ph=Math.min(srcCanvas.height,bbox.y+bbox.h+pad)-py;');
  parts.push('  var out=document.createElement("canvas"); out.width=outSize; out.height=outSize;');
  parts.push('  var ctx=out.getContext("2d");');
  parts.push('  ctx.imageSmoothingEnabled=smoothing;');
  parts.push('  ctx.clearRect(0,0,outSize,outSize);');
  parts.push('  var scale=outSize/Math.max(pw,ph);');
  parts.push('  var dw=Math.round(pw*scale), dh=Math.round(ph*scale);');
  parts.push('  var dx=Math.round((outSize-dw)/2), dy=Math.round((outSize-dh)/2);');
  parts.push('  ctx.drawImage(srcCanvas, px,py,pw,ph, dx,dy,dw,dh);');
  parts.push('  return {canvas:out, cropX:px,cropY:py,cropW:pw,cropH:ph,scale:scale,dx:dx,dy:dy};');
  parts.push('}');

  // Main async function
  parts.push('(async function(){');
  parts.push('try{');
  parts.push('  await waitForSDK();');

  // Create pixi app at RENDER_SIZE (larger intermediate canvas)
  parts.push('  var app = new PIXI.Application({');
  parts.push('    width:' + RENDER_SIZE + ',height:' + RENDER_SIZE + ',');
  parts.push('    backgroundAlpha:0,');
  parts.push('    antialias:false,');
  parts.push('    autoDensity:false,');
  parts.push('  });');
  parts.push('  document.body.appendChild(app.view);');
  parts.push('  app.view.id="pixi_canvas";');

  // Load the model via pixi-live2d-display
  parts.push('  var model = await PIXI.live2d.Live2DModel.from("' + modelUrl + '");');

  // Scale and center in the RENDER_SIZE canvas
  parts.push('  app.stage.addChild(model);');
  parts.push('  var sc = Math.min(' + RENDER_SIZE + '/model.width, ' + RENDER_SIZE + '/model.height);');
  // Use 1.0 scale (not 0.9) — we crop based on actual pixels anyway
  parts.push('  model.scale.set(sc);');
  parts.push('  model.x = (' + RENDER_SIZE + ' - model.width) / 2;');
  parts.push('  model.y = (' + RENDER_SIZE + ' - model.height) / 2;');

  // Render one frame
  parts.push('  app.render();');

  // Read pixels from pixi canvas into 2D canvas for pixel bbox
  parts.push('  var pixiCanvas = app.view;');
  parts.push('  var mid = document.createElement("canvas");');
  parts.push('  mid.width=' + RENDER_SIZE + '; mid.height=' + RENDER_SIZE + ';');
  parts.push('  var midCtx = mid.getContext("2d");');
  parts.push('  midCtx.drawImage(pixiCanvas, 0, 0);');

  // Find tight pixel bbox
  parts.push('  var bbox = pixelBbox(midCtx, ' + RENDER_SIZE + ', ' + RENDER_SIZE + ', 8);');
  parts.push('  if(!bbox) bbox = {x:0,y:0,w:' + RENDER_SIZE + ',h:' + RENDER_SIZE + '};');
  parts.push('  var padPx = Math.round(Math.max(bbox.w,bbox.h)*0.02);');

  // Crop and scale to output SIZE
  parts.push('  var cropped = cropAndScale(mid, bbox, padPx, ' + SIZE + ');');
  parts.push('  window.__result = cropped.canvas.toDataURL("image/png");');

  // Store crop transform for Python segmentation
  // pixiCanvas pixel → output pixel:
  //   px_out = (px_pixi - cropX) * scale + dx
  //   py_out = (py_pixi - cropY) * scale + dy
  // model space → pixiCanvas pixel:
  //   pixi applies: px_pixi = mx * sc + model.x  (where sc = model.scale.x)
  //                 py_pixi = -my * sc + model.y  (Y flipped: Live2D Y-up → pixi Y-down)
  // Export drawable metadata from the (now correctly deformed) core model
  // These are post-deform vertex positions used for segmentation mask building
  parts.push('  var im = model.internalModel;');
  parts.push('  var cm = im ? im.coreModel : null;');
  parts.push('  var d = (cm && cm._model) ? cm._model.drawables : null;');
  // Collect canvas info for the Python draw-order/seg transform
  parts.push('  var _ci2 = (cm && cm._model) ? cm._model.canvasinfo : null;');
  parts.push('  var cropInfo = {');
  parts.push('    renderSize: ' + RENDER_SIZE + ',');
  parts.push('    modelX: model.x, modelY: model.y,');
  parts.push('    modelScale: sc,');
  parts.push('    cropX: cropped.cropX, cropY: cropped.cropY,');
  parts.push('    cropW: cropped.cropW, cropH: cropped.cropH,');
  parts.push('    scale: cropped.scale,');
  parts.push('    dx: cropped.dx, dy: cropped.dy,');
  parts.push('    outSize: ' + SIZE + ',');
  parts.push('    pixelsPerUnit: _ci2 ? _ci2.PixelsPerUnit : 1,');
  parts.push('    canvasOriginX: _ci2 ? _ci2.CanvasOriginX : 0,');
  parts.push('    canvasOriginY: _ci2 ? _ci2.CanvasOriginY : 0,');
  parts.push('  };');
  parts.push('  var p = (cm && cm._model) ? cm._model.parts : null;');
  parts.push('  var drawableMeta = [];');
  parts.push('  if(d){');
  parts.push('    for(var mi=0;mi<d.count;mi++){');
  parts.push('      var partIdx = d.parentPartIndices ? d.parentPartIndices[mi] : -1;');
  parts.push('      var partId = (p && partIdx >= 0 && partIdx < p.ids.length) ? p.ids[partIdx] : "";');
  parts.push('      var entry={id:d.ids[mi],partId:partId,visible:!!(d.dynamicFlags[mi]&1),opacity:d.opacities[mi],drawOrder:d.drawOrders[mi],renderOrder:d.renderOrders[mi],textureIndex:d.textureIndices[mi]};');
  parts.push('      if(entry.visible&&entry.opacity>0){');
  parts.push('        entry.positions=Array.from(d.vertexPositions[mi]);');
  parts.push('        entry.uvs=Array.from(d.vertexUvs[mi]);');
  parts.push('        entry.indices=Array.from(d.indices[mi]);');
  parts.push('      }');
  parts.push('      drawableMeta.push(entry);');
  parts.push('    }');
  parts.push('  }');

  // Segmentation pass: render each drawable with flat region color using WebGL
  parts.push('  var segMap = ' + segMapJson + ';');
  parts.push('  if(segMap && d){');
  // Create an offscreen canvas for the seg pass
  parts.push('    var segCanvas = document.createElement("canvas");');
  parts.push('    segCanvas.width=' + RENDER_SIZE + '; segCanvas.height=' + RENDER_SIZE + ';');
  parts.push('    var gl = segCanvas.getContext("webgl",{premultipliedAlpha:false,alpha:true,antialias:false});');
  parts.push('    if(gl){');
  parts.push('      gl.viewport(0,0,' + RENDER_SIZE + ',' + RENDER_SIZE + ');');
  parts.push('      gl.clearColor(0,0,0,0); gl.clear(gl.COLOR_BUFFER_BIT);');
  // No blending for seg pass — each opaque fragment overwrites with its region ID.
  // Drawables are rendered back-to-front so the topmost visible drawable's region wins.
  parts.push('      gl.disable(gl.BLEND);');

  // Get textures from pixi's resource manager
  parts.push('      var pixiTextures = model.textures;');

  // Compile seg program
  parts.push('      var VERT_SEG=' + JSON.stringify(VERT_SEG_SRC) + ';');
  parts.push('      var FRAG_SEG=' + JSON.stringify(FRAG_SEG_SRC) + ';');
  parts.push('      var prog = compileProgram(gl, VERT_SEG, FRAG_SEG);');
  parts.push('      gl.useProgram(prog);');
  parts.push('      var aPos=gl.getAttribLocation(prog,"aPos");');
  parts.push('      var aUV=gl.getAttribLocation(prog,"aUV");');
  parts.push('      var uPPU=gl.getUniformLocation(prog,"uPPU");');
  parts.push('      var uOriginX=gl.getUniformLocation(prog,"uOriginX");');
  parts.push('      var uOriginY=gl.getUniformLocation(prog,"uOriginY");');
  parts.push('      var uModelScale=gl.getUniformLocation(prog,"uModelScale");');
  parts.push('      var uModelOffsetX=gl.getUniformLocation(prog,"uModelOffsetX");');
  parts.push('      var uModelOffsetY=gl.getUniformLocation(prog,"uModelOffsetY");');
  parts.push('      var uHalfSize=gl.getUniformLocation(prog,"uHalfSize");');
  parts.push('      var uTex=gl.getUniformLocation(prog,"uTex");');
  parts.push('      var uRegion=gl.getUniformLocation(prog,"uRegionId");');

  // Read canvas info (PixelsPerUnit, CanvasOrigin) from Cubism Core model.
  // Use pixi model placement (scale, x, y) to map to RENDER_SIZE canvas coords.
  parts.push('      var ci=cm&&cm._model?cm._model.canvasinfo:null;');
  parts.push('      var ppu=ci&&ci.PixelsPerUnit?ci.PixelsPerUnit:1;');
  parts.push('      var originX=ci&&ci.CanvasOriginX!==undefined?ci.CanvasOriginX:0;');
  parts.push('      var originY=ci&&ci.CanvasOriginY!==undefined?ci.CanvasOriginY:0;');
  parts.push('      var msc=model.scale.x, mox=model.x, moy=model.y;');
  parts.push('      gl.uniform1f(uPPU, ppu);');
  parts.push('      gl.uniform1f(uOriginX, originX);');
  parts.push('      gl.uniform1f(uOriginY, originY);');
  parts.push('      gl.uniform1f(uModelScale, msc);');
  parts.push('      gl.uniform1f(uModelOffsetX, mox);');
  parts.push('      gl.uniform1f(uModelOffsetY, moy);');
  parts.push('      gl.uniform1f(uHalfSize, ' + RENDER_SIZE + '/2.0);');
  parts.push('      gl.uniform1i(uTex,0);');

  // Sort drawables by drawOrders
  parts.push('      var order=[];');
  parts.push('      for(var i=0;i<d.count;i++) order.push(i);');
  parts.push('      order.sort(function(a,b){return d.drawOrders[a]-d.drawOrders[b];});');
  parts.push('      var posBuf=gl.createBuffer(),uvBuf=gl.createBuffer(),idxBuf=gl.createBuffer();');

  // Upload textures from pixi's textures
  parts.push('      var glTextures=[];');
  parts.push('      for(var ti=0;ti<pixiTextures.length;ti++){');
  parts.push('        var src=pixiTextures[ti];');
  parts.push('        var imgSrc=src.baseTexture?src.baseTexture.resource:null;');
  parts.push('        var imgEl=imgSrc?(imgSrc.source||imgSrc.data||imgSrc):null;');
  parts.push('        var tex=gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D,tex);');
  parts.push('        if(imgEl&&imgEl.width>0){');
  parts.push('          gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,imgEl);');
  parts.push('        } else {');
  // Fallback: 1×1 white pixel
  parts.push('          gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,1,1,0,gl.RGBA,gl.UNSIGNED_BYTE,new Uint8Array([255,255,255,255]));');
  parts.push('        }');
  parts.push('        gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);');
  parts.push('        gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);');
  parts.push('        gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);');
  parts.push('        gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);');
  parts.push('        glTextures.push(tex);');
  parts.push('      }');

  // Render each drawable with its region color
  parts.push('      var dbgDrawCount=0;');
  parts.push('      for(var oi=0;oi<order.length;oi++){');
  parts.push('        var idx=order[oi];');
  parts.push('        var meshId=d.ids[idx];');
  parts.push('        var inSegMap=(segMap&&(meshId in segMap));');
  parts.push('        if(!(d.dynamicFlags[idx]&1)||d.opacities[idx]<=0) continue;');
  parts.push('        var texIdx=d.textureIndices[idx];');
  parts.push('        if(texIdx>=glTextures.length) continue;');
  parts.push('        var rid=(segMap&&inSegMap)?segMap[meshId]:0;');
  // Skip background (rid=0) drawables — rendering them with black would overwrite body regions
  parts.push('        if(rid===0) continue;');
  parts.push('        var positions=d.vertexPositions[idx],uvs=d.vertexUvs[idx],indices=d.indices[idx];');
  parts.push('        gl.uniform1f(uRegion,rid);');
  parts.push('        gl.bindBuffer(gl.ARRAY_BUFFER,posBuf); gl.bufferData(gl.ARRAY_BUFFER,positions,gl.DYNAMIC_DRAW);');
  parts.push('        gl.enableVertexAttribArray(aPos); gl.vertexAttribPointer(aPos,2,gl.FLOAT,false,0,0);');
  parts.push('        gl.bindBuffer(gl.ARRAY_BUFFER,uvBuf); gl.bufferData(gl.ARRAY_BUFFER,uvs,gl.DYNAMIC_DRAW);');
  parts.push('        gl.enableVertexAttribArray(aUV); gl.vertexAttribPointer(aUV,2,gl.FLOAT,false,0,0);');
  parts.push('        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,idxBuf); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,indices,gl.DYNAMIC_DRAW);');
  parts.push('        gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D,glTextures[texIdx]);');
  parts.push('        gl.drawElements(gl.TRIANGLES,indices.length,gl.UNSIGNED_SHORT,0);');
  parts.push('        dbgDrawCount++;');
  parts.push('      }');
  // Read back a small sample to count nonzero pixels for the SEG_DEBUG summary line
  parts.push('      var dbgPx=new Uint8Array(' + RENDER_SIZE + '*' + RENDER_SIZE + '*4);');
  parts.push('      gl.readPixels(0,0,' + RENDER_SIZE + ',' + RENDER_SIZE + ',gl.RGBA,gl.UNSIGNED_BYTE,dbgPx);');
  parts.push('      var dbgNonzero=0;');
  parts.push('      for(var pi=0;pi<' + RENDER_SIZE + '*' + RENDER_SIZE + ';pi++){if(dbgPx[pi*4+3]>0)dbgNonzero++;}');
  parts.push('      window.__segDebug={glNonzero:dbgNonzero,drew:dbgDrawCount};');
  // Read back seg canvas. Detect the seg's own pixel bbox (different coord system
  // than pixi composite), then crop to the same output SIZE.
  // Use nearest-neighbor (smoothing=false) to preserve exact region ID pixel values.
  parts.push('      var midSeg=document.createElement("canvas");');
  parts.push('      midSeg.width=' + RENDER_SIZE + '; midSeg.height=' + RENDER_SIZE + ';');
  parts.push('      var midSegCtx=midSeg.getContext("2d"); midSegCtx.imageSmoothingEnabled=false; midSegCtx.drawImage(segCanvas,0,0);');
  parts.push('      var segBbox=pixelBbox(midSegCtx,' + RENDER_SIZE + ',' + RENDER_SIZE + ',0);');
  parts.push('      if(!segBbox) segBbox={x:0,y:0,w:' + RENDER_SIZE + ',h:' + RENDER_SIZE + '};');
  parts.push('      var segPad=Math.round(Math.max(segBbox.w,segBbox.h)*0.02);');
  parts.push('      var croppedSeg=cropAndScale(midSeg,segBbox,segPad,' + SIZE + ',false);');
  parts.push('      window.__segResult=croppedSeg.canvas.toDataURL("image/png");');
  parts.push('    }');
  parts.push('  }');

  // Export metadata
  // renderTransform maps model-space (post-deform, Y-up) → output pixel coords
  // The seg pass above uses: (pos - center) / (sz/2) → NDC → maps to RENDER_SIZE
  // Then same crop as composite
  // To reconstruct screen coords in Python from model-space coords:
  //   x_seg_canvas = ((mx - cX) / sz + 0.5) * RENDER_SIZE
  //   y_seg_canvas = (-(my - cY) / sz + 0.5) * RENDER_SIZE   (Y flipped)
  //   px_out = (x_seg_canvas - cropX) * cropScale + dx
  //   py_out = (y_seg_canvas - cropY) * cropScale + dy
  // For the composite (pixi) coordinate:
  //   x_pixi = mx * modelScale + modelX
  //   y_pixi = -my * modelScale + modelY  (Y flipped)
  //   px_out = (x_pixi - cropX) * cropScale + dx
  //   py_out = (y_pixi - cropY) * cropScale + dy
  parts.push('  window.__meta=JSON.stringify({drawableMeta:drawableMeta,renderTransform:cropInfo});');
  parts.push('  window.__done=true;');
  parts.push('}catch(e){window.__error=e.message+"\\n"+e.stack;}');
  parts.push('})();');
  parts.push('</script></body></html>');

  return parts.join('\n');
}

(async () => {
  let browser;
  try {
    browser = await puppeteer.launch({
      headless: true,
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--enable-webgl',
        '--use-gl=angle',
      ],
    });

    const page = await browser.newPage();
    await page.setViewport({ width: SIZE, height: SIZE });

    // Intercept requests and serve files from our resource map
    await page.setRequestInterception(true);
    const corsHeaders = { 'Access-Control-Allow-Origin': '*' };
    page.on('request', req => {
      const url = req.url();
      if (resMap[url]) {
        const { data, mime } = resMap[url];
        req.respond({ status: 200, contentType: mime, body: data, headers: corsHeaders });
      } else if (url.startsWith(BASE_URL)) {
        // Fallback: serve any file from model directory by relative path
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

    // Build and register the HTML page
    const segMapJson = JSON.stringify(segMap);
    const htmlContent = buildPageHtml(MODEL_URL, segMapJson);
    const HTML_URL = BASE_URL + 'index.html';
    resMap[HTML_URL] = { data: Buffer.from(htmlContent), mime: 'text/html' };
    resMap[BASE_URL + 'cubismcore.js'] = { data: Buffer.from(cubismCoreCode), mime: 'application/javascript' };
    resMap[BASE_URL + 'pixi.js'] = { data: Buffer.from(pixiCode), mime: 'application/javascript' };
    resMap[BASE_URL + 'cubism4.js'] = { data: Buffer.from(cubism4Code), mime: 'application/javascript' };

    await page.goto(HTML_URL, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction('window.__done || window.__error', { timeout: 60000 });

    const error = await page.evaluate(() => window.__error);
    if (error) {
      console.error('Render error:', error);
      process.exit(1);
    }

    // Save composite image
    const dataUrl = await page.evaluate(() => window.__result);
    if (!dataUrl) {
      console.error('No render result');
      process.exit(1);
    }
    const imgBuf = Buffer.from(dataUrl.split(',')[1], 'base64');
    fs.writeFileSync(outputPng, imgBuf);
    console.log('Saved: ' + outputPng);

    // Save segmentation PNG if produced
    const segDataUrl = await page.evaluate(() => window.__segResult);
    const segDebug = await page.evaluate(() => window.__segDebug);
    if (segDebug) {
      console.log('SEG_DEBUG: WebGL nonzero=' + segDebug.glNonzero + ' drew=' + segDebug.drew);
    }
    if (segDataUrl) {
      const segPath = outputPng.replace(/\.png$/i, '.seg.png');
      const segBuf = Buffer.from(segDataUrl.split(',')[1], 'base64');
      fs.writeFileSync(segPath, segBuf);
      console.log('Saved seg: ' + segPath);
    }

    // Save drawable metadata JSON
    const meta = await page.evaluate(() => window.__meta);
    if (meta) {
      const metaPath = outputPng.replace(/\.png$/i, '.meta.json');
      fs.writeFileSync(metaPath, meta);
    }

  } catch (e) {
    console.error('Fatal error:', e.message);
    process.exit(1);
  } finally {
    if (browser) await browser.close();
  }
})();
