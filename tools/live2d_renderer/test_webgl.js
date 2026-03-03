const puppeteer = require('puppeteer');

const html = `<!DOCTYPE html><html><body>
<script>
(async function() {
  // Create WebGL canvas and draw a specific color
  var wc = document.createElement('canvas');
  wc.width = 10; wc.height = 10;
  var gl = wc.getContext('webgl', {premultipliedAlpha:false, alpha:true, antialias:false});
  gl.viewport(0,0,10,10);
  gl.clearColor(0,0,0,0);
  gl.clear(gl.COLOR_BUFFER_BIT);
  
  // Simple shader to draw region 1 (v = 1*10/255)
  var vs = gl.createShader(gl.VERTEX_SHADER);
  gl.shaderSource(vs, 'attribute vec2 aPos; void main(){gl_Position=vec4(aPos,0,1);}');
  gl.compileShader(vs);
  var fs = gl.createShader(gl.FRAGMENT_SHADER);
  gl.shaderSource(fs, 'precision mediump float; void main(){float v=1.0*10.0/255.0; gl_FragColor=vec4(v,v,v,1.0);}');
  gl.compileShader(fs);
  var prog = gl.createProgram();
  gl.attachShader(prog,vs); gl.attachShader(prog,fs);
  gl.linkProgram(prog);
  gl.useProgram(prog);
  
  gl.disable(gl.BLEND);
  var buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);
  var aPos = gl.getAttribLocation(prog,'aPos');
  gl.enableVertexAttribArray(aPos);
  gl.vertexAttribPointer(aPos,2,gl.FLOAT,false,0,0);
  gl.drawArrays(gl.TRIANGLE_STRIP,0,4);
  
  // Read back directly from WebGL
  var webglPixels = new Uint8Array(4);
  gl.readPixels(0,0,1,1,gl.RGBA,gl.UNSIGNED_BYTE,webglPixels);
  console.log('WebGL readPixels (0,0):', Array.from(webglPixels));
  
  // Transfer to 2D canvas
  var c2 = document.createElement('canvas');
  c2.width=10; c2.height=10;
  var ctx2 = c2.getContext('2d');
  ctx2.imageSmoothingEnabled = false;
  ctx2.drawImage(wc, 0, 0);
  
  // Read from 2D canvas
  var d2 = ctx2.getImageData(0,0,1,1).data;
  console.log('2D canvas pixel (0,0):', Array.from(d2));
  
  // Export to PNG
  window.__result = c2.toDataURL('image/png');
  window.__done = true;
})();
</script></body></html>`;

(async () => {
  const browser = await puppeteer.launch({headless:true, args:['--no-sandbox','--disable-setuid-sandbox','--enable-webgl','--use-gl=angle']});
  const page = await browser.newPage();
  await page.setContent(html);
  await page.waitForFunction('window.__done');
  const [webglPx, c2Px] = await page.evaluate(() => [window.__webglPx, window.__c2Px]);
  const logs = await page.evaluate(() => window.__logs);
  const dataUrl = await page.evaluate(() => window.__result);
  const png = Buffer.from(dataUrl.split(',')[1], 'base64');
  require('fs').writeFileSync('/tmp/webgl_test.png', png);
  await browser.close();
})();
