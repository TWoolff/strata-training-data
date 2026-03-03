const puppeteer = require('puppeteer');
const html = `<!DOCTYPE html><html><body>
<canvas id='c' width='10' height='10'></canvas>
<script>
var c = document.getElementById('c');
var ctx = c.getContext('2d');
ctx.fillStyle = 'rgba(1, 0, 0, 255)';
ctx.fillRect(0, 0, 1, 1);
ctx.fillStyle = 'rgba(10, 0, 0, 255)';
ctx.fillRect(1, 0, 1, 1);
ctx.fillStyle = 'rgba(40, 0, 0, 255)';
ctx.fillRect(2, 0, 1, 1);
window.__result = c.toDataURL('image/png');
window.__done = true;
</script></body></html>`;

(async () => {
  const browser = await puppeteer.launch({headless:true, args:['--no-sandbox','--disable-setuid-sandbox']});
  const page = await browser.newPage();
  await page.setContent(html);
  await page.waitForFunction('window.__done');
  const dataUrl = await page.evaluate(() => window.__result);
  const png = Buffer.from(dataUrl.split(',')[1], 'base64');
  process.stdout.write(png);
  await browser.close();
})();
