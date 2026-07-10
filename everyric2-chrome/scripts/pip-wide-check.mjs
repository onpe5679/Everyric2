// PiP 창을 가로로 넓게(낮게) 리사이즈했을 때 영상 미러가 가사 스테이지를
// 짓누르지 않는지 검증하는 단독 체크. 실서버(:8000)에 해당 곡 싱크가 있어야 한다.
// 실행: node scripts/pip-wide-check.mjs [videoUrl]
import { chromium } from 'playwright';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import { mkdtempSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const distDir = resolve(__dirname, '../dist');
const videoUrl = process.argv[2] ?? 'https://www.youtube.com/watch?v=Xg-qfsKN2_E';
const userDataDir = mkdtempSync(join(tmpdir(), 'everyric-wide-'));

const ctx = await chromium.launchPersistentContext(userDataDir, {
  channel: process.env.EVERYRIC_E2E_CHANNEL ?? 'msedge',
  headless: false,
  viewport: { width: 1440, height: 900 },
  args: [
    `--disable-extensions-except=${distDir}`,
    `--load-extension=${distDir}`,
    '--mute-audio',
    '--autoplay-policy=no-user-gesture-required',
    '--window-position=40,40',
  ],
});

try {
  await (ctx.serviceWorkers()[0] ?? await ctx.waitForEvent('serviceworker', { timeout: 15000 }));
  const page = ctx.pages()[0] ?? await ctx.newPage();
  await page.goto(videoUrl, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForSelector('#everyric-root', { state: 'attached', timeout: 30000 });

  // 서버 싱크가 즉시 로드되어야 한다 (DB에 있음)
  await page.waitForFunction(() => {
    const root = document.getElementById('everyric-root')?.shadowRoot;
    return (root?.querySelectorAll('.ey-line:not(.ey-line-plain)').length ?? 0) > 0;
  }, null, { timeout: 45000, polling: 1000 });

  // 영상이 실제로 재생 중이도록 보장 (rs>=2 + playing) — 미러 캡처 전제
  await page.evaluate(() => {
    const v = document.querySelector('video.html5-main-video') ?? document.querySelector('video');
    if (v) { v.currentTime = 30; void v.play().catch(() => {}); }
  });
  await page.waitForFunction(() => {
    const v = document.querySelector('video.html5-main-video') ?? document.querySelector('video');
    return v && v.readyState >= 2 && !v.paused;
  }, null, { timeout: 30000, polling: 500 });

  await page.locator('[title="PiP 창으로 보기"]').click();
  await page.waitForTimeout(3000);

  const pipPage = ctx.pages().find(p => p !== page);
  if (!pipPage) throw new Error('PiP window not exposed as page');

  const measure = () => pipPage.evaluate(() => {
    const q = s => document.querySelector(s);
    const hOf = s => q(s)?.clientHeight ?? 0;
    const visible = s => {
      const el = q(s);
      return el ? getComputedStyle(el).display !== 'none' : false;
    };
    return {
      winH: window.innerHeight,
      videoVisible: visible('.ey-pip-video'),
      videoH: hOf('.ey-pip-video'),
      stageH: hOf('.ey-pip-stage'),
      pitchH: hOf('.ey-pip-pitch'),
      curFontPx: q('.ey-pip-line.current') ? parseFloat(getComputedStyle(q('.ey-pip-line.current')).fontSize) : 0,
    };
  });

  const before = await measure();
  console.log('default size =', JSON.stringify(before));
  if (!before.videoVisible) console.log('WARN: 영상 미러가 안 붙음 — 이 체크의 핵심 전제가 빠짐');

  await pipPage.setViewportSize({ width: 1100, height: 330 });
  await pipPage.waitForTimeout(800);
  const wide = await measure();
  console.log('wide 1100x330 =', JSON.stringify(wide));

  // 가라오케 레인이 켜지면 스테이지는 의도적으로 숨겨진다(중복 표시 제거) — 레인 높이로 판정
  const lyricsH = wide.stageH > 0 ? wide.stageH : wide.pitchH;
  const ok = lyricsH >= 90 && (!wide.videoVisible || wide.videoH >= 40);
  console.log(`${ok ? 'PASS' : 'FAIL'}: 넓은 창에서 가사영역 ${lyricsH}px (stage=${wide.stageH}/lane=${wide.pitchH}) / 영상 ${wide.videoH}px 공존`);
  await pipPage.screenshot({ path: resolve(__dirname, '../pip-wide-video.png') });
  console.log('screenshot: pip-wide-video.png');
  process.exitCode = ok ? 0 : 1;
} catch (e) {
  console.log('WIDE CHECK: ERROR —', String(e).slice(0, 300));
  process.exitCode = 1;
} finally {
  await ctx.close();
}
