// 카운트다운 검증: 첫 가사 라인 시작 3초 전으로 시킹해 PiP 레인에 4·3·2·1 숫자가
// 그려지는지 스크린샷으로 확인한다 (긴 묵음 뒤 라인 시작 예고 기능).
// 사전 조건: 실서버(:8000) + 해당 곡 싱크(notes 포함). 실행: node scripts/countdown-check.mjs [videoUrl]
import { chromium } from 'playwright';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import { mkdtempSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const distDir = resolve(__dirname, '../dist');
const videoUrl = process.argv[2] ?? 'https://www.youtube.com/watch?v=Xg-qfsKN2_E';
const videoId = new URL(videoUrl).searchParams.get('v');
const userDataDir = mkdtempSync(join(tmpdir(), 'everyric-cd-'));

const sync = await (await fetch(`http://localhost:8000/api/sync/${videoId}`)).json();
const firstStart = sync.timestamps[0].start;
console.log('첫 라인 start =', firstStart, '| tempo =', JSON.stringify(sync.tempo ?? null));

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
  const page = ctx.pages()[0] ?? await ctx.newPage();
  await page.goto(videoUrl, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForSelector('#everyric-root', { state: 'attached', timeout: 30000 });
  await page.waitForFunction(() => {
    const root = document.getElementById('everyric-root')?.shadowRoot;
    return (root?.querySelectorAll('.ey-line:not(.ey-line-plain)').length ?? 0) > 0;
  }, null, { timeout: 45000, polling: 1000 });

  await page.evaluate(() => {
    const v = document.querySelector('video.html5-main-video') ?? document.querySelector('video');
    if (v) { void v.play().catch(() => {}); }
  });
  await page.waitForTimeout(800);
  await page.locator('[title="PiP 창으로 보기"]').click();
  await page.waitForTimeout(2500);

  // 첫 라인 시작 3.2초 전으로 시킹 → 카운트다운 3(또는 4)이 보여야 한다
  await page.evaluate(t => {
    const v = document.querySelector('video.html5-main-video') ?? document.querySelector('video');
    if (v) { v.currentTime = t; void v.play().catch(() => {}); }
  }, firstStart - 3.2);
  await page.waitForTimeout(900);

  const pipPage = ctx.pages().find(p => p !== page);
  if (pipPage) {
    await pipPage.screenshot({ path: resolve(__dirname, '../countdown.png') });
    console.log('screenshot: countdown.png');
  } else {
    console.log('FAIL: PiP 페이지 없음');
    process.exitCode = 1;
  }
} catch (e) {
  console.log('ERROR —', String(e).slice(0, 300));
  process.exitCode = 1;
} finally {
  await ctx.close();
}
