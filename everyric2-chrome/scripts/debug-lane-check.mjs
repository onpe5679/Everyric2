// 디버그 레인 오버레이 검증: debugInfo를 켠 상태로 PiP 레인에
// VAD/간주 스트립·RAW f0 곡선·보정 전 타이밍 고스트·곡 conf 헤더가 그려지는지 스크린샷.
// 사전 조건: 실서버(:8000) + 해당 곡 싱크(debug payload 포함, 독음 정렬 재생성본).
// 실행: node scripts/debug-lane-check.mjs [videoUrl] [seekSec]
import { chromium } from 'playwright';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import { mkdtempSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const distDir = resolve(__dirname, '../dist');
const videoUrl = process.argv[2] ?? 'https://www.youtube.com/watch?v=b2NTglk9tvI';
const seekSec = Number(process.argv[3] ?? 163);
const videoId = new URL(videoUrl).searchParams.get('v');
const userDataDir = mkdtempSync(join(tmpdir(), 'everyric-dbg-'));

const sync = await (await fetch(`http://localhost:8000/api/sync/${videoId}`)).json();
const dbg = sync.debug ?? {};
console.log(
  `align=${dbg.alignment_text} f0=${(dbg.f0_curve?.midi ?? []).length}pts ` +
  `fixed=${sync.timestamps.filter(s => s.debug?.fixes).length}줄`,
);

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
  // 디버그 모드 + 넓은 창(8마디)을 미리 저장해 두고 페이지를 연다
  const sw = ctx.serviceWorkers()[0] ?? await ctx.waitForEvent('serviceworker', { timeout: 15000 });
  await sw.evaluate(s => chrome.storage.local.set({ settings: s }), {
    debugInfo: true, pitchWindowMeasures: 8,
  });

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

  const pipPage = ctx.pages().find(p => p !== page);
  if (!pipPage) {
    console.log('FAIL: PiP 페이지 없음');
    process.exitCode = 1;
  } else {
    await pipPage.setViewportSize({ width: 880, height: 560 });
    for (const [name, t] of [['debug-lane-1.png', seekSec], ['debug-lane-2.png', 20]]) {
      await page.evaluate(sec => {
        const v = document.querySelector('video.html5-main-video') ?? document.querySelector('video');
        if (v) { v.currentTime = sec; void v.play().catch(() => {}); }
      }, t);
      await page.waitForTimeout(1200);
      await pipPage.screenshot({ path: resolve(__dirname, `../${name}`) });
      console.log(`screenshot: ${name} (t=${t}s)`);
    }
  }
} catch (e) {
  console.log('ERROR —', String(e).slice(0, 300));
  process.exitCode = 1;
} finally {
  await ctx.close();
}
