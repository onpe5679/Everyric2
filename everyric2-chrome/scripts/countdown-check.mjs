// 카운트다운 검증: 첫 가사 라인 시작 3초 전으로 시킹해 PiP 레인에 4·3·2·1 숫자가
// 그려지는지 스크린샷으로 확인한다 (긴 묵음 뒤 라인 시작 예고 기능).
// 사전 조건: 실서버(:8000) + 해당 곡 싱크(notes 포함). 실행: node scripts/countdown-check.mjs [videoUrl]
import { chromium } from 'playwright';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import { mkdtempSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { ensureLocalServerPermissionForServerUrl } from './lib/local-server-permission.mjs';
import { resolveVideoUrl } from './lib/pick-song.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const distDir = resolve(__dirname, '../dist');
// 기본 영상은 **로컬 DB에 있을 때만** 그대로 쓴다 — Xg-qfsKN2_E는 프로드에만 남아 있어
// (2026-08-04 실측) 인자 없이 돌리면 「가사를 찾지 못했어요」에서 전부 실패했다. 그건 제품
// 결함이 아니라 죽은 기본값이므로, 없을 때만 조건에 맞는 곡으로 갈아끼우고 무엇을 골랐는지 찍는다.
const pickedSong = process.argv[2]
  ? { url: process.argv[2], source: 'argv' }
  : resolveVideoUrl('https://www.youtube.com/watch?v=Xg-qfsKN2_E', { minLines: 20, minFirstStart: 5 });
if (pickedSong.source !== 'argv') {
  console.log(`[곡] ${pickedSong.videoId}${pickedSong.title ? ' — ' + pickedSong.title : ''} (${pickedSong.note})`);
}
const videoUrl = pickedSong.url;
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
  const sw = ctx.serviceWorkers()[0] ?? await ctx.waitForEvent('serviceworker', { timeout: 15000 });
  const extId = new URL(sw.url()).host;
  const localServerUrl = 'http://127.0.0.1:8000';
  // serverUrl 기본값이 프로드로 바뀐 뒤로는(host-permissions.ts) 여기서 로컬을 명시하고
  // optional_host_permissions도 실제 흐름으로 부여해야 이 곡의 로컬 싱크(notes 포함)가 뜬다.
  await ensureLocalServerPermissionForServerUrl(ctx, sw, extId, localServerUrl);
  await sw.evaluate(s => chrome.storage.local.set({ settings: s }), { serverUrl: localServerUrl });

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
