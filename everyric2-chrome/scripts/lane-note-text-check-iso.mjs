// lane-note-text-check.mjs와 **같은 검사**를, 공유 dist/와 공유 프로필을 건드리지 않고
// 돌리는 격리판. 검사 내용(노트 위 음절 텍스트가 pitchPronPosition의 모든 값에서 그려지는가)
// 은 그대로다 — 바뀐 것은 «어디서 로드하는가»뿐이다.
//
// 왜 따로 두나: 병렬 검수가 도는 동안 원본 하네스는 공유 dist/를 스냅샷하므로, 내 변경을
// 반영하려면 공유 dist/를 다시 빌드해야 한다. 그러면 남의 실행을 덮는다. 격리 outDir로
// 빌드한 산출물과 고정 전용 프로필을 인자로 받아 그 충돌을 없앤다.
//
// 실행:
//   npx vite build --outDir dist-lane2-check --emptyOutDir
//   node scripts/lane-note-text-check-iso.mjs <videoId> --dist dist-lane2-check --profile <dir>
import { chromium } from 'playwright';
import { fileURLToPath } from 'url';
import { dirname, resolve, join } from 'path';
import { cpSync, mkdirSync, mkdtempSync, readFileSync } from 'fs';
import { tmpdir } from 'os';
import { ensureLocalServerPermissionForServerUrl } from './lib/local-server-permission.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const arg = (n, d) => { const i = process.argv.indexOf(`--${n}`); return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : d; };
const videoId = process.argv[2];
const SERVER_URL = arg('server', 'http://127.0.0.1:8000');
const IS_LOCAL = /127\.0\.0\.1|localhost/.test(SERVER_URL);
if (!videoId) {
  console.log('사용법: node scripts/lane-note-text-check-iso.mjs <videoId> [--dist dir] [--profile dir]');
  process.exit(2);
}

// dist는 스냅샷해서 쓴다(원본이 도중에 다시 빌드돼도 이 실행은 흔들리지 않는다)
const distDir = mkdtempSync(join(tmpdir(), 'everyric-dist-iso-'));
cpSync(resolve(__dirname, '..', arg('dist', 'dist')), distDir, { recursive: true });
JSON.parse(readFileSync(join(distDir, 'manifest.json'), 'utf8'));
const userDataDir = arg('profile', join(tmpdir(), 'exec-lane2-note-profile'));
mkdirSync(userDataDir, { recursive: true });

let failed = false;
const check = (ok, label, detail) => {
  console.log(`${ok ? 'PASS' : 'FAIL'}: ${label}${detail !== undefined ? ' = ' + JSON.stringify(detail) : ''}`);
  if (!ok) failed = true;
};

/** 레인 캔버스에서 «노트 영역»(위 55%)의 잉크 픽셀 수 — 음절 글자가 있으면 확 는다 */
const MEASURE = `(() => {
  const c = document.querySelector('.ey-main-lane')
    ?? document.getElementById('everyric-root')?.shadowRoot?.querySelector('.ey-main-lane');
  if (!c || !c.width) return null;
  const g = c.getContext('2d');
  const h = Math.floor(c.height * 0.55);
  const d = g.getImageData(0, 0, c.width, h).data;
  let ink = 0;
  for (let i = 3; i < d.length; i += 4) if (d[i] > 40) ink++;
  return { w: c.width, h: c.height, noteInk: ink };
})()`;

const ctx = await chromium.launchPersistentContext(userDataDir, {
  ignoreDefaultArgs: ['--disable-extensions'],
  headless: false,
  viewport: { width: 1440, height: 900 },
  args: [
    `--disable-extensions-except=${distDir}`, `--load-extension=${distDir}`,
    '--mute-audio', '--autoplay-policy=no-user-gesture-required', '--window-position=30,30',
  ],
});

try {
  const sw = ctx.serviceWorkers()[0] ?? await ctx.waitForEvent('serviceworker', { timeout: 15000 });
  if (IS_LOCAL) {
    await ensureLocalServerPermissionForServerUrl(ctx, sw, new URL(sw.url()).host, SERVER_URL);
  }
  const setSettings = (patch) => sw.evaluate(async (p) => {
    const cur = (await chrome.storage.local.get('settings')).settings ?? {};
    await chrome.storage.local.set({ settings: { ...cur, ...p } });
  }, patch);

  await setSettings({
    serverUrl: SERVER_URL, uiLanguage: 'ko', theme: 'dark',
    modMainLane: true, mainLanePos: 'bottom', pitchPronPosition: 'off',
  });

  const page = ctx.pages()[0] ?? await ctx.newPage();
  await page.goto(`https://www.youtube.com/watch?v=${videoId}`, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForSelector('#everyric-root', { state: 'attached', timeout: 30000 });
  await page.waitForFunction(() => {
    const r = document.getElementById('everyric-root')?.shadowRoot;
    return (r?.querySelectorAll('.ey-line').length ?? 0) > 0;
  }, null, { timeout: 45000, polling: 1000 });
  await page.evaluate(() => {
    const v = document.querySelector('video.html5-main-video') ?? document.querySelector('video');
    if (v) { v.currentTime = 30; void v.play().catch(() => {}); }
  });
  await page.waitForTimeout(2500);

  const source = await page.evaluate(() => {
    const root = document.getElementById('everyric-root')?.shadowRoot;
    const lines = [...(root?.querySelectorAll('.ey-line') ?? [])].slice(0, 6);
    return lines.map(el => ({
      text: (el.querySelector('.ey-line-text')?.textContent ?? el.textContent ?? '').slice(0, 40),
      pron: (el.querySelector('.ey-line-pron')?.textContent ?? '').slice(0, 40),
    }));
  });
  console.log('INFO: 화면 줄 샘플 =', JSON.stringify(source.slice(0, 3)));

  // 커밋본과 같은 이유로 재기 전에 영상을 멈춘다(설정마다 다른 순간을 보면 잉크가
  // 통째로 달라진다 — 근거는 lane-note-text-check.mjs 주석).
  await page.evaluate(() => {
    const v = document.querySelector('video.html5-main-video') ?? document.querySelector('video');
    v?.pause();
  });
  await page.waitForTimeout(800);

  const results = {};
  for (const pos of ['off', 'bottom', 'center', 'both']) {
    await setSettings({ pitchPronPosition: pos });
    await page.waitForTimeout(1500);
    const m = await page.evaluate(MEASURE);
    results[pos] = m;
    check(m !== null && m.noteInk > 500,
      `pitchPronPosition='${pos}' 에서도 노트 영역에 글자가 그려짐`, m);
  }

  // 판정 기준은 커밋본(lane-note-text-check.mjs)과 **같은 것**을 쓴다 — 근거·문턱의
  // 실측 출처는 그쪽 주석에 있다(요약: 'both'는 발음 줄이 세로 공간을 먹어 가사 줄이
  // 측정 밴드로 밀려 올라오므로 비교 상대가 못 된다. 'center'는 레이아웃이 off와 같다).
  const a = results.off?.noteInk ?? 0;
  const c = results.center?.noteInk ?? 0;
  const ratio = c > 0 ? a / c : 0;
  check(ratio >= 0.72,
    "노트 영역 잉크가 설정값에 좌우되지 않음('off'가 'center' 대비 안 줄어듦)",
    { off: a, center: c, ratio: +ratio.toFixed(3) });

  await page.screenshot({ path: resolve(__dirname, '../lane-note-text-check-iso.png') });
  console.log('screenshot: lane-note-text-check-iso.png');
  console.log(failed ? 'LANE NOTE TEXT CHECK(ISO): FAIL' : 'LANE NOTE TEXT CHECK(ISO): PASS');
  process.exitCode = failed ? 1 : 0;
} catch (e) {
  console.log('LANE NOTE TEXT CHECK(ISO): ERROR —', String(e).slice(0, 400));
  process.exitCode = 1;
} finally {
  await ctx.close();
}
