// 두 가지를 잰다:
//   (A) 가라오케 단독 모드 — 사용자가 중앙 열·가사창·재생목록을 명시적으로 접어
//       레인만 남기는 배치가 실제로 되는가. 자동 접힘 우선순위(중앙 열 최후 생존)는
//       «폭이 모자랄 때»의 규칙이지 사용자 선택의 제한이 아니라는 것이 요점이다.
//   (B) 모듈 토글이 다른 표면의 기하를 건드리지 않는가 — PiP에서 모듈을 켜고 끄는 동안
//       **메인 창 패널의 위치·크기와 저장된 geometry가 한 픽셀도 안 변해야** 한다.
//       실제 사고: 토글 브로드캐스트 → 메인 재배치 → ResizeObserver → geometry 덮어쓰기+저장.
//
// 실행: node scripts/pip-solo-geom-check.mjs <syncedVideoId>
import { chromium } from 'playwright';
import { fileURLToPath } from 'url';
import { dirname, resolve, join } from 'path';
import { cpSync, mkdtempSync, readFileSync } from 'fs';
import { tmpdir } from 'os';
import { ensureLocalServerPermissionForServerUrl } from './lib/local-server-permission.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const distDir = mkdtempSync(join(tmpdir(), 'everyric-dist-snap-'));
cpSync(resolve(__dirname, '../dist'), distDir, { recursive: true });
JSON.parse(readFileSync(join(distDir, 'manifest.json'), 'utf8'));

const videoId = process.argv[2];
const LOCAL_SERVER_URL = 'http://127.0.0.1:8000';
if (!videoId) {
  console.log('사용법: node scripts/pip-solo-geom-check.mjs <syncedVideoId>');
  process.exit(2);
}

let failed = false;
const check = (ok, label, detail) => {
  console.log(`${ok ? 'PASS' : 'FAIL'}: ${label}${detail !== undefined ? ' = ' + JSON.stringify(detail) : ''}`);
  if (!ok) failed = true;
};

const PIP_COLS = `(() => {
  const vis = (el) => {
    if (!el) return false;
    const cs = getComputedStyle(el);
    if (cs.display === 'none') return false;
    const b = el.getBoundingClientRect();
    return b.width > 0 && b.height > 0;
  };
  const box = (el) => {
    if (!vis(el)) return null;
    const b = el.getBoundingClientRect();
    return { x: Math.round(b.x), w: Math.round(b.width) };
  };
  return {
    lane: box(document.querySelector('.ey-pip-lane-col')),
    center: box(document.querySelector('.ey-pip-center')),
    panel: box(document.getElementById('everyric-root')),
    playlist: box(document.querySelector('.ey-pip-playlist-col')),
    laneHead: !!document.querySelector('.ey-lane-head'),
    laneHeadBtns: document.querySelectorAll('.ey-lane-head-btn').length,
    cornerBtns: document.querySelectorAll('.ey-pip-corner button').length,
    win: { w: innerWidth, h: innerHeight },
  };
})()`;

/** 메인 페이지 패널의 화면 기하 (인라인 스타일 + 실제 사각형) */
const MAIN_GEOM = `(() => {
  const root = document.getElementById('everyric-root');
  const p = root?.shadowRoot?.querySelector('.ey-panel');
  if (!p) return null;
  const b = p.getBoundingClientRect();
  return {
    rect: { x: Math.round(b.x), y: Math.round(b.y), w: Math.round(b.width), h: Math.round(b.height) },
    style: { left: p.style.left, top: p.style.top, width: p.style.width, height: p.style.height },
  };
})()`;

const userDataDir = mkdtempSync(join(tmpdir(), 'everyric-solo-geom-'));
const ctx = await chromium.launchPersistentContext(userDataDir, {
  ignoreDefaultArgs: ['--disable-extensions'],
  headless: false,
  viewport: { width: 1600, height: 1000 },
  args: [
    `--disable-extensions-except=${distDir}`, `--load-extension=${distDir}`,
    '--mute-audio', '--autoplay-policy=no-user-gesture-required', '--window-position=20,20',
  ],
});

try {
  const sw = ctx.serviceWorkers()[0] ?? await ctx.waitForEvent('serviceworker', { timeout: 15000 });
  await ensureLocalServerPermissionForServerUrl(ctx, sw, new URL(sw.url()).host, LOCAL_SERVER_URL);
  const setSettings = (patch) => sw.evaluate(async (p) => {
    const cur = (await chrome.storage.local.get('settings')).settings ?? {};
    await chrome.storage.local.set({ settings: { ...cur, ...p } });
  }, patch);
  const readSaved = () => sw.evaluate(async () => {
    const st = await chrome.storage.local.get(null);
    const geomKey = Object.keys(st).find(k => k.startsWith('geometry:'));
    return { geometry: geomKey ? st[geomKey] : null };
  });

  await setSettings({
    serverUrl: LOCAL_SERVER_URL, uiLanguage: 'ko', theme: 'dark',
    pitchGuide: true, pipPlaylist: true, pipShortLyrics: true,
    pipShowPanel: true, pipShowCenter: true, pipShowVideo: true,
    modMainLane: true, mainLanePos: 'bottom', modPlaylist: false,
    pipLaneWidth: 300, pipPanelWidth: 360,
  });

  // 기하를 **미리 심어 둔다** — 새 프로필은 저장된 geometry가 없어서 «안 바뀌었다»가
  // 공허하게 참이 된다(실측: before null / after null). 실제 값을 넣어야 덮어쓰기를 잡는다.
  await sw.evaluate(async () => {
    await chrome.storage.local.set({
      'geometry:www.youtube.com': { x: 320, y: 140, width: 420, height: 560, collapsed: false },
    });
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
    if (v) { v.currentTime = 25; void v.play().catch(() => {}); }
  });
  await page.waitForTimeout(1500);
  await page.locator('#everyric-root [title="PiP 창으로 보기"]').first().click();
  await page.waitForTimeout(3500);

  const pip = ctx.pages().find(p => p !== page);
  if (!pip) throw new Error('PiP 창이 페이지로 노출되지 않음');
  await pip.setViewportSize({ width: 1280, height: 700 });
  await pip.waitForTimeout(800);

  // ── (A) 가라오케 단독 모드 ────────────────────────────────────
  console.log('\n── (A) 가라오케 단독 모드');
  const full = await pip.evaluate(PIP_COLS);
  check(!!full.lane && !!full.center && !!full.panel, '전제: 세 열이 보인다',
    { lane: !!full.lane, center: !!full.center, panel: !!full.panel });
  check(full.laneHead && full.laneHeadBtns >= 8,
    '가라오케 컨트롤이 레인 열 머리에 붙어 있다', { btns: full.laneHeadBtns });

  await setSettings({ pipShowCenter: false, pipShowPanel: false, pipPlaylist: false });
  await page.waitForTimeout(1400);
  const solo = await pip.evaluate(PIP_COLS);
  check(!!solo.lane && !solo.center && !solo.panel && !solo.playlist,
    '레인만 남는다(가라오케 단독 모드)',
    { lane: !!solo.lane, center: !!solo.center, panel: !!solo.panel, playlist: !!solo.playlist });
  check(solo.lane && solo.lane.w >= solo.win.w - 4,
    '단독 모드에서 레인이 창 폭을 다 쓴다', { laneW: solo.lane?.w, win: solo.win.w });
  check(solo.cornerBtns >= 7, '단독 모드에서도 코너 컨트롤이 전부 남는다(고아 방지)',
    { corner: solo.cornerBtns });
  await pip.screenshot({ path: resolve(__dirname, '../pip-lane-solo-check.png') });

  // 되살릴 수 있는가 — 단독 모드가 막다른 길이면 안 된다
  await setSettings({ pipShowCenter: true, pipShowPanel: true, pipPlaylist: true });
  await page.waitForTimeout(1400);
  const back = await pip.evaluate(PIP_COLS);
  check(!!back.lane && !!back.center && !!back.panel, '단독 모드에서 되돌아올 수 있다',
    { lane: !!back.lane, center: !!back.center, panel: !!back.panel });

  // ── (B) 모듈 토글 → 메인 기하 불변 ────────────────────────────
  console.log('\n── (B) PiP 모듈 토글 중 메인 창 기하 불변');
  await page.waitForTimeout(600);
  const geomBefore = await page.evaluate(MAIN_GEOM);
  const savedBefore = await readSaved();
  check(geomBefore !== null, '전제: 메인 패널 기하를 읽었다', geomBefore?.rect);

  // PiP 쪽 모듈을 여러 번 토글한다 — 매번 applySettings가 양쪽에 방송된다
  for (const patch of [
    { pipPlaylist: false }, { pitchGuide: false }, { pipShortLyrics: false },
    { pipPlaylist: true }, { pitchGuide: true }, { pipShortLyrics: true },
  ]) {
    await setSettings(patch);
    await page.waitForTimeout(700);
  }
  await page.waitForTimeout(1200); // geometry 저장 디바운스(400ms)보다 넉넉히

  const geomAfter = await page.evaluate(MAIN_GEOM);
  const savedAfter = await readSaved();
  check(JSON.stringify(geomAfter?.rect) === JSON.stringify(geomBefore?.rect),
    'PiP 모듈 토글 6회 뒤에도 메인 패널 화면 기하가 그대로',
    { before: geomBefore?.rect, after: geomAfter?.rect });
  check(JSON.stringify(geomAfter?.style) === JSON.stringify(geomBefore?.style),
    'PiP 모듈 토글 뒤에도 메인 패널 인라인 좌표/크기가 그대로',
    { before: geomBefore?.style, after: geomAfter?.style });
  check(JSON.stringify(savedAfter.geometry) === JSON.stringify(savedBefore.geometry),
    '**저장된** 메인 geometry가 덮어써지지 않았다',
    { before: savedBefore.geometry, after: savedAfter.geometry });

  console.log(failed ? '\nPIP SOLO/GEOM CHECK: FAIL' : '\nPIP SOLO/GEOM CHECK: PASS');
  process.exitCode = failed ? 1 : 0;
} catch (e) {
  console.log('PIP SOLO/GEOM CHECK: ERROR —', String(e).slice(0, 400));
  process.exitCode = 1;
} finally {
  await ctx.close();
}
