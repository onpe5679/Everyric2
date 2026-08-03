// 열 제어 «고아화» 불변식 검증.
//
// 불변식(운영자 지시 2026-08-04): **어떤 열의 제어 수단이 다른 접힘 가능한 열 안에만
// 존재하면 안 된다.** 무엇이 접혀 있든, 표시 중인 모든 열은 자기 자신을 닫을 수단이
// PiP UI에 항상 남아 있어야 한다.
//
// 실제 버그: 레인을 켠 채 가사창 열을 접으면 레인 토글이 그 안(퀵 줄)에만 있어서
// 레인을 끌 방법이 사라졌다. 가사창을 다시 펴야만 끌 수 있는 «고아» 상태.
//
// 재는 법: 각 열을 하나씩 접어 가며, 남은 각 열의 토글 버튼이 **실제로 보이는지**
// (getBoundingClientRect 폭 > 0) 확인한다. 코너는 열 행 바깥의 툴바 줄이라 열 접힘과 무관하지만,
// 그 사실 자체를 검증에 박아 두어야 나중에 누가 코너를 열 안으로 옮기면 즉시 걸린다.
//
// 실행: node scripts/pip-orphan-check.mjs <syncedVideoId>
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
  console.log('사용법: node scripts/pip-orphan-check.mjs <syncedVideoId>');
  process.exit(2);
}

let failed = false;
const check = (ok, label, detail) => {
  console.log(`${ok ? 'PASS' : 'FAIL'}: ${label}${detail !== undefined ? ' = ' + JSON.stringify(detail) : ''}`);
  if (!ok) failed = true;
};

/** 열의 표시 여부 + 그 열을 끌 수 있는 «보이는» 버튼이 있는지 */
const MEASURE = `(() => {
  const vis = (el) => {
    if (!el) return false;
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden') return false;
    const b = el.getBoundingClientRect();
    return b.width > 0 && b.height > 0;
  };
  const corner = document.querySelector('.ey-pip-corner');
  const btns = corner ? [...corner.querySelectorAll('button')] : [];
  const byTitle = (frag) => btns.find(b => (b.title || '').includes(frag)) ?? null;
  return {
    cols: {
      lane: vis(document.querySelector('.ey-pip-lane-col')),
      center: vis(document.querySelector('.ey-pip-center')),
      panel: vis(document.getElementById('everyric-root')),
      playlist: vis(document.querySelector('.ey-pip-playlist-col')),
    },
    cornerVisible: vis(corner),
    cornerCount: btns.length,
    // 각 열을 끌 수 있는 «보이는» 컨트롤 (코너 버튼, 툴팁으로 식별)
    ctrl: {
      lane: vis(byTitle('레인')),
      panel: vis(byTitle('가사창')),
      playlist: vis(byTitle('재생목록')),
      short: vis(byTitle('가사 한 줄')),
    },
    titles: btns.map(b => b.title),
  };
})()`;

const userDataDir = mkdtempSync(join(tmpdir(), 'everyric-orphan-'));
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

  await setSettings({
    serverUrl: LOCAL_SERVER_URL, uiLanguage: 'ko', theme: 'dark',
    pitchGuide: true, pipPlaylist: true, pipShortLyrics: true, pipShowPanel: true,
    pipShowVideo: true, pipLaneWidth: 260, pipPanelWidth: 340, pipLaneSwapped: false,
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
  await page.waitForTimeout(1200);
  await page.locator('#everyric-root [title="PiP 창으로 보기"]').first().click();
  await page.waitForTimeout(3500);

  const pip = ctx.pages().find(p => p !== page);
  if (!pip) throw new Error('PiP 창이 페이지로 노출되지 않음');
  await pip.setViewportSize({ width: 1400, height: 760 }); // 네 열이 모두 들어가는 폭
  await pip.waitForTimeout(800);

  // 접기 조합 — «가사창을 접었을 때»가 실제 버그가 났던 자리다
  const CASES = [
    ['전부 펼침', {}],
    ['가사창 접음(실제 버그 상황)', { pipShowPanel: false }],
    ['가사창 + 재생목록 접음', { pipShowPanel: false, pipPlaylist: false }],
    ['가사창 + 단축 표시 접음', { pipShowPanel: false, pipShortLyrics: false }],
    ['레인만 남김', { pipShowPanel: false, pipPlaylist: false, pipShortLyrics: false }],
    ['재생목록만 접음', { pipShowPanel: true, pipPlaylist: false, pipShortLyrics: true }],
  ];

  for (const [name, patch] of CASES) {
    await setSettings({
      pitchGuide: true, pipPlaylist: true, pipShortLyrics: true, pipShowPanel: true, ...patch,
    });
    await page.waitForTimeout(1100);
    const m = await pip.evaluate(MEASURE);
    const shown = Object.entries(m.cols).filter(([, v]) => v).map(([k]) => k);
    console.log(`\n── ${name} — 보이는 열: ${shown.join(' | ')}`);
    check(m.cornerVisible, `${name}: 코너 컨트롤 묶음이 보인다`, { count: m.cornerCount });
    // **불변식**: 표시 중인 각 열은 자기를 닫을 «보이는» 수단이 있어야 한다
    if (m.cols.lane) check(m.ctrl.lane, `${name}: 레인 열을 끌 수단이 보인다`, m.ctrl);
    if (m.cols.panel) check(m.ctrl.panel, `${name}: 가사창 열을 끌 수단이 보인다`, m.ctrl);
    if (m.cols.playlist) check(m.ctrl.playlist, `${name}: 재생목록 열을 끌 수단이 보인다`, m.ctrl);
    // 접힌 열도 **되살릴** 수단이 있어야 한다 (같은 버튼이 양방향이다)
    if (!m.cols.panel) check(m.ctrl.panel, `${name}: 접힌 가사창을 되살릴 수단이 보인다`, m.ctrl);
    if (!m.cols.playlist) check(m.ctrl.playlist, `${name}: 접힌 재생목록을 되살릴 수단이 보인다`, m.ctrl);
    check(m.ctrl.short, `${name}: 단축 표시 토글이 보인다`, m.ctrl);
  }

  // 실제로 눌러서 꺼지는지 — 「보인다」와 「먹는다」는 다르다
  await setSettings({ pitchGuide: true, pipShowPanel: false });
  await page.waitForTimeout(1100);
  const before = await pip.evaluate(MEASURE);
  await pip.locator('.ey-pip-corner button[title*="레인"]').first().click();
  await page.waitForTimeout(1200);
  const after = await pip.evaluate(MEASURE);
  console.log('\n── 실제 클릭 (가사창 접힌 상태에서 레인 끄기)');
  check(before.cols.lane && !after.cols.lane,
    '가사창이 접힌 채로도 코너에서 레인을 실제로 끌 수 있다',
    { before: before.cols.lane, after: after.cols.lane });

  await pip.screenshot({ path: resolve(__dirname, '../pip-orphan-check.png') });
  console.log('screenshot: pip-orphan-check.png');
  console.log(failed ? '\nPIP ORPHAN CHECK: FAIL' : '\nPIP ORPHAN CHECK: PASS');
  process.exitCode = failed ? 1 : 0;
} catch (e) {
  console.log('PIP ORPHAN CHECK: ERROR —', String(e).slice(0, 400));
  process.exitCode = 1;
} finally {
  await ctx.close();
}
