// PiP를 연 채로 «싱크 없는 영상»으로 넘어가는 흐름 검증.
//
// 이 경로가 따로 필요한 이유: PiP는 사용자가 닫기 전까지 살아 있어야 하고(재생목록을
// 돌리다 가사 없는 곡 하나를 만났다고 창이 증발하면 안 된다), 그때 창 안에는 «이전 곡의
// 가사»가 아니라 검색·붙여넣기 화면이 떠야 한다. 2026-08-04 재작업으로 PiP 안 UI가
// 메인 가사창과 같은 인스턴스가 되면서 이 비우기 경로가 통째로 다시 쓰였다
// (content.ts cleanupForPage / paintPanels) — 그 결과를 실브라우저에서 확인한다.
//
// 함께 보는 것: 메인 패널의 가사는 **지워지면 안 된다**. 방송(broadcast)으로 양쪽을
// 한꺼번에 비우면 사용자가 보던 메인 창까지 날아가므로, 비우기는 PiP 인스턴스 전용
// 경로여야 한다(재작업 인수인계 문서의 ★함정 ⑤).
//
// 실행: node scripts/pip-nosync-check.mjs <syncedVideoId> [noSyncVideoId]
// 주소는 반드시 127.0.0.1 — 이 개발 머신은 localhost가 IPv6 폴백 스톨로 요청당 2초.
import { chromium } from 'playwright';
import { fileURLToPath } from 'url';
import { dirname, resolve, join } from 'path';
import { cpSync, mkdtempSync, readFileSync } from 'fs';
import { tmpdir } from 'os';
import { ensureLocalServerPermissionForServerUrl } from './lib/local-server-permission.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const liveDist = resolve(__dirname, '../dist');
const distDir = mkdtempSync(join(tmpdir(), 'everyric-dist-snap-'));
cpSync(liveDist, distDir, { recursive: true });
JSON.parse(readFileSync(join(distDir, 'manifest.json'), 'utf8'));

const syncedId = process.argv[2];
const noSyncId = process.argv[3] ?? 'dQw4w9WgXcQ'; // 싱크가 없을 법한 아무 영상
const LOCAL_SERVER_URL = 'http://127.0.0.1:8000';

if (!syncedId) {
  console.log('사용법: node scripts/pip-nosync-check.mjs <syncedVideoId> [noSyncVideoId]');
  process.exit(2);
}

let failed = false;
const check = (ok, label, detail) => {
  console.log(`${ok ? 'PASS' : 'FAIL'}: ${label}${detail !== undefined ? ' = ' + JSON.stringify(detail) : ''}`);
  if (!ok) failed = true;
  return ok;
};

const userDataDir = mkdtempSync(join(tmpdir(), 'everyric-pip-nosync-'));
const ctx = await chromium.launchPersistentContext(userDataDir, {
  ignoreDefaultArgs: ['--disable-extensions'],
  headless: false,
  viewport: { width: 1440, height: 900 },
  args: [
    `--disable-extensions-except=${distDir}`, `--load-extension=${distDir}`,
    '--mute-audio', '--autoplay-policy=no-user-gesture-required', '--window-position=40,40',
  ],
});

/** PiP 창(별도 최상위 문서) 안의 패널 상태를 읽는다 */
const readPip = () => `(() => {
  const w = window.documentPictureInPicture?.window;
  if (!w) return { open: false };
  const host = w.document.getElementById('everyric-root');
  const root = host?.shadowRoot;
  if (!root) return { open: true, panel: false };
  const body = root.querySelector('.ey-body');
  return {
    open: true, panel: true,
    lineCount: root.querySelectorAll('.ey-line').length,
    title: root.querySelector('.ey-song-title')?.textContent ?? '',
    laneVisible: root.querySelector('.ey-lane-wrap')
      ? w.getComputedStyle(root.querySelector('.ey-lane-wrap')).display !== 'none' : false,
    // 검색·붙여넣기 화면이 떴는가 (showEmpty가 그리는 조각)
    emptyState: !!body?.querySelector('.ey-state'),
    bodyText: (body?.textContent ?? '').replace(/\\s+/g, ' ').trim().slice(0, 120),
    // 「화면 정중앙에 뜨는」 문제 확인용 — 본문 안 상자가 창 한가운데 떠 있지 않은가
    stateRect: (() => {
      const el = body?.querySelector('.ey-state');
      if (!el) return null;
      const b = el.getBoundingClientRect();
      return { x: Math.round(b.x), y: Math.round(b.y), w: Math.round(b.width), h: Math.round(b.height),
               winW: w.innerWidth, winH: w.innerHeight };
    })(),
  };
})()`;

const readMain = () => `(() => {
  const root = document.getElementById('everyric-root')?.shadowRoot;
  if (!root) return { panel: false };
  return {
    panel: true,
    lineCount: root.querySelectorAll('.ey-line').length,
    hostVisible: document.getElementById('everyric-root').style.display !== 'none',
    title: root.querySelector('.ey-song-title')?.textContent ?? '',
  };
})()`;

try {
  const sw = ctx.serviceWorkers()[0] ?? await ctx.waitForEvent('serviceworker', { timeout: 15000 });
  const extId = new URL(sw.url()).host;
  await ensureLocalServerPermissionForServerUrl(ctx, sw, extId, LOCAL_SERVER_URL);
  await sw.evaluate(s => chrome.storage.local.set({ settings: s }), {
    serverUrl: LOCAL_SERVER_URL, uiLanguage: 'ko', pipKeepPanel: true,
    // pipKeepPanel은 기본이 false로 바뀌었고(운영자 지시), 마커 없는 저장본은
    // 1회성 마이그레이션이 «레거시 사용자»로 보고 강제로 내린다(src/lib/settings.ts).
    // 이 검사는 «동시 표시 모드»를 재는 것이므로, 마이그레이션을 이미 거친 뒤 사용자가
    // 다시 켠 상태를 흉내 내야 한다 — 마커를 함께 넣지 않으면 true가 무시된다.
    settingsMigrations: ['pipKeepPanel-default-false-2026-08'],
  });

  const page = ctx.pages()[0] ?? await ctx.newPage();
  await page.goto(`https://www.youtube.com/watch?v=${syncedId}`, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForSelector('#everyric-root', { state: 'attached', timeout: 30000 });
  await page.waitForFunction(() => {
    const root = document.getElementById('everyric-root')?.shadowRoot;
    return (root?.querySelectorAll('.ey-line').length ?? 0) > 0;
  }, null, { timeout: 45000, polling: 1000 });

  await page.evaluate(() => {
    const v = document.querySelector('video.html5-main-video') ?? document.querySelector('video');
    if (v) { v.currentTime = 20; void v.play().catch(() => {}); }
  });
  await page.waitForTimeout(1200);
  await page.locator('#everyric-root [title="PiP 창으로 보기"]').first().click();
  await page.waitForTimeout(3000);

  const before = await page.evaluate(readPip());
  check(before.open && before.panel, 'PiP가 열리고 패널 인스턴스가 세워짐', before);
  check(before.lineCount > 0, `PiP에 이 곡 가사가 떠 있음 (${before.lineCount}줄)`);

  // ── 곡이 없는 페이지로 **진짜 SPA 이동** (홈 로고 클릭).
  //
  // page.goto는 쓸 수 없다 — 하드 내비게이션은 content script 컨텍스트를 통째로 새로
  // 만들고 그 과정에서 PiP 창이 함께 사라져, 검증하려던 것이 시작도 못 한다.
  // pushState만 흉내 내는 것도 안 된다 — 페이지 DOM은 그대로라 확장이 «제목이 같은 그
  // 곡»을 다시 찾아 붙이고 만다(실측: 같은 가사가 위키 판본으로 갈아끼워졌다).
  // 유튜브 로고 클릭이 실제 SPA 이동을 일으키고, 그것이 content.ts에서 정확히
  // cleanupForPage 경로다("다음 곡 고르러 홈을 거치는" 그 흐름).
  await page.locator('ytd-topbar-logo-renderer a, a#logo').first().click();
  await page.waitForTimeout(9000);

  const after = await page.evaluate(readPip());
  const main = await page.evaluate(readMain());
  check(after.open, 'PiP 창이 살아남음 (닫히지 않음)', after);
  if (after.open) {
    check(after.panel, 'PiP 패널 인스턴스가 그대로 살아 있음', after);
    check(after.lineCount === 0, `이전 곡 가사가 PiP에서 비워짐 (남은 줄 ${after.lineCount})`, after);
    check(after.laneVisible === false, '이전 곡 레인이 PiP에서 꺼짐', after);
    check(after.emptyState === true, 'PiP 안에 검색·붙여넣기 화면이 떠 있음', after);
    if (after.stateRect) {
      // 「화면 정중앙에 뜬다」의 반증 — 본문 흐름 안에 있으면 상단부터 시작한다
      const centered = Math.abs(after.stateRect.y - (after.stateRect.winH - after.stateRect.h) / 2) < 20;
      check(!centered, '생성/검색 화면이 창 정중앙에 떠 있지 않다(본문 흐름 안)', after.stateRect);
    }
  }
  check(main.panel === true, '메인 패널 인스턴스도 살아 있음', main);
  // ★함정 ⑤ — 비우기를 방송으로 걸면 여기가 0으로 떨어진다. 메인은 숨기기만 하고
  // 내용은 지키므로, 사용자가 다시 열면 보던 곡이 그 자리에 있어야 한다.
  check(main.lineCount > 0, `메인 패널의 가사는 지워지지 않음 (${main.lineCount}줄 보존)`, main);
  check(main.hostVisible === false, '메인 패널은 숨겨짐(곡 없는 페이지)', main);

  const pipPage = ctx.pages().find(p => p !== page);
  if (pipPage) {
    // *-check.png는 .gitignore 대상 — 리포에 산출물을 남기지 않는다
    await pipPage.screenshot({ path: resolve(__dirname, '../pip-nosync-check.png') });
    console.log('screenshot: pip-nosync-check.png');
  }
  console.log(failed ? 'PIP NOSYNC CHECK: FAIL' : 'PIP NOSYNC CHECK: PASS');
  process.exitCode = failed ? 1 : 0;
} catch (e) {
  console.log('PIP NOSYNC CHECK: ERROR —', String(e).slice(0, 400));
  process.exitCode = 1;
} finally {
  await ctx.close();
}
