// 재생목록 표면 분리 P1 수리 검증 — 코드 수정 없이 관측만. 6개 시나리오.
// 팀리드 지시: 실브라우저, list=RD 자동 믹스, 고정 프로필(taskkill 금지).
import { chromium } from 'playwright';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import { mkdirSync } from 'fs';
import { ensureLocalServerPermissionForServerUrl } from './lib/local-server-permission.mjs';
import { readPipPanel } from './lib/pip-panel.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const distDir = resolve(__dirname, '../dist');
const SERVER = 'http://127.0.0.1:8000';
const VIDEO = process.argv[2] ?? 'arX83q0oJhM';

let failed = false;
function check(ok, label, detail) {
  console.log(`${ok ? 'PASS' : 'FAIL'}: ${label}${detail !== undefined ? ' = ' + JSON.stringify(detail) : ''}`);
  if (!ok) failed = true;
  return ok;
}
function info(label, detail) { console.log(`INFO: ${label} = ${JSON.stringify(detail)}`); }

const profileDir = 'C:\\Users\\user\\AppData\\Local\\Temp\\claude\\C--DevAT-Everyric2\\ccf2cc19-4a2e-4cc6-9f6d-e92f242a5aa3\\scratchpad\\ey-pl-p1-profile';
mkdirSync(profileDir, { recursive: true });

const ctx = await chromium.launchPersistentContext(profileDir, {
  ignoreDefaultArgs: ['--disable-extensions'],
  headless: false,
  viewport: { width: 1600, height: 950 },
  args: [
    `--disable-extensions-except=${distDir}`, `--load-extension=${distDir}`,
    '--mute-audio', '--autoplay-policy=no-user-gesture-required',
  ],
});
const sw = ctx.serviceWorkers()[0] ?? await ctx.waitForEvent('serviceworker', { timeout: 20000 });
const extId = new URL(sw.url()).host;
await ensureLocalServerPermissionForServerUrl(ctx, sw, extId, SERVER);
await sw.evaluate(async (url) => {
  const cur = (await chrome.storage.local.get('settings')).settings ?? {};
  // 시작 상태: 메인 on(modPlaylist), PiP는 기본값(pipPlaylist:true) 그대로 둔다 —
  // "양쪽 다 on"에서 출발(시나리오1). pipPlaylist는 DEFAULT_SETTINGS 기본이 true이므로
  // 명시 안 해도 되지만 확실히 하기 위해 같이 켠다.
  await chrome.storage.local.set({ settings: { ...cur, serverUrl: url, uiLanguage: 'ko', modPlaylist: true, pipPlaylist: true } });
}, SERVER);

const page = await ctx.newPage();
page.on('crash', () => console.error('PAGE CRASHED'));

async function mainState() {
  return page.evaluate(() => {
    const root = document.getElementById('everyric-root')?.shadowRoot;
    if (!root) return null;
    const attach = root.querySelector('.ey-attach-playlist');
    const quickBtn = [...root.querySelectorAll('.ey-quick-row .ey-mini')].find((b) => b.title.includes('재생목록'));
    const panel = root.querySelector('.ey-panel');
    const rect = panel ? panel.getBoundingClientRect() : null;
    return {
      attachDisplay: attach ? getComputedStyle(attach).display : null,
      rows: root.querySelectorAll('.ey-pl-row').length,
      statusText: root.querySelector('.ey-pl-status')?.textContent ?? null,
      quickOn: quickBtn ? quickBtn.classList.contains('on') : null,
      panelRect: rect ? { x: Math.round(rect.x), y: Math.round(rect.y), w: Math.round(rect.width), h: Math.round(rect.height) } : null,
    };
  });
}
async function pipState() {
  return page.evaluate(() => {
    const w = window.documentPictureInPicture?.window;
    if (!w) return { open: false };
    const root = w.document.getElementById('everyric-root')?.shadowRoot;
    if (!root) return { open: true, panel: false };
    // .ey-attach-playlist는 filled(PiP) 인스턴스에서 slots.playlistSlot(라이트 DOM 열,
    // pip.ts가 3열 구조로 제공)으로 reparent된다(overlay.ts attach()) — 레인 캔버스와
    // 같은 패턴. 그래서 shadow root 안이 아니라 PiP 문서 전체에서 찾아야 한다.
    const attach = w.document.querySelector('.ey-attach-playlist');
    const quickBtn = [...root.querySelectorAll('.ey-quick-row .ey-mini')].find((b) => b.title.includes('재생목록'));
    return {
      open: true, panel: true,
      attachDisplay: attach ? w.getComputedStyle(attach).display : null,
      attachFound: !!attach,
      rows: w.document.querySelectorAll('.ey-pl-row').length,
      statusText: w.document.querySelector('.ey-pl-status')?.textContent ?? null,
      quickOn: quickBtn ? quickBtn.classList.contains('on') : null,
    };
  });
}
async function clickQuickToggle(surface) {
  return page.evaluate((surface) => {
    const root = surface === 'pip'
      ? window.documentPictureInPicture?.window?.document?.getElementById('everyric-root')?.shadowRoot
      : document.getElementById('everyric-root')?.shadowRoot;
    if (!root) return false;
    const btn = [...root.querySelectorAll('.ey-quick-row .ey-mini')].find((b) => b.title.includes('재생목록'));
    if (!btn) return false;
    btn.click();
    return true;
  }, surface);
}
async function toggleViaSettingsSheet(surface) {
  // 설정 열기 → 모든 범주 펼치기 → "재생목록 패널" 라벨의 체크박스 클릭 → 뒤로가기
  return page.evaluate((surface) => {
    const root = surface === 'pip'
      ? window.documentPictureInPicture?.window?.document?.getElementById('everyric-root')?.shadowRoot
      : document.getElementById('everyric-root')?.shadowRoot;
    if (!root) return { ok: false, reason: 'no root' };
    const gear = [...root.querySelectorAll('.ey-actions button, .ey-header button')].find((b) => b.title === '설정');
    if (!gear) return { ok: false, reason: 'no gear btn' };
    gear.click();
    return { ok: true };
  }, surface).then(async (res) => {
    if (!res.ok) return res;
    await page.waitForTimeout(400);
    const clicked = await page.evaluate((surface) => {
      const root = surface === 'pip'
        ? window.documentPictureInPicture?.window?.document?.getElementById('everyric-root')?.shadowRoot
        : document.getElementById('everyric-root')?.shadowRoot;
      const sheet = root?.querySelector('.ey-settings') ?? root?.querySelector('.ey-state');
      if (!sheet) return { ok: false, reason: 'no sheet' };
      for (const d of sheet.querySelectorAll('details')) d.open = true;
      const rows = [...sheet.querySelectorAll('.ey-settings-row')];
      const row = rows.find((r) => r.querySelector('label')?.textContent?.includes('재생목록 패널'));
      if (!row) return { ok: false, reason: 'no playlist row', rowCount: rows.length };
      const box = row.querySelector('input[type=checkbox]');
      if (!box) return { ok: false, reason: 'no checkbox' };
      box.click();
      return { ok: true };
    }, surface);
    await page.waitForTimeout(300);
    // 뒤로가기(닫기) — back 버튼 또는 헤더의 닫기류
    await page.evaluate((surface) => {
      const root = surface === 'pip'
        ? window.documentPictureInPicture?.window?.document?.getElementById('everyric-root')?.shadowRoot
        : document.getElementById('everyric-root')?.shadowRoot;
      const back = root?.querySelector('.ey-sheet-back');
      if (back) back.click();
    }, surface);
    await page.waitForTimeout(300);
    return clicked;
  });
}

console.log('navigating to', VIDEO, 'in playlist(mix) context');
await page.goto(`https://www.youtube.com/watch?v=${VIDEO}&list=RD${VIDEO}`, { waitUntil: 'domcontentloaded', timeout: 90000 });
await page.waitForFunction(
  () => !!document.getElementById('everyric-root')?.shadowRoot?.querySelector('.ey-line'),
  { timeout: 120000 },
);
await page.waitForTimeout(2500);

// 메인 목록이 실제로 채워질 시간을 준다(스크랩 주기·백오프)
await page.waitForFunction(
  () => (document.getElementById('everyric-root')?.shadowRoot?.querySelectorAll('.ey-pl-row').length ?? 0) > 0,
  { timeout: 20000 },
).catch(() => {});

// PiP 열기
await page.locator('[title="PiP 창으로 보기"]').click();
await page.waitForTimeout(3000);

// ── 시나리오 1: 메인 on + PiP on(기본) → 양쪽 다 표시 ──────────────────
const m1 = await mainState();
const p1 = await pipState();
info('S1 메인 상태', m1);
info('S1 PiP 상태', p1);
check(m1.rows > 0 && m1.attachDisplay !== 'none', 'S1 메인 재생목록 표시됨', m1);
check(p1.open && p1.rows > 0 && p1.attachDisplay !== 'none', 'S1 PiP 재생목록 표시됨', p1);

const initialMainRect = m1.panelRect;

// ── 시나리오 2: 메인에서 끔(퀵 토글) → 메인만 사라지고 PiP는 유지 ──────────
await clickQuickToggle('main');
await page.waitForTimeout(500);
const m2 = await mainState();
const p2 = await pipState();
info('S2 메인 끈 뒤 메인 상태(퀵토글)', m2);
info('S2 메인 끈 뒤 PiP 상태(퀵토글)', p2);
check(m2.attachDisplay === 'none' || m2.rows === 0, 'S2(퀵토글) 메인 재생목록이 꺼짐', m2);
check(p2.open && p2.rows > 0 && p2.attachDisplay !== 'none', 'S2(퀵토글) PiP는 계속 표시됨(핵심 회귀 케이스)', p2);

// 원복(퀵 토글로 다시 켬)
await clickQuickToggle('main');
await page.waitForTimeout(500);

// 설정 시트 경로로 다시 확인
const s2sheet = await toggleViaSettingsSheet('main');
info('S2 설정시트 토글 결과', s2sheet);
await page.waitForTimeout(500);
const m2b = await mainState();
const p2b = await pipState();
info('S2 메인 끈 뒤 메인 상태(설정시트)', m2b);
info('S2 메인 끈 뒤 PiP 상태(설정시트)', p2b);
check(s2sheet.ok, 'S2 설정시트에서 재생목록 행을 찾아 클릭함', s2sheet);
check(m2b.attachDisplay === 'none' || m2b.rows === 0, 'S2(설정시트) 메인 재생목록이 꺼짐', m2b);
check(p2b.open && p2b.rows > 0 && p2b.attachDisplay !== 'none', 'S2(설정시트) PiP는 계속 표시됨', p2b);

// 메인 다시 켠다(다음 시나리오 준비)
await clickQuickToggle('main');
await page.waitForTimeout(800);

// ── 시나리오 3: PiP에서 끔(코너 퀵토글) → PiP만 사라지고 메인은 불변 ────────
const m3before = await mainState();
await clickQuickToggle('pip');
await page.waitForTimeout(500);
const m3 = await mainState();
const p3 = await pipState();
info('S3 PiP 끈 뒤 메인 상태(퀵토글)', m3);
info('S3 PiP 끈 뒤 PiP 상태(퀵토글)', p3);
check(p3.attachDisplay === 'none' || p3.rows === 0, 'S3(퀵토글) PiP 재생목록이 꺼짐', p3);
check(m3.rows > 0 && m3.attachDisplay !== 'none', 'S3(퀵토글) 메인 상태 불변(계속 표시)', m3);
check(m3.rows === m3before.rows, 'S3(퀵토글) 메인 행 수 불변', { before: m3before.rows, after: m3.rows });

// 원복
await clickQuickToggle('pip');
await page.waitForTimeout(500);

// 설정 시트 경로로 다시 확인(PiP 자신의 설정 시트)
const s3sheet = await toggleViaSettingsSheet('pip');
info('S3 설정시트(PiP) 토글 결과', s3sheet);
await page.waitForTimeout(500);
const m3b = await mainState();
const p3b = await pipState();
info('S3 PiP 끈 뒤 메인 상태(설정시트)', m3b);
info('S3 PiP 끈 뒤 PiP 상태(설정시트)', p3b);
check(s3sheet.ok, 'S3 PiP 설정시트에서 재생목록 행을 찾아 클릭함', s3sheet);
check(p3b.attachDisplay === 'none' || p3b.rows === 0, 'S3(설정시트) PiP 재생목록이 꺼짐', p3b);
check(m3b.rows > 0 && m3b.attachDisplay !== 'none', 'S3(설정시트) 메인 상태 불변', m3b);

// PiP 다시 켠다
await clickQuickToggle('pip');
await page.waitForTimeout(800);

// ── 시나리오 6: 토글 후 메인 패널 geometry 불변 ─────────────────────────
const finalMain = await mainState();
info('S6 최종 메인 panelRect', finalMain.panelRect);
info('S6 최초 메인 panelRect', initialMainRect);
check(
  JSON.stringify(finalMain.panelRect) === JSON.stringify(initialMainRect),
  'S6 여러 토글 왕복 후 메인 패널 geometry(위치·크기) 불변',
  { initial: initialMainRect, final: finalMain.panelRect },
);

// ── 시나리오 4: 메인 off + PiP만 on 상태에서 PiP 목록이 실제로 채워지는가(조달 게이트) ──
// page.goto()는 YouTube SPA 내부 라우팅이 아니라 진짜 풀 네비게이션이라 PiP 창 자체가
// 닫힌다(document Picture-in-Picture는 오프너 페이지의 생명주기에 묶인다) — 실측으로
// 확인됨. 그래서 "메인 off·PiP on"을 처음부터 설정한 뒤 새 탭 세션으로 프레시 로딩해
// 조달 게이트가 실제로 도는지를 잰다(캐시 재사용이 아니라 진짜 첫 스크랩인지 확인하는
// 목적은 동일 — 이 페이지는 이 프로필에서 처음 여는 영상이다).
await ctx.pages().find((p) => p !== page)?.close().catch(() => {}); // 이전 PiP 창 정리
await page.close().catch(() => {});
await sw.evaluate(async () => {
  const cur = (await chrome.storage.local.get('settings')).settings ?? {};
  await chrome.storage.local.set({ settings: { ...cur, modPlaylist: false, pipPlaylist: true } });
});
const page2 = await ctx.newPage();
const VIDEO2 = process.argv[3] ?? 'lw7pcm1W5tw';
await page2.goto(`https://www.youtube.com/watch?v=${VIDEO2}&list=RD${VIDEO2}`, { waitUntil: 'domcontentloaded', timeout: 90000 });
await page2.waitForFunction(
  () => !!document.getElementById('everyric-root')?.shadowRoot?.querySelector('.ey-line'),
  { timeout: 120000 },
);
await page2.waitForTimeout(2000);
const preNav = await page2.evaluate(() => {
  const root = document.getElementById('everyric-root')?.shadowRoot;
  const attach = root?.querySelector('.ey-attach-playlist');
  return { attachDisplay: attach ? getComputedStyle(attach).display : null, rows: root?.querySelectorAll('.ey-pl-row').length ?? 0 };
});
check(preNav.attachDisplay === 'none' || preNav.rows === 0, 'S4 사전조건: 프레시 로딩에서 메인이 꺼져 있다(modPlaylist:false)', preNav);

await page2.locator('[title="PiP 창으로 보기"]').click();
await page2.waitForTimeout(3000);
await page2.waitForFunction(() => {
  const w = window.documentPictureInPicture?.window;
  return (w?.document.querySelectorAll('.ey-pl-row').length ?? 0) > 0;
}, { timeout: 20000 }).catch(() => {});
const p4 = await page2.evaluate(() => {
  const w = window.documentPictureInPicture?.window;
  if (!w) return { open: false };
  const attach = w.document.querySelector('.ey-attach-playlist');
  return {
    open: true,
    attachDisplay: attach ? w.getComputedStyle(attach).display : null,
    rows: w.document.querySelectorAll('.ey-pl-row').length,
    statusText: w.document.querySelector('.ey-pl-status')?.textContent ?? null,
  };
});
info('S4 프레시 로딩(메인 off·PiP on) PiP 상태 — 조달 게이트 수리 확인', p4);
check(p4.open && p4.rows > 0, `S4 메인 off인데도 PiP 목록이 실제로 채워짐(${p4.rows}곡) — 조달 게이트 수리 확인`, p4);

await page.screenshot({ path: resolve(__dirname, '../pl-p1-final.png') }).catch(() => {});
await ctx.close().catch(() => {});
console.log(failed ? '\n== 결과: FAIL 있음' : '\n== 결과: 전부 PASS');
process.exit(failed ? 1 : 0);
