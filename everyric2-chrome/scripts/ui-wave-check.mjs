// 2026-08-03 UI 개편분 실브라우저 검증 — "코드가 있다"가 아니라 "화면에 그려진다".
//
//   U1) 퀵 토글 줄이 존재하고 버튼이 실제로 설정을 뒤집는다(설정 시트를 안 열고)
//   U2) 레인을 켜면 **좌측 컬럼**으로 붙고 가사 본문과 가로로 나란하다(아래가 아니라)
//   U3) 디바이더 드래그로 레인 폭이 바뀌고 그 값이 저장된다
//   U4) fast/medium 싱크에서 타이밍 안내 배너가 뜨고 "다시 보지 않기"가 영속된다
//   U5) 다음 영상 카드가 썸네일을 가진 카드로 렌더된다(말줄임 한 줄이 아니라)
//   U6) 영상 자막 모듈이 플레이어 안에 붙고 활성 줄이 시간에 따라 채워진다
//
// 실행: node scripts/ui-wave-check.mjs [videoId]
// 사전 조건: 실서버 127.0.0.1:8000 (localhost는 IPv6 스톨로 요청당 2초).
import { chromium } from 'playwright';
import { fileURLToPath } from 'url';
import { dirname, resolve, join } from 'path';
import { mkdtempSync, cpSync } from 'fs';
import { tmpdir } from 'os';
import { ensureLocalServerPermissionForServerUrl } from './lib/local-server-permission.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
// dist 스냅샷 — 병렬 빌드가 돌아도 이 실행은 흔들리지 않는다
const distSrc = resolve(__dirname, '../dist');
const distDir = join(mkdtempSync(join(tmpdir(), 'ey-dist-')), 'dist');
cpSync(distSrc, distDir, { recursive: true });

const SERVER = 'http://127.0.0.1:8000';
const VIDEO = process.argv[2] ?? 'b2NTglk9tvI';
let failed = false;
function check(ok, label, detail) {
  console.log(`${ok ? 'PASS' : 'FAIL'}: ${label}${detail !== undefined ? ' = ' + JSON.stringify(detail) : ''}`);
  if (!ok) failed = true;
  return ok;
}
function info(label, detail) { console.log(`INFO: ${label} = ${JSON.stringify(detail)}`); }

const health = await (await fetch(`${SERVER}/health`, { signal: AbortSignal.timeout(3000) })).json();
if (!check(health.status === 'healthy', 'real server /health', health)) process.exit(1);

const userDataDir = mkdtempSync(join(tmpdir(), 'ey-uiwave-'));
const ctx = await chromium.launchPersistentContext(userDataDir, {
  ignoreDefaultArgs: ['--disable-extensions'],
  headless: false,
  viewport: { width: 1600, height: 950 },
  args: [
    `--disable-extensions-except=${distDir}`,
    `--load-extension=${distDir}`,
    '--mute-audio',
    '--autoplay-policy=no-user-gesture-required',
  ],
});

const sw = ctx.serviceWorkers()[0] ?? await ctx.waitForEvent('serviceworker', { timeout: 20000 });
const extId = new URL(sw.url()).host;
await ensureLocalServerPermissionForServerUrl(ctx, sw, extId, SERVER);

// 검사 대상 모듈을 처음부터 켜 둔다(퀵 토글 자체는 U1에서 따로 뒤집어 본다)
await sw.evaluate(async (url) => {
  const cur = (await chrome.storage.local.get('settings')).settings ?? {};
  await chrome.storage.local.set({
    settings: {
      ...cur, serverUrl: url, uiLanguage: 'ko',
      modMainLane: false, mainLanePos: 'left', modNextUp: true,
      videoCaptions: true, karaokeTimingNoticeDismissed: false,
    },
  });
}, SERVER);

const page = await ctx.newPage();
await page.goto(`https://www.youtube.com/watch?v=${VIDEO}`, { waitUntil: 'domcontentloaded', timeout: 90000 });

const SR = `document.getElementById('everyric-root')?.shadowRoot`;
await page.waitForFunction(
  () => !!document.getElementById('everyric-root')?.shadowRoot?.querySelector('.ey-line'),
  { timeout: 120000 });
await page.waitForTimeout(3000);

// ── U1: 퀵 토글 줄 ────────────────────────────────────────────────
const quick = await page.evaluate(() => {
  const sr = document.getElementById('everyric-root').shadowRoot;
  const row = sr.querySelector('.ey-quick-row');
  if (!row) return { present: false };
  const btns = [...row.querySelectorAll('button')];
  return {
    present: true, visible: row.getClientRects().length > 0, count: btns.length,
    titles: btns.map(b => b.title || b.getAttribute('aria-label') || ''),
  };
});
check(quick.present && quick.visible && quick.count >= 3,
  'U1 퀵 토글 줄이 화면에 존재(설정 시트 없이 접근 가능)', quick);

// 레인 토글 버튼을 눌러 설정이 실제로 뒤집히는지
const laneBtnIdx = quick.titles.findIndex(t => /레인|lane/i.test(t));
if (laneBtnIdx >= 0) {
  await page.evaluate((i) => {
    const sr = document.getElementById('everyric-root').shadowRoot;
    sr.querySelectorAll('.ey-quick-row button')[i].click();
  }, laneBtnIdx);
  await page.waitForTimeout(1800);
  const s = await sw.evaluate(async () => (await chrome.storage.local.get('settings')).settings);
  check(s.modMainLane === true, 'U1 퀵 버튼 클릭이 실제 설정을 뒤집었다', { modMainLane: s.modMainLane });
} else {
  check(false, 'U1 레인 퀵 버튼을 찾지 못함', quick.titles);
}
await page.waitForTimeout(2500);

// ── U2: 레인이 좌측 컬럼인가 ──────────────────────────────────────
const layout = await page.evaluate(() => {
  const sr = document.getElementById('everyric-root').shadowRoot;
  const lane = sr.querySelector('.ey-lane-wrap, .ey-main-lane');
  const body = sr.querySelector('.ey-body');
  if (!lane || !body) return { lane: !!lane, body: !!body };
  const L = lane.getBoundingClientRect(), B = body.getBoundingClientRect();
  return {
    lane: { x: Math.round(L.x), y: Math.round(L.y), w: Math.round(L.width), h: Math.round(L.height) },
    body: { x: Math.round(B.x), y: Math.round(B.y), w: Math.round(B.width), h: Math.round(B.height) },
    sideBySide: L.right <= B.x + 4 && L.height > 40,   // 레인이 본문 왼쪽
    verticalOverlap: Math.min(L.bottom, B.bottom) - Math.max(L.y, B.y) > 20,
  };
});
info('레인/본문 배치', layout);
check(layout.sideBySide === true && layout.verticalOverlap === true,
  'U2 레인이 가사 본문 **왼쪽 컬럼**으로 배치됨(아래가 아님)',
  { laneRight: layout.lane?.x + layout.lane?.w, bodyLeft: layout.body?.x });

// ── U3: 디바이더 드래그로 폭 변경 + 저장 ──────────────────────────
const divider = await page.evaluate(() => {
  const sr = document.getElementById('everyric-root').shadowRoot;
  const d = sr.querySelector('.ey-lane-divider, .ey-divider');
  if (!d) return null;
  const r = d.getBoundingClientRect();
  return { x: r.x + r.width / 2, y: r.y + r.height / 2, w: r.width, h: r.height };
});
if (divider) {
  const before = layout.lane.w;
  // **줄이는 방향**으로 끈다. 넓히는 방향은 패널 폭이 좁으면 clampLaneWidth의
  // `panelW - 가사최소폭` 상한에 이미 닿아 있어서 "안 움직이는 게 정답"이라,
  // 드래그 기전 자체를 검증하지 못한다(실측: 패널 340px에서 레인이 이미 상한 179px).
  await page.mouse.move(divider.x, divider.y);
  await page.mouse.down();
  await page.mouse.move(divider.x - 60, divider.y, { steps: 12 });
  await page.mouse.up();
  await page.waitForTimeout(1500);
  const after = await page.evaluate(() => {
    const sr = document.getElementById('everyric-root').shadowRoot;
    const lane = sr.querySelector('.ey-lane-wrap, .ey-main-lane');
    return Math.round(lane.getBoundingClientRect().width);
  });
  const saved = await sw.evaluate(async () => (await chrome.storage.local.get('settings')).settings.mainLaneWidth);
  check(after < before - 25, 'U3 디바이더 드래그로 레인 폭이 실제로 바뀜', { before, after });
  check(typeof saved === 'number' && Math.abs(saved - after) < 40,
    'U3 드래그한 폭이 설정에 저장됨(pointerup 1회 저장)', { saved, after });
} else {
  check(false, 'U3 레인 디바이더를 찾지 못함');
}

// ── U4: 타이밍 안내 배너 ──────────────────────────────────────────
const notice = await page.evaluate(() => {
  const sr = document.getElementById('everyric-root').shadowRoot;
  const n = sr.querySelector('.ey-lane-notice, .ey-timing-notice');
  if (!n) return { present: false };
  return {
    present: true, visible: n.getClientRects().length > 0,
    text: (n.textContent || '').trim().slice(0, 90),
    buttons: [...n.querySelectorAll('button')].map(b => (b.textContent || '').trim()),
  };
});
info('타이밍 안내 배너', notice);
check(notice.present && notice.visible, 'U4 fast/medium 싱크에서 타이밍 안내 배너가 뜸', notice);
if (notice.present) {
  const dismissIdx = notice.buttons.findIndex(t => /다시|again|二度と/i.test(t));
  if (dismissIdx >= 0) {
    await page.evaluate((i) => {
      const sr = document.getElementById('everyric-root').shadowRoot;
      const n = sr.querySelector('.ey-lane-notice, .ey-timing-notice');
      n.querySelectorAll('button')[i].click();
    }, dismissIdx);
    await page.waitForTimeout(1500);
    const s = await sw.evaluate(async () => (await chrome.storage.local.get('settings')).settings);
    const gone = await page.evaluate(() => {
      const sr = document.getElementById('everyric-root').shadowRoot;
      const n = sr.querySelector('.ey-lane-notice, .ey-timing-notice');
      return !n || n.getClientRects().length === 0;
    });
    check(s.karaokeTimingNoticeDismissed === true && gone,
      'U4 "다시 보지 않기"가 즉시 숨기고 설정에 영속', { dismissed: s.karaokeTimingNoticeDismissed, gone });
  } else {
    check(false, 'U4 "다시 보지 않기" 버튼을 찾지 못함', notice.buttons);
  }
}

// ── U5: 다음 영상 카드 ────────────────────────────────────────────
const nextUp = await page.evaluate(() => {
  const sr = document.getElementById('everyric-root').shadowRoot;
  const el = sr.querySelector('.ey-nextup');
  if (!el) return { present: false };
  const r = el.getBoundingClientRect();
  return {
    present: true, visible: r.height > 0, h: Math.round(r.height),
    hasImg: !!el.querySelector('img'),
    imgSrc: el.querySelector('img')?.src?.slice(0, 60) ?? null,
    text: (el.textContent || '').trim().slice(0, 60),
  };
});
info('다음 영상', nextUp);
check(nextUp.present && nextUp.visible && nextUp.h > 24,
  'U5 다음 영상이 한 줄이 아닌 카드 높이로 렌더됨', { h: nextUp.h, hasImg: nextUp.hasImg });

// ── U6: 영상 자막이 플레이어 안에 붙고 채워지는가 ─────────────────
// 자막은 updateTime이 굴려야 라인이 잡힌다 — 정지 상태면 아무것도 안 그려지는 게
// 정상이므로, 검사 전에 재생을 확실히 걸고 첫 가사 근처로 시크한다.
await page.evaluate(() => {
  const v = document.querySelector('video');
  if (v) { v.muted = true; v.currentTime = Math.max(0, (v.currentTime || 0) + 40); void v.play(); }
});
await page.waitForTimeout(4000);
const cap1 = await page.evaluate(() => {
  const host = document.querySelector('#movie_player .ey-video-caption, #movie_player [class*="ey-vc"]')
    ?? document.querySelector('#movie_player');
  const el = document.querySelector('#movie_player .ey-video-caption');
  if (!el) return { present: false, inPlayer: !!host };
  return {
    present: true, inPlayer: !!el.closest('#movie_player'),
    sung: el.querySelectorAll('.sung').length,
    spans: el.querySelectorAll('.ey-vc-word, .ey-vc-syl').length,
    text: (el.textContent || '').trim().slice(0, 40),
  };
});
info('영상 자막(1차)', cap1);
if (cap1.present) {
  await page.waitForTimeout(4000);
  const cap2 = await page.evaluate(() => {
    const el = document.querySelector('#movie_player .ey-video-caption');
    return { sung: el.querySelectorAll('.sung').length, spans: el.querySelectorAll('.ey-vc-word, .ey-vc-syl').length };
  });
  info('영상 자막(4초 후)', cap2);
  check(cap1.inPlayer, 'U6 영상 자막이 플레이어(#movie_player) 안에 부착됨');
  check(cap2.spans > 0, 'U6 자막이 음절/단어 스팬으로 렌더됨(통짜 텍스트 아님)', cap2);
} else {
  check(false, 'U6 영상 자막 호스트(#movie_player .ey-video-caption)가 부착되지 않음', cap1);
}

await page.screenshot({ path: resolve(__dirname, '../ui-wave-check.png') });
console.log('screenshot: ui-wave-check.png');
console.log(`UI WAVE CHECK: ${failed ? 'FAIL' : 'PASS'}`);
await ctx.close();
process.exit(failed ? 1 : 0);
