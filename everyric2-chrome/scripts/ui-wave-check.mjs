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
import { resolveVideoId } from './lib/pick-song.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
// dist 스냅샷 — 병렬 빌드가 돌아도 이 실행은 흔들리지 않는다
const distSrc = resolve(__dirname, '../dist');
const distDir = join(mkdtempSync(join(tmpdir(), 'ey-dist-')), 'dist');
cpSync(distSrc, distDir, { recursive: true });

const SERVER = 'http://127.0.0.1:8000';
// U4(타이밍 안내 배너)는 **fast/medium 싱크에서만** 뜬다 — overlay.ts의 조건이
// `laneShown && route !== 'heavy' && !dismissed`이기 때문이다. 그래서 곡을 아무거나 쓰면
// 안 된다: **heavy 곡을 고르면 U4는 원리적으로 성립하지 못한 채 실패한다**(실측 2026-08-04 —
// 기존 기본값 b2NTglk9tvI가 heavy라 U4가 계속 FAIL이었고, 그것을 제품 결함으로 오해할 뻔했다).
// 인자로 곡을 직접 주면 그 곡을 그대로 쓴다(조건 확인은 호출자 책임).
const pickedVideo = process.argv[2]
  ? { videoId: process.argv[2], source: 'argv' }
  : resolveVideoId('b2NTglk9tvI', { minLines: 20, routeIn: ['fast', 'medium'] });
if (pickedVideo.source !== 'argv') {
  console.log(`[곡] ${pickedVideo.videoId} (route=${pickedVideo.route}) — ${pickedVideo.note}`);
}
const VIDEO = pickedVideo.videoId;
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
      // 레인은 **꺼진 채로 시작해야 한다** — U1이 퀵 버튼을 눌러 true로 뒤집히는 것을
      // 확인하는 검사라, 켜 두면 그 클릭이 도리어 끄게 되어 U1·U2가 무너진다.
      // U4(배너)가 요구하는 laneShown은 그 U1 클릭 이후 이미 충족돼 있다.
      modMainLane: false, mainLanePos: 'left', modNextUp: true,
      // 다음 영상 카드는 **재생목록 부착 패널 안에서** 그려진다(overlay.renderPlaylistPanel이
      // buildNextUpCard를 부르는 유일한 자리이고, 그 함수는 playlistVisible()이 아니면
      // 조기 반환한다). modNextUp만 켜면 데이터 조달만 돌고 화면에는 안 나온다 — U5가
      // 그래서 실패했다(2026-08-04).
      modPlaylist: true,
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
  // 클래스는 `.ey-nextup-card`다. 예전 `.ey-nextup`은 **어느 소스에도 없는 토큰**이라
  // 이 검사는 모듈을 다 켜 놔도 영원히 present:false였다(실측 2026-08-04: modPlaylist를
  // 켜고 돌려도 그대로 실패 → 설정 노후가 아니라 선택자 노후였다).
  const el = sr.querySelector('.ey-nextup-card');
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
// 고정 대기가 아니라 **조건 폴링**으로 잰다. 자막 채움은 시간 구동이라 재생이 필요한데,
// 유튜브가 버퍼링으로 멈추면 «4초 뒤 스팬 수»는 제품과 무관하게 0이 된다(앞선 회차에서
// ~60초 지점 buffered 고정으로 그렇게 실패했다). 그래서 스팬이 생길 때까지 기다리고,
// 끝내 못 만나면 **재생기 상태를 함께 찍어 환경 실패와 제품 실패가 로그에서 갈리게** 한다.
const CAP_TIMEOUT_MS = 15000;
const readCaption = () => page.evaluate(() => {
  const el = document.querySelector('#movie_player .ey-video-caption');
  const v = document.querySelector('video');
  const media = v ? {
    readyState: v.readyState, paused: v.paused, currentTime: Number(v.currentTime.toFixed(2)),
    bufferedEnd: v.buffered.length ? Number(v.buffered.end(v.buffered.length - 1).toFixed(3)) : null,
  } : null;
  if (!el) return { present: false, inPlayer: !!document.querySelector('#movie_player'), media };
  return {
    present: true, inPlayer: !!el.closest('#movie_player'),
    sung: el.querySelectorAll('.sung').length,
    spans: el.querySelectorAll('.ey-vc-word, .ey-vc-syl').length,
    text: (el.textContent || '').trim().slice(0, 40),
    media,
  };
});

let cap = await readCaption();
const capStart = Date.now();
while ((!cap.present || cap.spans === 0) && Date.now() - capStart < CAP_TIMEOUT_MS) {
  await page.waitForTimeout(700);
  cap = await readCaption();
}
info('영상 자막', { ...cap, 대기ms: Date.now() - capStart });

/**
 * 실패했을 때 «환경 탓인가 제품 탓인가»를 문구 자체가 말하게 한다 — 다음에 진짜 결함이
 * 나면 이 로그가 판정을 대신한다. 재생이 못 굴러간 정황(디코드 준비 안 됨/정지/재생
 * 위치가 버퍼 끝을 넘음)이면 환경으로 본다.
 */
const stallNote = (m) => {
  if (!m) return '재생기 상태를 못 읽음';
  const beyond = m.bufferedEnd !== null && m.currentTime > m.bufferedEnd + 0.5;
  const stalled = m.readyState < 3 || m.paused || beyond;
  return `${stalled ? '환경(재생 정지·버퍼 부족)' : '제품'} — `
    + `readyState=${m.readyState} paused=${m.paused} currentTime=${m.currentTime} bufferedEnd=${m.bufferedEnd}`;
};

if (!cap.present) {
  check(false, `U6 영상 자막 호스트가 부착되지 않음 [${stallNote(cap.media)}]`, cap);
} else {
  check(cap.inPlayer, 'U6 영상 자막이 플레이어(#movie_player) 안에 부착됨');
  check(cap.spans > 0,
    cap.spans > 0
      ? 'U6 자막이 음절/단어 스팬으로 렌더됨(통짜 텍스트 아님)'
      : `U6 자막 스팬이 ${CAP_TIMEOUT_MS / 1000}초 동안 하나도 안 생김 [${stallNote(cap.media)}]`,
    { spans: cap.spans, sung: cap.sung, media: cap.media });
}

await page.screenshot({ path: resolve(__dirname, '../ui-wave-check.png') });
console.log('screenshot: ui-wave-check.png');
console.log(`UI WAVE CHECK: ${failed ? 'FAIL' : 'PASS'}`);
await ctx.close();
process.exit(failed ? 1 : 0);
