// 2026-08-03 2차 피드백 수리분 실브라우저 검증 — "코드가 있다"가 아니라 "화면에 그려진다".
//
//   R1) 번역 언어 칩이 퀵 토글 줄 **안** 오른쪽에 있다 (별도 행이 사라졌다)
//   R2) 공지 읽음이 영속된다 — 시트를 연 뒤 storage에 **문자열**로 저장되고,
//       새로고침해도 안 읽음 점이 되살아나지 않는다 (int/string 불일치 회귀 방지)
//   R3) 영상 자막 발음 줄 기본색이 청회색(#a9c4e6)이다 — 노랑×노랑 대비 사고 회귀 방지
//   R4) 설정 시트에 mainFontScale 슬라이더·hidePronForEnglish 토글·전체 초기화 버튼이 있다
//   R5) 초기화 버튼이 헤더가 아니라 풋터(별점 옆)에 '싱크 초기화' 의미로 있다
//       (2026-08-04 #46 항목2 갱신 — 원래는 헤더에서 문구만 바뀐 걸 봤지만, 이후
//       버튼 자체가 풋터로 옮겨갔다)
//
// 실행: node scripts/feedback-round2-check.mjs [videoId]
// 사전 조건: 실서버 127.0.0.1:8000 (localhost는 IPv6 스톨), dist 최신 빌드.
import { chromium } from 'playwright';
import { fileURLToPath } from 'url';
import { dirname, resolve, join } from 'path';
import { mkdtempSync, cpSync } from 'fs';
import { tmpdir } from 'os';
import { ensureLocalServerPermissionForServerUrl } from './lib/local-server-permission.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
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
function skip(label, why) { console.log(`SKIP: ${label} — ${why}`); }
function info(label, detail) { console.log(`INFO: ${label} = ${JSON.stringify(detail)}`); }

const health = await (await fetch(`${SERVER}/health`, { signal: AbortSignal.timeout(3000) })).json();
if (!check(health.status === 'healthy', 'real server /health', health)) process.exit(1);
const notices = await (await fetch(`${SERVER}/api/notices`, { signal: AbortSignal.timeout(3000) })).json()
  .catch(() => null);
const hasNotice = Array.isArray(notices?.notices) && notices.notices.length > 0;
info('서버 공지 수', notices?.notices?.length ?? null);

const userDataDir = mkdtempSync(join(tmpdir(), 'ey-r2-'));
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

await sw.evaluate(async (url) => {
  const cur = (await chrome.storage.local.get('settings')).settings ?? {};
  await chrome.storage.local.set({
    settings: { ...cur, serverUrl: url, uiLanguage: 'ko', videoCaptions: true },
  });
}, SERVER);

const page = await ctx.newPage();
await page.goto(`https://www.youtube.com/watch?v=${VIDEO}`, { waitUntil: 'domcontentloaded', timeout: 90000 });
await page.waitForFunction(
  () => !!document.getElementById('everyric-root')?.shadowRoot?.querySelector('.ey-line'),
  { timeout: 120000 });
await page.waitForTimeout(3000);

// ── R1: 언어 칩이 퀵 토글 줄 안 오른쪽에 ─────────────────────────
const chips = await page.evaluate(() => {
  const sr = document.getElementById('everyric-root').shadowRoot;
  const inRow = sr.querySelector('.ey-quick-row .ey-lang-chips');
  const stray = sr.querySelector('.ey-panel > .ey-lang-chips'); // 옛 별도 행
  const row = sr.querySelector('.ey-quick-row');
  if (!inRow || !row) return { inRow: !!inRow, stray: !!stray };
  const rowRect = row.getBoundingClientRect();
  const chipRect = inRow.getBoundingClientRect();
  const firstBtn = row.querySelector('button')?.getBoundingClientRect();
  return {
    inRow: true,
    stray: !!stray,
    visible: inRow.getClientRects().length > 0,
    sameLine: Math.abs(chipRect.y + chipRect.height / 2 - (rowRect.y + rowRect.height / 2)) < 12,
    rightAligned: firstBtn ? chipRect.x > firstBtn.x + 40 : null,
  };
});
check(chips.inRow === true && chips.stray === false && chips.visible !== false && chips.sameLine !== false,
  'R1 번역 칩이 퀵 토글 줄 안(같은 행)·별도 행 없음', chips);

// ── R2: 공지 읽음 영속 ────────────────────────────────────────────
if (!hasNotice) {
  skip('R2 공지 읽음 영속', '서버에 활성 공지가 없음 (POST /api/notices로 심은 뒤 재실행)');
} else {
  const noticesBtnHandle = await page.evaluateHandle(() => {
    const sr = document.getElementById('everyric-root').shadowRoot;
    return [...sr.querySelectorAll('.ey-actions button')]
      .find(b => /공지|notice/i.test(b.title || ''));
  });
  const hasBtn = await page.evaluate(b => !!b, noticesBtnHandle);
  if (!check(hasBtn, 'R2 공지 버튼 존재')) {
    // 버튼이 없으면 이후 단계 무의미
  } else {
    const dotBefore = await page.evaluate(b => b.innerHTML.includes('ey-dot') || !!b.querySelector('[class*="dot"], [class*="unread"]'), noticesBtnHandle);
    info('R2 읽기 전 안읽음 점', dotBefore);
    await page.evaluate(b => b.click(), noticesBtnHandle);
    await page.waitForTimeout(2500); // 시트 렌더 + markNoticesSeen
    const seen = await sw.evaluate(async () => (await chrome.storage.local.get('ey_notices_seen')).ey_notices_seen);
    check(typeof seen === 'string' && seen.length > 0,
      'R2a 읽음 표식이 문자열로 저장됨 (int/string 회귀 방지)', { seen, type: typeof seen });
    await page.reload({ waitUntil: 'domcontentloaded' });
    await page.waitForFunction(
      () => !!document.getElementById('everyric-root')?.shadowRoot?.querySelector('.ey-actions'),
      { timeout: 120000 });
    await page.waitForTimeout(4000); // refreshNoticesButton probe 완료 대기
    const dotAfter = await page.evaluate(() => {
      const sr = document.getElementById('everyric-root').shadowRoot;
      const btn = [...sr.querySelectorAll('.ey-actions button')]
        .find(b => /공지|notice/i.test(b.title || ''));
      if (!btn) return { btn: false };
      const dot = btn.querySelector('[class*="dot"], [class*="unread"]');
      const visible = dot ? getComputedStyle(dot).display !== 'none' && dot.getClientRects().length > 0 : false;
      return { btn: true, dotVisible: visible };
    });
    check(dotAfter.btn === true && dotAfter.dotVisible === false,
      'R2b 새로고침 뒤 안읽음 점이 되살아나지 않음', dotAfter);
  }
}

// ── R3: 영상 자막 발음 줄 색 ─────────────────────────────────────
const pronColor = await page.evaluate(() => {
  const host = document.querySelector('.ey-video-caption');
  if (!host) return { host: false };
  const colored = [...host.querySelectorAll('div')]
    .map(d => getComputedStyle(d).color)
    .filter(c => c && c !== 'rgb(255, 255, 255)');
  return { host: true, colors: [...new Set(colored)] };
});
if (!pronColor.host) {
  skip('R3 자막 발음 줄 색', '자막 호스트 미부착 (이 곡에서 자막 모듈이 안 떴음)');
} else {
  check(pronColor.colors.some(c => c === 'rgb(169, 196, 230)'),
    'R3 발음 줄 기본색이 청회색(#a9c4e6) — 노랑 하이라이트와 대비', pronColor);
}

// ── R4: 설정 시트 신규 컨트롤 3종 ────────────────────────────────
await page.evaluate(() => {
  const sr = document.getElementById('everyric-root').shadowRoot;
  const gear = [...sr.querySelectorAll('.ey-actions button')].find(b => /설정|settings/i.test(b.title || ''));
  gear?.click();
});
await page.waitForTimeout(1500);
const settingsProbe = await page.evaluate(() => {
  const sr = document.getElementById('everyric-root').shadowRoot;
  const sheet = sr.querySelector('.ey-settings');
  if (!sheet) return { sheet: false };
  // 접이식 범주를 전부 펼친다
  for (const d of sheet.querySelectorAll('details')) d.open = true;
  const text = sheet.textContent ?? '';
  const ranges = sheet.querySelectorAll('input[type="range"]').length;
  const fullResetBtn = [...sheet.querySelectorAll('button')]
    .some(b => /전체 초기화|full reset/i.test(b.textContent ?? ''));
  return {
    sheet: true,
    ranges,
    hasPronEnRow: /영어.*발음|발음.*영어/i.test(text),
    hasFontScaleRow: /글자 크기|폰트/.test(text),
    fullResetBtn,
  };
});
check(settingsProbe.sheet === true && settingsProbe.hasPronEnRow === true
  && settingsProbe.fullResetBtn === true && settingsProbe.ranges >= 2,
  'R4 설정 시트: 영어 발음 끔·폰트 슬라이더·전체 초기화 존재', settingsProbe);

// ── R5: 초기화 버튼 — 헤더가 아니라 풋터(별점 옆)로 이동, 문구는 초기화 의미 ────
// (#46 항목2, 2026-08-04: regenBtn→resetSyncBtn 개명 + 헤더→풋터 이동. 헤더에는
// 더 이상 이 버튼이 없는 게 정상 — reset-sync-btn-smoke.mjs R1/R2와 같은 전제를 공유한다.
// "재생성"이라는 이름은 REGENERATE_SYNC 와이어 계약(깊이 올리기, onDepthUpgrade)에는
// 그대로 남아 있으니 그 문구가 다른 버튼에 있는 건 정상 — 여기선 초기화 버튼 자체만 본다.)
const resetBtnProbe = await page.evaluate(() => {
  const sr = document.getElementById('everyric-root').shadowRoot;
  const headerHasReset = !!sr.querySelector('.ey-actions .ey-reset-sync-btn');
  const footer = sr.querySelector('.ey-footer');
  const footerBtn = footer?.querySelector('.ey-reset-sync-btn');
  return {
    headerHasReset,
    footerBtnPresent: !!footerBtn,
    footerBtnVisible: footerBtn ? footerBtn.getClientRects().length > 0 : false,
    title: footerBtn?.title ?? '',
  };
});
check(resetBtnProbe.headerHasReset === false && resetBtnProbe.footerBtnPresent === true
  && resetBtnProbe.footerBtnVisible === true && /초기화/.test(resetBtnProbe.title)
  && !/다시 생성|재생성/.test(resetBtnProbe.title),
  'R5 풋터: 싱크 초기화 버튼 존재(헤더엔 없음)·재생성 문구 부재', resetBtnProbe);

// 설정 시트 닫기 (이후 체크가 시트에 가리지 않게)
await page.evaluate(() => {
  const sr = document.getElementById('everyric-root').shadowRoot;
  const gear = [...sr.querySelectorAll('.ey-actions button')].find(b => /설정|settings/i.test(b.title || ''));
  gear?.click();
});
await page.waitForTimeout(800);

// ── R6: 부착 레인 — 패널 왼쪽에 별도 부착 패널로 렌더 ────────────
await sw.evaluate(async () => {
  const cur = (await chrome.storage.local.get('settings')).settings ?? {};
  await chrome.storage.local.set({
    settings: { ...cur, modMainLane: true, mainLanePos: 'attached' },
  });
});
await page.waitForTimeout(2500);
const attach = await page.evaluate(() => {
  const sr = document.getElementById('everyric-root').shadowRoot;
  const panel = sr.querySelector('.ey-panel');
  const att = sr.querySelector('.ey-attach-lane');
  if (!att || !panel) return { present: !!att };
  const A = att.getBoundingClientRect(), P = panel.getBoundingClientRect();
  const canvas = att.querySelector('canvas');
  let drawnPx = 0;
  if (canvas) {
    try {
      const data = canvas.getContext('2d').getImageData(0, 0, canvas.width || 1, canvas.height || 1).data;
      for (let i = 3; i < data.length; i += 4) if (data[i] > 0) drawnPx++;
    } catch { /* ignore */ }
  }
  return {
    present: true,
    visible: att.getClientRects().length > 0,
    leftOfPanel: A.right <= P.x + 8,
    isSibling: att.parentNode === panel.parentNode,
    heightFollows: Math.abs(A.height - P.height) < 40,
    drawnPx,
  };
});
check(attach.present === true && attach.visible === true && attach.leftOfPanel === true
  && attach.isSibling === true && attach.drawnPx > 50,
  'R6 부착 레인: 패널 밖 왼쪽 형제 패널 + 노트 픽셀 렌더', attach);

// ── R7: PIP 가사 목록 컬럼 ───────────────────────────────────────
await sw.evaluate(async () => {
  const cur = (await chrome.storage.local.get('settings')).settings ?? {};
  await chrome.storage.local.set({ settings: { ...cur, pipLyricsList: true } });
});
await page.waitForTimeout(1200);
await page.evaluate(() => {
  const v = document.querySelector('video.html5-main-video') ?? document.querySelector('video');
  if (v) { v.currentTime = Math.min(25, (v.duration || 40) * 0.3); void v.play().catch(() => {}); }
});
const pipBtn = page.locator('#everyric-root [title*="PiP"], #everyric-root [title*="PIP"]').first();
let pipOpened = false;
try {
  await pipBtn.click({ timeout: 5000 });
  await page.waitForTimeout(3500);
  pipOpened = await page.evaluate(() => !!window.documentPictureInPicture?.window);
} catch { /* 버튼 미발견 → 아래 skip */ }
if (!pipOpened) {
  skip('R7/R8 PIP 검사', 'PIP 창을 열지 못함');
} else {
  const pipCol = await page.evaluate(() => {
    const w = window.documentPictureInPicture.window;
    // 2026-08-04 재작업: 오른쪽 전용 가사 컬럼(.ey-pip-lyricscol)과 설정 pipLyricsList는
    // 사라졌다 — PiP 창 안이 통째로 메인 가사창과 같은 패널이 되면서 «본문이 곧 가사
    // 목록»이 됐기 때문이다. 이 검사의 의도(간이 렌더 금지, 메인과 같은 .ey-line 구조)는
    // 그대로 살아 있으므로 대상만 패널 본문으로 바꾼다.
    const root = w.document.getElementById('everyric-root')?.shadowRoot;
    const col = root?.querySelector('.ey-body');
    if (!col) return { present: false };
    const lines = col.querySelectorAll('.ey-line').length;
    const pronLines = col.querySelectorAll('.ey-line-pron').length;
    const karaokeSpans = col.querySelectorAll('.ey-word, .ey-pron-syl').length;
    return {
      present: true,
      visible: w.getComputedStyle(col).display !== 'none' && col.getClientRects().length > 0,
      items: col.children.length,
      lines, pronLines, karaokeSpans,
    };
  });
  check(pipCol.present === true && pipCol.visible === true && pipCol.lines > 3
    && pipCol.pronLines > 0 && pipCol.karaokeSpans > 10,
    'R7 PIP 가사 목록이 메인과 동일한 .ey-line 구조(발음 줄·카라오케 스팬 포함)', pipCol);

  // ── R8: PIP ↑↓ 볼륨 ────────────────────────────────────────────
  const volBefore = await page.evaluate(() => {
    const v = document.querySelector('video.html5-main-video') ?? document.querySelector('video');
    if (v) v.volume = 0.5;
    return v?.volume ?? null;
  });
  await page.evaluate(() => {
    const w = window.documentPictureInPicture.window;
    // 핸들러는 e.code(물리 키)를 본다 — key만 넣은 합성 이벤트는 code가 빈 문자열이라 무시된다
    w.document.body.dispatchEvent(new w.KeyboardEvent('keydown', { key: 'ArrowUp', code: 'ArrowUp', bubbles: true }));
    w.document.body.dispatchEvent(new w.KeyboardEvent('keydown', { key: 'ArrowUp', code: 'ArrowUp', bubbles: true }));
  });
  await page.waitForTimeout(800);
  const volAfter = await page.evaluate(() => {
    const v = document.querySelector('video.html5-main-video') ?? document.querySelector('video');
    return v?.volume ?? null;
  });
  check(volBefore !== null && volAfter !== null && volAfter > volBefore + 0.05,
    'R8 PIP ↑↑ 두 번으로 볼륨 +0.1', { volBefore, volAfter });
}

await page.screenshot({ path: 'feedback-round2-check.png' }).catch(() => {});
await ctx.close();
console.log(failed ? '\n== 결과: FAIL 있음' : '\n== 결과: 전부 PASS');
process.exit(failed ? 1 : 0);
