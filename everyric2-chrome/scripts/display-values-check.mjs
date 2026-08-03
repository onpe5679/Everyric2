// 렌더 값 검증 E2E — 응답 JSON이 아니라 **화면에 실제로 그려진 DOM**을 읽어
// 신 스택(발음 표기·카라오케 스팬·피아노롤)이 사용자 눈에 보이는 값 그대로
// 올바른지 확인한다. 감사 배경: JSON 레벨 검증은 서버가 옳은 값을 줬는지는
// 보증하지만, 그 값이 화면까지 손상 없이 도달했는지는 보증하지 못한다
// (예: pron dict가 있어도 buildPronEl이 안 불리면 화면은 그대로다).
//
// 사전 조건: 로컬 서버(:8000)에 이 videoId의 everyric 싱크가 **이미 있어야** 한다.
// 이 스크립트는 싱크를 생성하지 않는다 — 없으면 그 사실만 보고하고 끝낸다
// (팀 지시: "싱크가 없으면 명확히 그 사실을 알리고 종료").
//
// 실행: node scripts/display-values-check.mjs <videoId> [expectedLang]
//   videoId: 필수, 하드코딩 금지 — 명령행 인자로만 받는다.
//   expectedLang: 선택, 로그 라벨용 힌트일 뿐 어서션 조건을 바꾸지 않는다.
//
// 주소는 반드시 127.0.0.1 — 이 개발 머신은 localhost가 IPv6 폴백 스톨로 요청당 2초.
import { chromium } from 'playwright';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import { cpSync, mkdtempSync, readFileSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { ensureLocalServerPermissionForServerUrl } from './lib/local-server-permission.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
// dist를 직접 로드하지 않고 **스냅샷 사본**을 로드한다 — 다른 세션/에이전트가 같은
// dist에 `npm run build`를 돌리는 순간과 겹치면 manifest.json이 갈리는 중이라 크롬이
// "매니페스트 없음" 모달을 띄우고, 그 모달이 Playwright 연결을 영영 막는다
// (실측 2026-08-03: launchPersistentContext 180s 타임아웃 2회). 스냅샷 뒤 manifest를
// 파싱해 온전한 빌드인지 확인하고 시작한다.
const liveDist = resolve(__dirname, '../dist');
const distDir = mkdtempSync(join(tmpdir(), 'everyric-dist-snap-'));
cpSync(liveDist, distDir, { recursive: true });
JSON.parse(readFileSync(join(distDir, 'manifest.json'), 'utf8')); // 깨진 스냅샷이면 여기서 즉사
const videoId = process.argv[2];
const expectedLang = process.argv[3] ?? '(미지정)';
const LOCAL_SERVER_URL = 'http://127.0.0.1:8000';
const SYNC_WAIT_MS = 25000; // 생성이 아니라 조회이므로 짧게 — 이미 DB에 있어야 하는 전제

if (!videoId) {
  console.log('사용법: node scripts/display-values-check.mjs <videoId> [expectedLang]');
  console.log('videoId는 명령행 인자로만 받습니다(하드코딩 금지) — 예: node scripts/display-values-check.mjs abcXYZ123 ja');
  process.exit(2);
}

let failed = false;
function check(ok, label, detail) {
  console.log(`${ok ? 'PASS' : 'FAIL'}: ${label}${detail !== undefined ? ' = ' + JSON.stringify(detail) : ''}`);
  if (!ok) failed = true;
  return ok;
}
function info(label, detail) {
  console.log(`INFO: ${label}${detail !== undefined ? ' = ' + JSON.stringify(detail) : ''}`);
}

// 0) 로컬 서버 확인 — 안 떠 있으면 여기서 뜨우지 않고 사실만 보고하고 종료(하네스 책임 아님)
try {
  const health = await (await fetch(`${LOCAL_SERVER_URL}/health`, { signal: AbortSignal.timeout(3000) })).json();
  if (!check(health.status === 'healthy', '로컬 서버(127.0.0.1:8000) /health', health)) process.exit(2);
} catch (e) {
  console.log('DATA: 로컬 서버가 127.0.0.1:8000에서 응답하지 않습니다 —', String(e).slice(0, 160));
  console.log('DATA: 서버 기동은 이 스크립트의 책임이 아닙니다. 서버를 띄운 뒤 다시 실행하세요.');
  process.exit(2);
}

const userDataDir = mkdtempSync(join(tmpdir(), 'everyric-display-check-'));
const channel = process.env.EVERYRIC_E2E_CHANNEL ?? '';
const ctx = await chromium.launchPersistentContext(userDataDir, {
  ignoreDefaultArgs: ['--disable-extensions'],
  ...(channel ? { channel } : {}),
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

const SR = `document.getElementById('everyric-root')?.shadowRoot`;

try {
  const sw = ctx.serviceWorkers()[0] ?? await ctx.waitForEvent('serviceworker', { timeout: 15000 });
  console.log('extension loaded:', sw.url());
  const extId = new URL(sw.url()).host;
  await ensureLocalServerPermissionForServerUrl(ctx, sw, extId, LOCAL_SERVER_URL);
  // uiLanguage를 명시적으로 ko로 고정 — 이 스크립트의 선택자(라벨 텍스트)가 한국어
  // 카탈로그 문자열에 의존하므로, 로케일이 auto로 다르게 풀리면 선택자가 깨진다.
  await sw.evaluate(s => chrome.storage.local.set({ settings: s }), {
    debugInfo: true,
    serverUrl: LOCAL_SERVER_URL,
    uiLanguage: 'ko',
    pronunciationScript: 'hangul',
    showPronunciation: true,
    showTranslation: true,
  });

  const page = ctx.pages()[0] ?? await ctx.newPage();
  page.on('console', msg => {
    if (msg.type() === 'error') console.log('[console:error]', msg.text().slice(0, 300));
  });

  const videoUrl = `https://www.youtube.com/watch?v=${videoId}`;
  console.log(`navigating to ${videoUrl} (expectedLang=${expectedLang})`);
  await page.goto(videoUrl, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForSelector('#everyric-root', { state: 'attached', timeout: 30000 });

  // 1) 싱크 존재 확인 — 조회만 한다(생성 트리거 안 함). 없으면 데이터 사실로 보고하고 종료.
  let synced;
  try {
    synced = await page.waitForFunction(() => {
      const root = document.getElementById('everyric-root')?.shadowRoot;
      if (!root) return null;
      const dbg = root.querySelector('.ey-debug')?.textContent ?? '';
      const lines = root.querySelectorAll('.ey-line:not(.ey-line-plain)').length;
      if (/src=everyric/.test(dbg) && lines > 0) return { lines, dbg: dbg.slice(0, 160) };
      const st = root.querySelector('.ey-state-text')?.textContent ?? '';
      if (st && !/검색/.test(st)) return { noSync: true, stateText: st };
      const plain = root.querySelectorAll('.ey-line-plain').length;
      if (plain > 0) return { noSync: true, stateText: `가사는 있으나 싱크 없음(plain=${plain})` };
      return null;
    }, null, { timeout: SYNC_WAIT_MS, polling: 1000 }).then(h => h.jsonValue());
  } catch {
    console.log(`DATA: videoId=${videoId} — ${SYNC_WAIT_MS / 1000}초 안에 검색 상태가 확정되지 않음(서버가 느리거나 응답 없음)`);
    process.exitCode = 2;
    throw new Error('SYNC_LOOKUP_TIMEOUT');
  }
  if (synced.noSync) {
    console.log(`DATA: videoId=${videoId}에 대해 로컬 DB에 everyric 싱크가 없습니다 — ${synced.stateText}`);
    console.log('DATA: 이 스크립트는 싱크를 생성하지 않습니다(팀 지시). 다른 videoId로 다시 실행하거나, 먼저 싱크를 생성하세요.');
    process.exitCode = 2;
    throw new Error('NO_SYNC_FOR_VIDEO');
  }
  info('everyric 싱크 확인됨', synced);

  // ── 체크 1: 발음 라인이 원문과 달라야 한다(한자가 그대로 남아 있으면 FAIL) ──
  // 원문에 한자(CJK 통합 한자)가 있는 줄만 대상 — 없는 줄(순수 가나/영어/한글)은 이 체크의
  // 대상이 아니므로 examined 카운트에서 제외한다(가짜 PASS를 막기 위해 examined=0이면
  // 별도로 보고한다).
  const kanjiCheck = await page.evaluate(() => {
    const root = document.getElementById('everyric-root')?.shadowRoot;
    const KANJI = /[一-鿿]/;
    const lines = Array.from(root.querySelectorAll('.ey-line'));
    let examined = 0;
    const offenders = [];
    for (const el of lines) {
      const clone = el.cloneNode(true);
      clone.querySelector('.ey-line-pron')?.remove();
      clone.querySelector('.ey-line-tr')?.remove();
      const original = clone.textContent.trim();
      if (!KANJI.test(original)) continue;
      examined++;
      const pronEl = el.querySelector('.ey-line-pron');
      const pron = pronEl ? pronEl.textContent.trim() : null;
      const stillHasKanji = pron !== null && KANJI.test(pron);
      const unchanged = pron === original;
      if (pron === null || stillHasKanji || unchanged) {
        offenders.push({ original: original.slice(0, 40), pron: pron?.slice(0, 40) ?? null, stillHasKanji, unchanged });
      }
    }
    return { examined, offenderCount: offenders.length, offenders: offenders.slice(0, 5) };
  });
  if (kanjiCheck.examined === 0) {
    console.log('WARN: 체크1 미실행 — 렌더된 줄 중 한자를 포함한 줄이 없음(이 곡이 ja가 아니거나 한자 없는 가사). 검사 대상 0줄.');
  } else {
    check(kanjiCheck.offenderCount === 0,
      `발음 표기가 원문 한자를 변환함 (검사 대상 ${kanjiCheck.examined}줄)`, kanjiCheck);
  }

  // ── 체크 2: hangul → romaji 전환 시 재생성 없이 화면 텍스트가 실제로 바뀐다 ──
  const baseline = await page.evaluate(() => Array.from(
    document.getElementById('everyric-root').shadowRoot.querySelectorAll('.ey-line'),
  ).map(el => el.querySelector('.ey-line-pron')?.textContent?.trim() ?? null));

  // 설정 시트를 열고(gear=설정) 발음 표기 방식 select를 실제 유저 조작처럼 바꾼다 —
  // storage를 직접 쓰면 handleSettingsChange 경로(콜백)를 안 타서 반영이 안 된다
  // (content.ts는 storage.onChanged를 job 키만 듣는다 — 설정 변경은 UI 콜백 전용 경로).
  await page.locator('#everyric-root [title="설정"]').first().click();
  // 8범주 접이식 — 접힌 범주 안 select는 hidden이라 먼저 전부 펼친다
  await page.waitForTimeout(600);
  await page.evaluate(() => {
    const sr = document.getElementById('everyric-root').shadowRoot;
    for (const d of sr.querySelectorAll('.ey-settings details')) d.open = true;
  });
  const pronScriptSelect = page.locator('#everyric-root .ey-settings-row', { has: page.locator('label', { hasText: '발음 표기 방식' }) }).locator('select');
  await pronScriptSelect.waitFor({ state: 'visible', timeout: 5000 });
  const beforeValue = await pronScriptSelect.inputValue();
  await pronScriptSelect.selectOption('romaji');
  await page.waitForTimeout(500); // refreshTranslations는 동기 DOM 재구성 — 여유만 둔다

  const afterRomaji = await page.evaluate(() => Array.from(
    document.getElementById('everyric-root').shadowRoot.querySelectorAll('.ey-line'),
  ).map(el => el.querySelector('.ey-line-pron')?.textContent?.trim() ?? null));

  // NETWORK 요청 없이 바뀌었는지도 확인 — 전환 자체가 재생성/검색 API를 부르면 안 된다
  let generateOrSearchCalled = false;
  const netListener = req => {
    const u = req.url();
    if (/\/api\/(sync|generate|search)/.test(u)) generateOrSearchCalled = true;
  };
  page.on('request', netListener);

  const changed = baseline.map((b, i) => ({ i, before: b, after: afterRomaji[i] })).filter(x => x.before !== x.after);
  const changedToNonEmpty = changed.filter(x => x.after !== null && x.after !== '');
  const changedToEmpty = changed.filter(x => x.after === null || x.after === '');
  info('발음 표기 전환 결과', {
    beforeValue, totalLines: baseline.length,
    changedCount: changed.length, changedToNonEmpty: changedToNonEmpty.length, changedToEmpty: changedToEmpty.length,
    sample: changed.slice(0, 3),
  });
  check(changed.length > 0, 'hangul→romaji 전환 시 최소 1줄 이상 화면 텍스트 변경(재생성 없이)');
  if (changedToNonEmpty.length === 0 && changed.length > 0) {
    console.log('WARN: 전환된 줄이 전부 "표시 안 함"(빈 값)으로만 바뀜 — romaji pron dict 데이터 자체가 없을 가능성(코드 결함이 아니라 데이터 부재일 수 있음)');
  }
  await page.waitForTimeout(300);
  page.off('request', netListener);
  check(!generateOrSearchCalled, '표기 전환이 재생성/검색 API를 호출하지 않음(순수 로컬 재렌더)');

  // 원복(다음 체크에 영향 주지 않도록)
  await pronScriptSelect.selectOption(beforeValue || 'hangul');
  await page.waitForTimeout(300);
  await page.locator('#everyric-root [title="설정"]').first().click();

  // ── 체크 3: 음절 가라오케 스팬이 라인을 덮는다(사라지는 스팬 없음) ──
  const spanCheck = await page.evaluate(() => {
    const root = document.getElementById('everyric-root')?.shadowRoot;
    const spans = Array.from(root.querySelectorAll('.ey-line .ey-word, .ey-line .ey-pron-syl'));
    let nonWhitespace = 0;
    const vanished = [];
    for (const el of spans) {
      const text = el.textContent ?? '';
      if (text.trim() === '') continue;
      nonWhitespace++;
      const rect = el.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) {
        vanished.push({ text: text.slice(0, 20), className: el.className, width: rect.width, height: rect.height });
      }
    }
    return { totalSpans: spans.length, nonWhitespace, vanishedCount: vanished.length, vanished: vanished.slice(0, 5) };
  });
  check(spanCheck.nonWhitespace > 0, `카라오케 스팬 렌더링됨 (${spanCheck.nonWhitespace}개, 공백 제외)`);
  check(spanCheck.vanishedCount === 0, '텍스트가 있는데 픽셀 폭/높이가 0인 스팬 없음(사라지는 스팬)', spanCheck);
  await page.screenshot({ path: resolve(__dirname, '../display-values-check-panel.png') });
  console.log('screenshot: display-values-check-panel.png');

  // ── 체크 4: 피아노롤(PiP 음정 캔버스)에 노트 픽셀이 실제로 그려진다 ──
  await page.evaluate(() => {
    const v = document.querySelector('video.html5-main-video') ?? document.querySelector('video');
    if (v) { v.currentTime = Math.min(20, (v.duration || 40) * 0.3); void v.play().catch(() => {}); }
  });
  await page.waitForTimeout(1000);
  await page.locator('#everyric-root [title="PiP 창으로 보기"]').first().click();
  await page.waitForTimeout(3000);
  const pitch = await page.evaluate(() => {
    const w = window.documentPictureInPicture?.window;
    if (!w) return { open: false };
    // PiP 창 안의 가사 UI는 **메인 가사창과 같은 인스턴스**다(2026-08-04 재작업) —
    // 예전의 반쪽 캔버스(.ey-pip-pitch)는 사라졌고, 레인은 패널의 Shadow DOM 안
    // .ey-main-lane 하나뿐이다. 두 창이 같은 셀렉터로 검사된다는 것 자체가 «구현이
    // 하나»라는 이 재작업의 목적을 그대로 재는 지표다.
    const host = w.document.getElementById('everyric-root');
    if (!host?.shadowRoot) return { open: true, panelPresent: false };
    const root = host.shadowRoot;
    // 레인은 Shadow DOM 밖(왼쪽 열)이다 — 3열 구조에서 부착 패널이 문서로 나간다
    const c = w.document.querySelector('.ey-main-lane') ?? root.querySelector('.ey-main-lane');
    if (!c) return { open: true, panelPresent: true, canvasPresent: false };
    const visible = w.getComputedStyle(c.closest('.ey-lane-wrap') ?? c).display !== 'none';
    let drawnPx = 0;
    try {
      const data = c.getContext('2d').getImageData(0, 0, c.width || 1, c.height || 1).data;
      for (let i = 3; i < data.length; i += 4) if (data[i] > 0) drawnPx++;
    } catch { /* ignore */ }
    return {
      open: true, panelPresent: true, canvasPresent: true, visible,
      width: c.width, height: c.height, drawnPx,
      // «메인 창에만 있던» 것들이 실제로 PiP 안에도 그려졌는가 (운영자 지적 항목)
      lineCount: root.querySelectorAll('.ey-line').length,
      hasSearchBtn: !!root.querySelector('.ey-header .ey-btn[title*="검색"]'),
      hasGearBtn: !!root.querySelector('.ey-header .ey-btn[title*="설정"]'),
      hasOffsetRow: !!root.querySelector('.ey-offset'),
      hasQuickRow: !!root.querySelector('.ey-quick-row'),
      filled: !!root.querySelector('.ey-panel.ey-panel-filled'),
      // 반쪽 구현이 정말로 사라졌는지 — 남아 있으면 이중 표시가 된다.
      // .ey-pip-stage(영상 아래 한 줄)는 유지가 지시된 화면이라 제외하고, 대신 그 안이
      // 공용 .ey-line인지를 본다(그게 「세 번째 구현 금지」의 진짜 불변식이다).
      legacyStage: !!w.document.querySelector('.ey-pip-pitch, .ey-pip-panel, .ey-pip-lyricscol'),
      shortUsesSharedLine: !!w.document.querySelector('.ey-pip-stage .ey-pip-line.current .ey-line'),
    };
  });
  check(pitch.open, 'PiP 창 열림', pitch);
  if (pitch.open) {
    check(pitch.panelPresent, 'PiP 안에 가사 패널 인스턴스가 세워짐', pitch);
    check(pitch.filled === true, 'PiP 패널이 filled 크롬으로 렌더됨(.ey-panel-filled)', pitch);
    check(pitch.legacyStage === false, '예전 반쪽 PiP UI(캔버스·패널·목록 컬럼)가 남아 있지 않음', pitch);
    check(pitch.shortUsesSharedLine === true, '가사 단축 표시가 공용 줄 렌더러(.ey-line)를 씀', pitch);
    check(pitch.lineCount > 0, `PiP 안에 가사 줄이 그려짐 (${pitch.lineCount}줄)`);
    check(Boolean(pitch.hasSearchBtn && pitch.hasGearBtn), 'PiP 헤더에 검색·설정 버튼 존재', pitch);
    check(Boolean(pitch.hasOffsetRow && pitch.hasQuickRow), 'PiP에 오프셋 줄·퀵 토글 줄 존재', pitch);
    check(pitch.canvasPresent && pitch.visible, '피아노롤 캔버스 존재 + 표시됨', pitch);
    check(pitch.canvasPresent && pitch.drawnPx > 50, '피아노롤에 노트 픽셀이 실제로 그려짐', pitch);
  }
  try {
    const pipPage = ctx.pages().find(p => p !== page);
    if (pipPage) await pipPage.screenshot({ path: resolve(__dirname, '../display-values-check-pip.png') });
  } catch { /* ignore */ }

  console.log(failed ? 'DISPLAY VALUES CHECK: FAIL' : 'DISPLAY VALUES CHECK: PASS');
  process.exitCode = failed ? 1 : 0;
} catch (e) {
  if (!['NO_SYNC_FOR_VIDEO', 'SYNC_LOOKUP_TIMEOUT'].includes(e.message)) {
    console.log('DISPLAY VALUES CHECK: ERROR —', String(e).slice(0, 500));
    process.exitCode = 1;
  }
  try {
    const page = ctx.pages()[0];
    if (page) await page.screenshot({ path: resolve(__dirname, '../display-values-check-error.png') });
  } catch { /* ignore */ }
} finally {
  await ctx.close();
}
