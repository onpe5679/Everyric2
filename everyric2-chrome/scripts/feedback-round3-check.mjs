// 2026-08-04 3차 피드백 수리분 실브라우저 검증 — "코드가 있다"가 아니라 "화면에 그려진다".
//
//   R1) 재생목록 부착 패널의 다음 영상 폴백 카드가 클릭 가능하고(cursor:pointer) 클릭하면
//       실제로 이동 트리거(다음 버튼 클릭 또는 location 변화)가 일어난다
//   R2) list= URL로 들어가 초기 스크랩이 빈 손이어도 "재생목록에 속하지 않아요"가
//       즉시 뜨지 않는다(백오프 재시도 중엔 이전 상태를 유지)
//   R3) fast 라우팅 + 저신뢰(qualityScore<0.001) 싱크에서 경고 배너가 느낌표·노란색 없이
//       중립 톤(.ey-warn-neutral)으로 뜬다
//   R4) en 곡 × romaji 표기에서 원문과 똑같은 발음 줄(중복)이 안 뜬다
//   R5) 설정 ipa 선택 시 ja 곡에서 발음 줄이 통째로 비지 않는다(hangul 폴백)
//   R6) 서버 싱크(everyric) 곡의 출처 표시가 "Everyric"으로 시작하지 않는다
//
// 실행: node scripts/feedback-round3-check.mjs
// 사전 조건: 실서버 127.0.0.1:8000(localhost는 IPv6 스톨), dist는 이번 라운드 수리분 포함 빌드,
//   everyric2.db(프로젝트 루트)에 R3~R6용 실곡 싱크가 이미 있어야 한다(생성 안 함, 조회만).
//
// **이번 실행 시점 주의(팀리드 지시)**: 다른 에이전트가 src/를 대개편 중이라 지금 dist는
// 그 이전 빌드다 — 이번 라운드는 하네스 완성 + 각 검사 함수의 셀렉터가 실제 DOM과
// 맞는지까지만 확인한다. 대개편이 커밋되고 새 dist가 나오면 그때 전체 PASS 판정을 다시 돌린다
// (그때 PIP 대칭 검사도 추가될 수 있어 R1~R6는 함수 단위로 분리해 확장 여지를 남겼다).
import { chromium } from 'playwright';
import { fileURLToPath } from 'url';
import { dirname, resolve, join } from 'path';
import { mkdtempSync, cpSync } from 'fs';
import { tmpdir } from 'os';
import { execFileSync } from 'child_process';
import { DatabaseSync } from 'node:sqlite';
import { ensureLocalServerPermissionForServerUrl } from './lib/local-server-permission.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const distSrc = resolve(__dirname, '../dist');
const distDir = join(mkdtempSync(join(tmpdir(), 'ey-dist-')), 'dist');
cpSync(distSrc, distDir, { recursive: true });

const SERVER = 'http://127.0.0.1:8000';
const DB_PATH = resolve(__dirname, '../../everyric2.db');
// 스크린샷은 리포에 안 남긴다(팀리드 지시) — 스크래치패드로
const SHOT_DIR = process.env.EY_SHOT_DIR
  ?? 'C:\\Users\\user\\AppData\\Local\\Temp\\claude\\C--DevAT-Everyric2\\ccf2cc19-4a2e-4cc6-9f6d-e92f242a5aa3\\scratchpad';

let failed = false;
function check(ok, label, detail) {
  console.log(`${ok ? 'PASS' : 'FAIL'}: ${label}${detail !== undefined ? ' = ' + JSON.stringify(detail) : ''}`);
  if (!ok) failed = true;
  return ok;
}
function skip(label, why) { console.log(`SKIP: ${label} — ${why}`); }
function info(label, detail) { console.log(`INFO: ${label} = ${JSON.stringify(detail)}`); }

// ── 실곡 선택 — everyric2.db 조회(생성 아님). 조건: R3=fast 라우팅+qualityScore<0.001,
//    R4=en 곡, R5=ja 곡, R6=attribution 있는 everyric 싱크. 근거는 REPORT 참고. ──
function pickSongs() {
  const db = new DatabaseSync(DB_PATH, { readOnly: true });
  try {
    const rows = db.prepare(
      'SELECT video_id, language, title, quality_score, timestamps FROM sync_results ORDER BY id DESC',
    ).all();
    const parsed = rows.map(r => {
      let route = null;
      try { route = JSON.parse(r.timestamps).debug?.routing?.route ?? null; } catch { /* 무시 */ }
      return { videoId: r.video_id, language: r.language, qs: r.quality_score, route, title: r.title };
    });
    const r3 = parsed.find(r => r.route === 'fast' && r.qs != null && r.qs < 0.001) ?? null;
    const r4 = parsed.find(r => r.language === 'en') ?? null;
    const r5 = parsed.find(r => r.language === 'ja') ?? null;
    const r6 = r3; // r3 후보(arX83q0oJhM류)는 attribution도 이미 있다(보카로 위키 채택곡)
    return { r3, r4, r5, r6 };
  } finally {
    db.close();
  }
}

const songs = pickSongs();
info('R3용 실곡(fast+qs<0.001)', songs.r3);
info('R4용 실곡(en)', songs.r4);
info('R5용 실곡(ja)', songs.r5);

const health = await (await fetch(`${SERVER}/health`, { signal: AbortSignal.timeout(3000) })).json().catch(() => null);
if (!check(health?.status === 'healthy', 'real server /health', health)) process.exit(1);

// ── 실행 전 정리 — 팀리드 지시(다른 세션 잔류 크롬이 프로필/포트를 물고 있는 사고 방지).
//    이 스크립트는 매번 새 userDataDir(mkdtempSync)를 쓰므로 프로필 충돌은 원래 없지만,
//    좀비 프로세스가 CPU/메모리를 눌러 타임아웃을 유발하는 사고를 막기 위해 best-effort로 정리한다.
//    실패(권한 없음·프로세스 없음)는 무시 — 이 스크립트의 성패를 좌우할 이유가 없다. ──
try {
  execFileSync('taskkill', ['/F', '/IM', 'chrome.exe', '/T'], { stdio: 'ignore' });
  info('사전 정리', 'chrome.exe 잔류 프로세스 종료(taskkill)');
} catch {
  info('사전 정리', '잔류 chrome.exe 없음(또는 종료 권한 없음) — 정상, 계속 진행');
}

const userDataDir = mkdtempSync(join(tmpdir(), 'ey-r3-'));
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

async function setSettings(patch) {
  await sw.evaluate(async (p) => {
    const cur = (await chrome.storage.local.get('settings')).settings ?? {};
    await chrome.storage.local.set({ settings: { ...cur, ...p } });
  }, patch);
}

// serverUrl 기본값은 원격 프로덕션 서버다 — 로컬 실곡(everyric2.db 조회분)을 실제로
// 보려면 세션 전체에 걸쳐 이걸 127.0.0.1:8000으로 고정해야 한다(권한 부여와는 별개 —
// 권한은 "호출 가능"만 보장하고 "이 서버를 쓴다"는 설정을 따로 켜야 한다). 이걸 빼먹으면
// 원격 프로덕션 서버로 조용히 나가 구조적 검사(R4~R6 등)는 우연히 통과해도, DB에서
// 정확한 quality_score로 골라온 R3 같은 값 의존 검사만 조용히 어긋난다(실측 — 최초
// 작성 시 이 줄을 빼먹어 R3만 FAIL했었다).
await setSettings({ serverUrl: SERVER });

async function gotoAndWaitSynced(page, videoId, extraParams = '') {
  await page.goto(`https://www.youtube.com/watch?v=${videoId}${extraParams}`, {
    waitUntil: 'domcontentloaded', timeout: 90000,
  });
  await page.waitForFunction(
    () => !!document.getElementById('everyric-root')?.shadowRoot?.querySelector('.ey-line'),
    { timeout: 120000 },
  );
  // .ey-line 등장은 showSyncedLyrics 완료 시점일 뿐 — setQualityWarning 등 뒤이은 부수
  // 반영(경고 바 등)까지의 짧은 지연을 흡수한다(실측: 2.5s는 가끔 부족, 3.5s는 안정적)
  await page.waitForTimeout(3500);
}

// ══════════════════════════════════════════════════════════════════
// R3: 경고 배너 중립 톤 (fast 라우팅 + 저신뢰)
// ══════════════════════════════════════════════════════════════════
async function checkR3(page) {
  if (!songs.r3) return skip('R3 경고 배너 중립 톤', 'DB에 fast+qualityScore<0.001 실곡 없음');
  await setSettings({ lowConfWarning: true, uiLanguage: 'ko' });
  await gotoAndWaitSynced(page, songs.r3.videoId);
  const warn = await page.evaluate(() => {
    const sr = document.getElementById('everyric-root').shadowRoot;
    const bar = sr.querySelector('.ey-warn-bar');
    if (!bar || bar.style.display === 'none') return { present: false };
    const text = sr.querySelector('.ey-warn-text')?.textContent ?? '';
    return {
      present: true,
      neutral: bar.classList.contains('ey-warn-neutral'),
      hasExclamation: text.includes('⚠'),
      text,
    };
  });
  check(warn.present === true && warn.neutral === true && warn.hasExclamation === false,
    `R3 fast 싱크(${songs.r3.videoId}, qs=${songs.r3.qs}) 경고 배너가 중립 톤(⚠ 없음, .ey-warn-neutral)`, warn);
}

// ══════════════════════════════════════════════════════════════════
// R4: en 곡 원문 중복 발음 줄 숨김
// ══════════════════════════════════════════════════════════════════
async function checkR4(page) {
  if (!songs.r4) return skip('R4 en 중복 발음 숨김', 'DB에 en 곡 없음');
  // pronunciationScript=auto + translationLanguage=en → resolveScript가 romaji를 고른다
  // (lib/lang.ts resolveScript) — en 곡의 romaji는 원문 철자 그대로라 이 조합이 재현 조건이다.
  await setSettings({
    translationLanguage: 'en', pronunciationScript: 'auto',
    showPronunciation: true, hidePronForEnglish: false, uiLanguage: 'ko',
  });
  await gotoAndWaitSynced(page, songs.r4.videoId);
  const probe = await page.evaluate(() => {
    const sr = document.getElementById('everyric-root').shadowRoot;
    const lines = Array.from(sr.querySelectorAll('.ey-line'));
    let dupCount = 0;
    const samples = [];
    for (const el of lines) {
      const pronEl = el.querySelector('.ey-line-pron');
      if (!pronEl) continue;
      const clone = el.cloneNode(true);
      clone.querySelector('.ey-line-pron')?.remove();
      clone.querySelector('.ey-line-tr')?.remove();
      const original = clone.textContent.trim();
      const pron = pronEl.textContent.trim();
      const norm = s => s.normalize('NFKC').toLowerCase().replace(/[\s\p{P}]+/gu, '');
      if (norm(original) === norm(pron)) {
        dupCount++;
        if (samples.length < 3) samples.push({ original: original.slice(0, 40), pron: pron.slice(0, 40) });
      }
    }
    return { totalLines: lines.length, dupCount, samples };
  });
  check(probe.totalLines > 0 && probe.dupCount === 0,
    `R4 en 곡(${songs.r4.videoId}) 원문=발음(정규화 후 동일)인 중복 줄 0건`, probe);
}

// ══════════════════════════════════════════════════════════════════
// R5: IPA 표기 선택 시 ja 곡 발음 줄 전멸 방지(hangul 폴백)
// ══════════════════════════════════════════════════════════════════
async function checkR5(page) {
  if (!songs.r5) return skip('R5 IPA 폴백', 'DB에 ja 곡 없음');
  await setSettings({
    pronunciationScript: 'ipa', showPronunciation: true, hidePronForEnglish: false, uiLanguage: 'ko',
  });
  await gotoAndWaitSynced(page, songs.r5.videoId);
  const probe = await page.evaluate(() => {
    const sr = document.getElementById('everyric-root').shadowRoot;
    const lines = Array.from(sr.querySelectorAll('.ey-line'));
    const pronLines = lines.filter(el => {
      const p = el.querySelector('.ey-line-pron');
      return p && p.textContent.trim().length > 0;
    }).length;
    // 설정 select에 ipa 옵션 자체가 살아있는지도 함께 확인(overlay.ts 옵션 행 복원)
    return { totalLines: lines.length, pronLines };
  });
  const optionProbe = await page.evaluate(() => {
    const sr = document.getElementById('everyric-root').shadowRoot;
    const gear = [...sr.querySelectorAll('.ey-actions button')].find(b => /설정|settings/i.test(b.title || ''));
    gear?.click();
    return true;
  }).then(async () => {
    await page.waitForTimeout(800);
    return page.evaluate(() => {
      const sr = document.getElementById('everyric-root').shadowRoot;
      const sheet = sr.querySelector('.ey-settings');
      for (const d of sheet?.querySelectorAll('details') ?? []) d.open = true;
      const select = [...(sheet?.querySelectorAll('select') ?? [])]
        .find(s => [...s.options].some(o => o.value === 'ipa'));
      return { selectFound: !!select, hasIpaOption: !!select && [...select.options].some(o => o.value === 'ipa') };
    });
  });
  await page.evaluate(() => {
    const sr = document.getElementById('everyric-root').shadowRoot;
    const gear = [...sr.querySelectorAll('.ey-actions button')].find(b => /설정|settings/i.test(b.title || ''));
    gear?.click();
  });
  await page.waitForTimeout(500);
  check(probe.totalLines > 0 && probe.pronLines > 0,
    `R5 ja 곡(${songs.r5.videoId}) ipa 선택해도 발음 줄이 통째로 비지 않음(hangul 폴백)`, probe);
  check(optionProbe.selectFound === true && optionProbe.hasIpaOption === true,
    'R5b 설정 시트에 ipa 옵션이 실제로 존재함', optionProbe);
}

// ══════════════════════════════════════════════════════════════════
// R6: 출처 표시가 "Everyric"으로 시작하지 않음
// ══════════════════════════════════════════════════════════════════
async function checkR6(page) {
  if (!songs.r6) return skip('R6 출처 표시', 'DB에 attribution 있는 everyric 싱크 없음');
  await gotoAndWaitSynced(page, songs.r6.videoId);
  const badge = await page.evaluate(() => {
    const sr = document.getElementById('everyric-root').shadowRoot;
    const el = sr.querySelector('.ey-source');
    return { present: !!el, text: el?.textContent ?? null };
  });
  check(badge.present === true && badge.text !== null && !/^everyric/i.test(badge.text.trim()),
    `R6 출처 배지가 "Everyric"으로 시작하지 않음(${songs.r6.videoId})`, badge);
}

// ══════════════════════════════════════════════════════════════════
// R1: 다음 영상 카드 클릭 이동
// ══════════════════════════════════════════════════════════════════
async function checkR1(page) {
  const videoId = songs.r4?.videoId ?? songs.r3?.videoId;
  if (!videoId) return skip('R1 다음 영상 카드 클릭', '사용할 실곡 없음');
  await setSettings({ modPlaylist: true, uiLanguage: 'ko' });
  await gotoAndWaitSynced(page, videoId);
  // 재생목록 부착 패널이 뜨고, 목록이 없는 단일 영상이면 폴백 다음 영상 카드가 채워질
  // 때까지 기다린다(refreshNextUp 5초 스로틀 + beginFollowing 즉시 1회 — 넉넉히 대기)
  await page.waitForFunction(() => {
    const sr = document.getElementById('everyric-root')?.shadowRoot;
    return !!sr?.querySelector('.ey-nextup-card');
  }, { timeout: 15000 }).catch(() => null);
  const cardProbe = await page.evaluate(() => {
    const sr = document.getElementById('everyric-root')?.shadowRoot;
    const card = sr?.querySelector('.ey-nextup-card');
    if (!card) return { present: false };
    const cursor = getComputedStyle(card).cursor;
    return { present: true, cursor, visible: card.getClientRects().length > 0 };
  });
  if (!cardProbe.present) {
    skip('R1 다음 영상 카드 클릭', '이 영상엔 다음 영상 정보가 안 잡힘(next 버튼에 다음 곡 정보 없음 — 다른 영상으로 재시도 필요)');
    return;
  }
  check(cardProbe.cursor === 'pointer' && cardProbe.visible === true,
    'R1a 카드에 클릭 가능 스타일(cursor:pointer)', cardProbe);

  const before = await page.evaluate(() => location.href);
  await page.evaluate(() => {
    document.getElementById('everyric-root').shadowRoot.querySelector('.ey-nextup-card')?.click();
  });
  // 실제 SPA 내비게이션(유튜브 자체 다음 버튼 클릭 경로) 또는 location.assign 폴백을 관측한다.
  // 다음 영상 로드에는 수 초가 걸릴 수 있어 넉넉히 기다린다.
  let navigated = false;
  try {
    await page.waitForFunction((prevHref) => location.href !== prevHref, before, { timeout: 12000 });
    navigated = true;
  } catch { /* 아래에서 미달로 보고 */ }
  const after = await page.evaluate(() => location.href);
  check(navigated === true && after !== before,
    'R1b 카드 클릭 시 실제 내비게이션 발생(location 변화)', { before, after, navigated });
}

// ══════════════════════════════════════════════════════════════════
// R2: list= URL 멤버십 오탐 재시도 (백오프 중 이전 상태 유지)
// ══════════════════════════════════════════════════════════════════
async function checkR2(page) {
  const videoId = songs.r5?.videoId ?? songs.r4?.videoId;
  if (!videoId) return skip('R2 재생목록 오탐 재시도', '사용할 실곡 없음');
  await setSettings({ modPlaylist: true, uiLanguage: 'ko' });
  // 유튜브가 모든 영상에 항상 제공하는 자동 믹스(RD 접두)로 list= 컨텍스트를 만든다 —
  // 특정 재생목록을 고를 필요 없이 어떤 videoId로도 재현 가능하다.
  const listId = `RD${videoId}`;
  await page.goto(`https://www.youtube.com/watch?v=${videoId}&list=${listId}`, {
    waitUntil: 'domcontentloaded', timeout: 90000,
  });
  await page.waitForFunction(
    () => !!document.getElementById('everyric-root')?.shadowRoot?.querySelector('.ey-line'),
    { timeout: 120000 },
  );
  // 초기 상태 즉시 스냅샷(백오프 재시도가 아직 안 끝났을 시점) — 오탐 문구가 이 순간에
  // 뜨면 안 된다. 정확한 첫 스크랩 타이밍은 알 수 없으므로 패널 등장 직후 최대한 빨리 본다.
  const immediate = await page.evaluate(() => {
    const sr = document.getElementById('everyric-root')?.shadowRoot;
    return sr?.querySelector('.ey-pl-status')?.textContent ?? null;
  });
  info('R2 내비게이션 직후 상태 문구', immediate);
  const falsePositiveImmediately = immediate != null && /재생목록에 속하지 않아요|not part of a playlist|doesn.t belong/i.test(immediate);
  check(!falsePositiveImmediately,
    'R2a 내비게이션 직후 "재생목록에 속하지 않아요"가 즉시 뜨지 않음(재시도 유예)', { immediate });

  // 백오프 전체 구간(500+1000+2000+4000=7500ms) 이후 최종 상태 — list=가 실제 목록이면
  // 항목이 채워져야 한다(유튜브 자동 믹스는 항상 채워진다).
  await page.waitForTimeout(9000);
  const settled = await page.evaluate(() => {
    const sr = document.getElementById('everyric-root')?.shadowRoot;
    const rows = sr?.querySelectorAll('.ey-pl-row').length ?? 0;
    return { statusText: sr?.querySelector('.ey-pl-status')?.textContent ?? null, rows };
  });
  check(settled.rows > 0,
    'R2b 백오프 종료 후 실제 재생목록(자동 믹스) 항목이 채워짐(오탐 아님 확정)', settled);
}

// ── 실행 ──────────────────────────────────────────────────────────
const page = await ctx.newPage();
try {
  await checkR3(page);
  await checkR4(page);
  await checkR5(page);
  await checkR6(page);
  await checkR1(page);
  await checkR2(page);
} finally {
  try { await page.screenshot({ path: join(SHOT_DIR, 'feedback-round3-check-final.png') }); } catch { /* 무시 */ }
  await ctx.close();
}

console.log(failed ? '\n== 결과: FAIL 있음' : '\n== 결과: 전부 PASS(또는 SKIP)');
process.exit(failed ? 1 : 0);
