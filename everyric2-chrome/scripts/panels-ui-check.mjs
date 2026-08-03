// 2026-08-03 panels.ts 조각 배선분 실브라우저 검증 — "빌드가 된다"가 아니라
// "설정을 찾을 수 있고, 제보가 실수로 나가지 않는다".
//
//   P1) 설정 시트가 범주(details)로 나뉘고 접었다 폈다 된다
//   P2) 검색칸이 보이는 줄을 실제로 줄인다(안 걸린 범주는 통째로 사라진다)
//   P3) '초기화' 범주의 가라오케 안내 되살리기 버튼에 도달할 수 있다
//        — "다시 보지 않기"를 무를 유일한 경로라 없어지면 되돌릴 방법이 사라진다
//   P4) 별점은 별 하나로 전송이 끝난다(보내기 버튼 없이)
//   P5) 신고는 **확인 단계 없이는 나가지 않는다** (되돌릴 수 없는 동작)
//   P6) 공지·기여 진입점이 헤더에 있고, 시트를 닫으면 보던 가사가 그대로 돌아온다
//        (공지 버튼은 서버에 그 기능이 없으면 숨는 것이 정상이다 — 오류 화면이 아니다)
//   P7) "이 가사가 아니에요"도 확인을 거친다 — 매칭 표시줄이 있는 곡에서만 검사한다
//
// 실행: node scripts/panels-ui-check.mjs [videoId]
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
const VIDEO = process.argv[2] ?? 'UgK6n1KKUxY';
let failed = false;
function check(ok, label, detail) {
  console.log(`${ok ? 'PASS' : 'FAIL'}: ${label}${detail !== undefined ? ' = ' + JSON.stringify(detail) : ''}`);
  if (!ok) failed = true;
  return ok;
}
function info(label, detail) { console.log(`INFO: ${label} = ${JSON.stringify(detail)}`); }

const health = await (await fetch(`${SERVER}/health`, { signal: AbortSignal.timeout(3000) })).json();
if (!check(health.status === 'healthy', 'real server /health', health)) process.exit(1);

const userDataDir = mkdtempSync(join(tmpdir(), 'ey-panels-'));
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

// 표시 언어를 ko로 고정 — 아래 판정이 한국어 라벨을 본다(브라우저 로케일에 흔들리지 않게)
await sw.evaluate(async (url) => {
  const cur = (await chrome.storage.local.get('settings')).settings ?? {};
  await chrome.storage.local.set({ settings: { ...cur, serverUrl: url, uiLanguage: 'ko' } });
}, SERVER);

const page = await ctx.newPage();
await page.goto(`https://www.youtube.com/watch?v=${VIDEO}`, { waitUntil: 'domcontentloaded', timeout: 90000 });

await page.waitForFunction(
  () => !!document.getElementById('everyric-root')?.shadowRoot?.querySelector('.ey-line'),
  { timeout: 120000 });
await page.waitForTimeout(3000);

/** shadow root 안에서 라벨(title 또는 본문)로 버튼을 찾아 누른다 */
async function clickByText(selector, pattern) {
  return page.evaluate(({ selector, pattern }) => {
    const sr = document.getElementById('everyric-root').shadowRoot;
    const re = new RegExp(pattern);
    const el = [...sr.querySelectorAll(selector)]
      .find(b => re.test(b.title || '') || re.test((b.textContent || '').trim()));
    if (!el) return false;
    el.click();
    return true;
  }, { selector, pattern });
}

// ── P1: 범주 + 접힘 ──────────────────────────────────────────────
if (!check(await clickByText('.ey-actions .ey-btn', '^설정$'), 'P1 헤더 설정 버튼을 찾음')) {
  await ctx.close();
  process.exit(1);
}
await page.waitForTimeout(800);

const sheet = await page.evaluate(() => {
  const sr = document.getElementById('everyric-root').shadowRoot;
  const el = sr.querySelector('.ey-settings');
  if (!el) return { present: false };
  const sections = [...el.querySelectorAll('.ey-settings-section')];
  return {
    present: true,
    hasFilter: !!el.querySelector('.ey-settings-filter'),
    filterFocused: sr.activeElement?.classList.contains('ey-settings-filter') ?? false,
    sections: sections.length,
    titles: sections.map(s => (s.querySelector('.ey-settings-section-title')?.textContent || '').trim()),
    open: sections.map(s => s.open),
    // 옛 평면 시트에는 범주가 없었다 — 줄 수는 그대로여야 한다(설정이 사라지면 안 됨)
    rows: el.querySelectorAll('.ey-settings-row').length,
  };
});
info('설정 시트', sheet);
check(sheet.present && sheet.sections >= 6, 'P1 설정이 범주(details)로 나뉘어 있다', sheet.sections);
check(sheet.hasFilter, 'P1 설정 검색칸이 있다');
check(sheet.open.filter(Boolean).length === 1 && sheet.open[0],
  'P1 기본 펼침은 첫 범주 하나뿐(40줄 벽이 그대로 돌아오지 않는다)', sheet.open);
check(sheet.rows >= 40, 'P1 설정 줄 수가 유지됨(범주화로 사라진 설정 없음)', sheet.rows);

// 접힌 두 번째 범주를 눌러 펼친다
const toggled = await page.evaluate(() => {
  const sr = document.getElementById('everyric-root').shadowRoot;
  const target = [...sr.querySelectorAll('.ey-settings-section')].find(s => !s.open);
  if (!target) return null;
  const before = target.open;
  target.querySelector('.ey-settings-section-head').click();
  return { before, after: target.open };
});
check(toggled !== null && toggled.before === false && toggled.after === true,
  'P1 범주 머리를 누르면 펼쳐진다', toggled);

// ── P2: 검색이 보이는 줄을 줄인다 ────────────────────────────────
const filtered = await page.evaluate(() => {
  const sr = document.getElementById('everyric-root').shadowRoot;
  const el = sr.querySelector('.ey-settings');
  const visibleRows = () => [...el.querySelectorAll('.ey-settings-row, .ey-settings-note')]
    .filter(r => r.getClientRects().length > 0).length;
  const hiddenSections = () => [...el.querySelectorAll('.ey-settings-section')]
    .filter(s => s.style.display === 'none').length;
  // 먼저 전부 펼쳐 기준선을 만든다 — 접힌 범주의 줄은 원래 안 보이므로 비교가 안 된다
  el.querySelector('.ey-settings-expand-all').click();
  const all = visibleRows();
  const filter = el.querySelector('.ey-settings-filter');
  filter.value = '마이크';
  filter.dispatchEvent(new Event('input', { bubbles: true }));
  const narrowed = visibleRows();
  const hidden = hiddenSections();
  // 지운 뒤에는 사용자가 정해 둔 펼침 상태로 정확히 돌아와야 한다
  filter.value = '';
  filter.dispatchEvent(new Event('input', { bubbles: true }));
  return { all, narrowed, hidden, restored: visibleRows() };
});
info('검색 결과', filtered);
check(filtered.narrowed > 0 && filtered.narrowed < filtered.all,
  'P2 검색어가 보이는 줄을 실제로 줄인다', filtered);
check(filtered.hidden > 0, 'P2 안 걸린 범주는 통째로 사라진다', filtered.hidden);
check(filtered.restored === filtered.all, 'P2 검색어를 지우면 원래 펼침 상태로 돌아온다', filtered);

// 설정 키로도 찾을 수 있어야 한다 — 라벨을 모른 채 'serverUrl'로 오는 경로
const byKey = await page.evaluate(() => {
  const sr = document.getElementById('everyric-root').shadowRoot;
  const el = sr.querySelector('.ey-settings');
  const filter = el.querySelector('.ey-settings-filter');
  filter.value = 'serverUrl';
  filter.dispatchEvent(new Event('input', { bubbles: true }));
  const rows = [...el.querySelectorAll('.ey-settings-row')].filter(r => r.getClientRects().length > 0);
  const out = { hits: rows.length, labels: rows.map(r => (r.querySelector('label')?.textContent || '').trim()) };
  filter.value = '';
  filter.dispatchEvent(new Event('input', { bubbles: true }));
  return out;
});
check(byKey.hits > 0, 'P2 라벨이 아닌 설정 키(serverUrl)로도 찾힌다', byKey);

// ── P3: 초기화 범주의 안내 되살리기 버튼 ─────────────────────────
const reset = await page.evaluate(() => {
  const sr = document.getElementById('everyric-root').shadowRoot;
  const el = sr.querySelector('.ey-settings');
  const filter = el.querySelector('.ey-settings-filter');
  filter.value = '가라오케 타이밍';
  filter.dispatchEvent(new Event('input', { bubbles: true }));
  const btn = [...el.querySelectorAll('button')]
    .find(b => /다시 보기/.test((b.textContent || '').trim()));
  if (!btn) return { found: false };
  const visible = btn.getClientRects().length > 0;
  btn.click();
  return { found: true, visible, after: (btn.textContent || '').trim() };
});
info('초기화 버튼', reset);
check(reset.found && reset.visible, 'P3 "가라오케 타이밍 안내 다시 보기"에 도달 가능', reset);
await page.waitForTimeout(1200);
const dismissedAfter = await sw.evaluate(async () =>
  (await chrome.storage.local.get('settings')).settings.karaokeTimingNoticeDismissed);
check(dismissedAfter === false, 'P3 누르면 karaokeTimingNoticeDismissed가 풀린다', { dismissedAfter });

// 설정 시트를 닫는다 — 아래 별점 팝오버는 푸터 위에 뜬다
await clickByText('.ey-settings > .ey-secondary-btn', '^닫기$');
await page.waitForTimeout(600);

// ── P4: 별점은 별 하나로 끝난다 ──────────────────────────────────
if (!check(await clickByText('.ey-feedback-btn', ''), 'P4 별점 버튼(★)을 찾음')) {
  await page.screenshot({ path: resolve(__dirname, '../panels-ui-check.png') });
  await ctx.close();
  process.exit(1);
}
await page.waitForTimeout(500);

const pop = await page.evaluate(() => {
  const sr = document.getElementById('everyric-root').shadowRoot;
  const el = sr.querySelector('.ey-feedback-pop');
  return {
    visible: el?.getClientRects().length > 0,
    isRatingPop: !!el?.querySelector('.ey-rating-pop'),
    stars: el?.querySelectorAll('.ey-rating-star').length ?? 0,
    // 옛 팝오버의 '보내기' 버튼·유형 select가 별점 화면에 남아 있으면 분리가 안 된 것
    selects: el?.querySelectorAll('select').length ?? 0,
    reportBtn: !!el?.querySelector('.ey-rating-report-btn'),
  };
});
info('별점 팝오버', pop);
check(pop.visible && pop.isRatingPop && pop.stars === 5, 'P4 별점 팝오버가 별 5개로 뜬다', pop);
check(pop.selects === 0 && pop.reportBtn,
  'P4 별점 화면에는 유형/코멘트가 없고 신고는 별도 버튼이다', pop);

// 별 하나 클릭 = 전송(추가 버튼 없음) — 로컬 서버에 별점 한 건이 실제로 남는다
await page.evaluate(() => {
  const sr = document.getElementById('everyric-root').shadowRoot;
  sr.querySelectorAll('.ey-rating-star')[4].click();
});
await page.waitForTimeout(2500);
const rated = await page.evaluate(() => {
  const sr = document.getElementById('everyric-root').shadowRoot;
  const el = sr.querySelector('.ey-feedback-pop');
  return { status: (el?.querySelector('.ey-rating-status')?.textContent || '').trim() };
});
info('별점 전송 결과', rated);
check(/고마워요/.test(rated.status), 'P4 별 클릭 한 번으로 전송이 완료된다', rated);

// ── P5: 신고는 확인 없이는 나가지 않는다 ─────────────────────────
// 전송 성공 시 1.2초 뒤 팝오버가 닫히므로 다시 연다
await page.waitForTimeout(1500);
await clickByText('.ey-feedback-btn', '');
await page.waitForTimeout(500);
check(await clickByText('.ey-rating-report-btn', '신고'), 'P5 신고 버튼을 누름');
await page.waitForTimeout(500);

// fast/medium 싱크면 깊이 안내가 먼저 뜬다 — 그 경우 '그래도 제보'로 폼까지 간다
const hint = await page.evaluate(() => {
  const sr = document.getElementById('everyric-root').shadowRoot;
  const el = sr.querySelector('.ey-report-hint');
  return el ? { present: true, text: (el.textContent || '').trim().slice(0, 70) } : { present: false };
});
info('깊이 안내', hint);
if (hint.present) {
  check(await clickByText('.ey-report-hint-actions button', '그래도'), 'P5 깊이 안내에서 제보로 계속');
  await page.waitForTimeout(400);
}

const form = await page.evaluate(() => {
  const sr = document.getElementById('everyric-root').shadowRoot;
  const el = sr.querySelector('.ey-report-sheet');
  if (!el) return { present: false };
  const comment = el.querySelector('input.ey-input');
  if (comment) {
    comment.value = 'panels-ui-check (전송되지 않아야 함)';
    comment.dispatchEvent(new Event('input', { bubbles: true }));
  }
  return {
    present: true,
    hasCategory: !!el.querySelector('select'),
    hasComment: !!comment,
    title: (el.querySelector('.ey-report-title')?.textContent || '').trim(),
  };
});
info('제보 폼', form);
check(form.present && form.hasCategory && form.hasComment, 'P5 제보 폼에 유형+코멘트가 있다', form);

// '제보'를 눌러도 **아직 나가면 안 된다** — 확인 단계가 서야 한다
await page.evaluate(() => {
  const sr = document.getElementById('everyric-root').shadowRoot;
  const el = sr.querySelector('.ey-report-sheet');
  [...el.querySelectorAll('.ey-report-actions button')]
    .find(b => /제보/.test((b.textContent || '').trim()))?.click();
});
await page.waitForTimeout(600);
const confirmStage = await page.evaluate(() => {
  const sr = document.getElementById('everyric-root').shadowRoot;
  const el = sr.querySelector('.ey-report-confirm');
  return {
    present: !!el,
    text: (el?.querySelector('.ey-report-title')?.textContent || '').trim(),
    // 여기서 취소하면 폼으로 돌아가되 입력이 남아 있어야 한다(다시 쓰게 만들지 않는다)
    buttons: [...(el?.querySelectorAll('button') ?? [])].map(b => (b.textContent || '').trim()),
  };
});
info('확인 단계', confirmStage);
check(confirmStage.present && /제보하시겠어요/.test(confirmStage.text),
  'P5 제보는 확인 단계를 반드시 거친다(클릭 즉시 전송 아님)', confirmStage);

// 취소 → 폼 복귀 + 입력 보존 (여기서 끝내므로 이 제보는 서버로 나가지 않는다)
await page.evaluate(() => {
  const sr = document.getElementById('everyric-root').shadowRoot;
  [...sr.querySelectorAll('.ey-report-confirm button')]
    .find(b => /취소/.test((b.textContent || '').trim()))?.click();
});
await page.waitForTimeout(500);
const backToForm = await page.evaluate(() => {
  const sr = document.getElementById('everyric-root').shadowRoot;
  const el = sr.querySelector('.ey-report-sheet');
  return {
    form: !!el?.querySelector('.ey-report-actions'),
    comment: el?.querySelector('input.ey-input')?.value ?? '',
  };
});
info('취소 후', backToForm);
check(backToForm.form && backToForm.comment.includes('panels-ui-check'),
  'P5 확인을 취소하면 입력이 남은 폼으로 돌아온다', backToForm);

// ── P6: 공지 · 기여 진입점 ───────────────────────────────────────
// 공지 버튼은 서버에 공지 기능이 **있을 때만** 나온다(구서버는 진입점 자체가 없다) —
// 어느 쪽이든 "오류 화면"이 뜨면 안 되므로 보이면 눌러 보고, 안 보이면 그 사실을 남긴다.
const entries = await page.evaluate(() => {
  const sr = document.getElementById('everyric-root').shadowRoot;
  const btn = t => [...sr.querySelectorAll('.ey-actions .ey-btn')].find(b => b.title === t);
  const notices = btn('공지사항');
  const contrib = btn('내 기여 · 남은 횟수');
  return {
    noticesPresent: !!notices,
    noticesVisible: (notices?.getClientRects().length ?? 0) > 0,
    noticesUnread: (notices?.querySelector('.ey-unread-dot')?.getClientRects().length ?? 0) > 0,
    contribVisible: (contrib?.getClientRects().length ?? 0) > 0,
  };
});
info('헤더 진입점', entries);
check(entries.contribVisible, 'P6 기여 진입점이 헤더에 있다', entries);

if (entries.noticesVisible) {
  await clickByText('.ey-actions .ey-btn', '^공지사항$');
  await page.waitForTimeout(2000);
  const sheet = await page.evaluate(() => {
    const sr = document.getElementById('everyric-root').shadowRoot;
    const el = sr.querySelector('.ey-notices');
    return { present: !!el, items: el?.querySelectorAll('.ey-notice-item').length ?? 0,
      text: (el?.textContent || '').trim().slice(0, 60) };
  });
  info('공지 시트', sheet);
  check(sheet.present, 'P6 공지 시트가 본문에 그려진다', sheet);
  check(await clickByText('.ey-sheet-back', '돌아가기|Back|戻'), 'P6 공지에서 뒤로 나갈 수 있다');
  await page.waitForTimeout(2000);
} else {
  info('공지 진입점', '이 서버에는 공지 기능이 없어 버튼이 숨겨졌다(정상 경로)');
}

await clickByText('.ey-actions .ey-btn', '^내 기여');
await page.waitForTimeout(2500);
const contrib = await page.evaluate(() => {
  const sr = document.getElementById('everyric-root').shadowRoot;
  const el = sr.querySelector('.ey-contrib');
  return {
    present: !!el,
    headline: (el?.querySelector('.ey-contrib-headline')?.textContent || '').trim(),
    items: el?.querySelectorAll('.ey-contrib-item').length ?? 0,
    privacy: (el?.querySelector('.ey-contrib-privacy')?.textContent || '').trim().slice(0, 30),
  };
});
info('기여 시트', contrib);
check(contrib.present && contrib.privacy.length > 0, 'P6 기여 시트가 본문에 그려진다', contrib);
check(await clickByText('.ey-sheet-back', '돌아가기|Back|戻'), 'P6 기여에서 뒤로 나갈 수 있다');
await page.waitForTimeout(2500);
const restored = await page.evaluate(() => {
  const sr = document.getElementById('everyric-root').shadowRoot;
  return { lines: sr.querySelectorAll('.ey-line').length };
});
check(restored.lines > 0, 'P6 뒤로 나가면 보던 가사가 그대로 돌아온다', restored);

// ── P7: 오매칭 제보도 확인을 거친다 ──────────────────────────────
// 매칭 표시줄은 위키가 고른 곡에서만 뜬다. 어떤 곡을 넣어도 이 관문은 검사돼야 하므로,
// 안 뜬 곡에서는 **이미 만들어져 숨어 있는 그 줄**을 보이게만 한다 — 버튼도 핸들러도
// 제품 코드 그대로다(가짜 DOM을 심는 것이 아니라 display만 되돌린다).
const matched = await page.evaluate(() => {
  const sr = document.getElementById('everyric-root').shadowRoot;
  const bar = sr.querySelector('.ey-matched-bar');
  if (!bar) return { visible: false, nudged: false };
  if (bar.getClientRects().length > 0) return { visible: true, nudged: false };
  bar.style.display = '';
  return { visible: bar.getClientRects().length > 0, nudged: true };
});
info('매칭 표시줄', matched);
if (matched.visible) {
  await clickByText('.ey-matched-report', '');
  await page.waitForTimeout(500);
  const confirmPop = await page.evaluate(() => {
    const sr = document.getElementById('everyric-root').shadowRoot;
    const el = sr.querySelector('.ey-confirm-pop');
    return {
      present: !!el, visible: (el?.getClientRects().length ?? 0) > 0,
      text: (el?.querySelector('.ey-confirm-text')?.textContent || '').trim(),
      buttons: [...(el?.querySelectorAll('button') ?? [])].map(b => (b.textContent || '').trim()),
    };
  });
  info('오매칭 확인', confirmPop);
  check(confirmPop.present && confirmPop.visible,
    'P7 "이 가사가 아니에요"는 즉시 제보하지 않고 확인을 세운다', confirmPop);
  // 취소로 물러난다 — 여기서 확인을 누르면 실제 오매칭 제보가 나간다
  await clickByText('.ey-confirm-actions button', '취소|Cancel|キャンセル');
  await page.waitForTimeout(400);
  const closed = await page.evaluate(() => {
    const sr = document.getElementById('everyric-root').shadowRoot;
    return (sr.querySelector('.ey-confirm-slot')?.getClientRects().length ?? 0) === 0;
  });
  check(closed, 'P7 취소하면 확인이 닫힌다(제보는 나가지 않는다)', { closed });
} else {
  info('오매칭 확인', '이 곡에는 매칭 표시줄이 없어 건너뜀');
}

await page.screenshot({ path: resolve(__dirname, '../panels-ui-check.png') });
console.log('screenshot: panels-ui-check.png');
console.log(`PANELS UI CHECK: ${failed ? 'FAIL' : 'PASS'}`);
await ctx.close();
process.exit(failed ? 1 : 0);
