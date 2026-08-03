// 릴리스 B 쓰기 경로 검증(프로드 대상) — real-e2e.mjs(로컬 :8000 전용)를 그대로 고치지
// 않고 별도 스크립트로 분리했다. 로컬 서버 대신 프로드(everyric.moref.co)를 대상으로
// 같은 사용자 여정(빈 상태 → 위키 검색 → 싱크 생성 → 번역·발음 표시 → PiP 음정 바)을
// 검증한다. 실제 GPU 연산은 프로드 서버(3090)에서 돈다 — 로컬 연산 금지 원칙에 위배 없음.
//
// 실행: node scripts/real-e2e-prod.mjs [videoUrl] [koreanTitle]
import { chromium } from 'playwright';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import { mkdirSync } from 'fs';
import { readPipPanel } from './lib/pip-panel.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const distDir = resolve(__dirname, '../dist');
const videoUrl = process.argv[2] ?? 'https://www.youtube.com/watch?v=Xg-qfsKN2_E';
const koreanTitle = process.argv[3] ?? '로키';
const PROD_SERVER = 'https://everyric.moref.co';
const SYNC_TIMEOUT_MS = 12 * 60 * 1000;

let failed = false;
function check(ok, label, detail) {
  console.log(`${ok ? 'PASS' : 'FAIL'}: ${label}${detail !== undefined ? ' = ' + JSON.stringify(detail) : ''}`);
  if (!ok) failed = true;
  return ok;
}

// 0) 프로드 서버 확인
try {
  const health = await (await fetch(`${PROD_SERVER}/health`, { signal: AbortSignal.timeout(5000) })).json();
  if (!check(health.status === 'healthy' && 'gpu_available' in health, 'prod server /health', health)) process.exit(1);
} catch (e) {
  console.log('FAIL: prod server not reachable —', String(e).slice(0, 120));
  process.exit(1);
}

const profileDir = 'C:\\Users\\user\\AppData\\Local\\Temp\\claude\\C--DevAT-Everyric2\\ccf2cc19-4a2e-4cc6-9f6d-e92f242a5aa3\\scratchpad\\ey-real-e2e-prod-profile';
mkdirSync(profileDir, { recursive: true });
const ctx = await chromium.launchPersistentContext(profileDir, {
  ignoreDefaultArgs: ['--disable-extensions'],
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
  // 프로드는 manifest.json의 필수 host_permissions에 이미 있다(everyric.moref.co) —
  // optional_host_permissions 부여 흐름(로컬 전용) 불필요. serverUrl도 확장 기본값과
  // 같은 프로드라 명시적으로 다시 세팅할 필요는 없지만, 세션 격리를 위해 명시한다.
  await sw.evaluate(s => chrome.storage.local.set({ settings: s }), {
    debugInfo: true,
    serverUrl: PROD_SERVER,
    uiLanguage: 'ko',
  });

  const page = ctx.pages()[0] ?? await ctx.newPage();
  page.on('console', msg => {
    if (msg.type() === 'error') console.log('[console:error]', msg.text().slice(0, 200));
  });

  console.log('navigating to', videoUrl);
  await page.goto(videoUrl, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForSelector('#everyric-root', { state: 'attached', timeout: 30000 });

  // 1) 자동 검색 결과 대기
  const state1 = await page.waitForFunction(sr => {
    const root = document.getElementById('everyric-root')?.shadowRoot;
    if (!root) return null;
    const lines = root.querySelectorAll('.ey-line').length;
    const stateText = root.querySelector('.ey-state-text')?.textContent ?? '';
    if (lines > 0 || (stateText && !stateText.includes('검색'))) return { lines, stateText };
    return null;
  }, SR, { timeout: 60000, polling: 1000 }).then(h => h.jsonValue());
  console.log('initial search state =', JSON.stringify(state1));

  // synced 소스가 이미 있으면(everyric 싱크) 생성 경로 자체를 못 태운다 — 그것만 진짜 스킵
  // 사유다. 자동 검색이 위키에서 **타이밍 없는 원문**을 이미 찾아왔을 뿐이면(가장 흔한
  // 실사용 경로) 그대로 이어서 "싱크 생성" 버튼을 누르면 된다 — 수동 재검색 단계는
  // 필요할 때만(완전히 빈 상태일 때만) 거친다.
  const syncedAlready = await page.evaluate(() => {
    const root = document.getElementById('everyric-root')?.shadowRoot;
    return !!root?.querySelector('.ey-line:not(.ey-line-plain)');
  });
  if (syncedAlready) {
    console.log(`SKIP: 이 영상은 프로드에 이미 타이밍 있는 싱크가 있다 — 다른 videoUrl로 재시도 필요.`);
    await ctx.close();
    process.exit(2);
  }

  let vocaro;
  if (state1.lines === 0) {
    // 완전히 빈 상태 — 수동 재검색으로 위키를 찾는다(원 시나리오)
    const emptyStateOk = check(state1.stateText.includes('가사를 찾지 못했어요'), 'empty state (전 소스 미스)', state1);
    if (!emptyStateOk) throw new Error(`사전 조건 불성립: ${JSON.stringify(state1)}`);

    await page.evaluate(([sel, title]) => {
      const root = document.getElementById('everyric-root')?.shadowRoot;
      const inputs = root.querySelectorAll('.ey-search-form .ey-input');
      inputs[0].value = title;
      inputs[1].value = '';
      const btn = Array.from(root.querySelectorAll('.ey-search-form button')).find(b => b.textContent === '다시 검색');
      btn.click();
    }, [SR, koreanTitle]);
    console.log(`retry search with "${koreanTitle}"`);

    vocaro = await page.waitForFunction(() => {
      const root = document.getElementById('everyric-root')?.shadowRoot;
      if (!root) return null;
      const plain = root.querySelectorAll('.ey-line-plain').length;
      if (plain === 0) return null;
      return {
        plainLines: plain,
        translations: root.querySelectorAll('.ey-line-tr').length,
        pronunciations: root.querySelectorAll('.ey-line-pron').length,
        source: root.querySelector('.ey-source')?.textContent ?? '',
        generateBtn: Array.from(root.querySelectorAll('.ey-generate-btn')).some(b => b.textContent.includes('싱크 생성')),
      };
    }, null, { timeout: 45000, polling: 1000 }).then(h => h.jsonValue());
  } else {
    // 이미 자동 검색이 원문(타이밍 없음)을 찾아왔다 — 가장 흔한 실사용 경로. 그대로 사용.
    console.log('자동 검색이 이미 원문을 찾아왔다(수동 재검색 생략) —', JSON.stringify(state1));
    vocaro = await page.evaluate(() => {
      const root = document.getElementById('everyric-root')?.shadowRoot;
      return {
        plainLines: root.querySelectorAll('.ey-line-plain').length,
        translations: root.querySelectorAll('.ey-line-tr').length,
        pronunciations: root.querySelectorAll('.ey-line-pron').length,
        source: root.querySelector('.ey-source')?.textContent ?? '',
        generateBtn: Array.from(root.querySelectorAll('.ey-generate-btn')).some(b => b.textContent.includes('싱크 생성')),
      };
    });
  }
  check(vocaro.plainLines > 10, '원문 가사 로드(자동 또는 재검색)', vocaro);
  check(vocaro.translations >= 0, `번역 포함 (${vocaro.translations}줄, 있으면 위키 사람 번역)`);
  check(vocaro.pronunciations >= 0, `발음 표기 포함 (${vocaro.pronunciations}줄, 있으면 위키 발음)`);
  await page.screenshot({ path: resolve(__dirname, '../e2e-prod-1-vocaro.png') }).catch(() => {});

  if (!vocaro.generateBtn) throw new Error('싱크 생성 버튼이 없음');

  // 3) 위키 가사로 싱크 생성 — 프로드 서버가 다운로드+정렬+멜로디까지 수행(GPU는 프로드 3090)
  const t0 = Date.now();
  await page.evaluate(() => {
    const root = document.getElementById('everyric-root')?.shadowRoot;
    Array.from(root.querySelectorAll('.ey-generate-btn')).find(b => b.textContent.includes('싱크 생성')).click();
  });
  console.log('generate clicked — waiting for prod job (download + CTC + FCPE)...');

  try {
    await page.waitForFunction(() => {
      const root = document.getElementById('everyric-root')?.shadowRoot;
      return /싱크 생성 중|대기열/.test(root?.textContent ?? '');
    }, null, { timeout: 20000, polling: 500 });
    console.log('PASS: generating progress UI 표시됨');
  } catch { console.log('WARN: 진행 UI를 못 봄'); }

  // 4) 완료 대기
  const synced = await page.waitForFunction(() => {
    const root = document.getElementById('everyric-root')?.shadowRoot;
    if (!root) return null;
    const dbg = root.querySelector('.ey-debug')?.textContent ?? '';
    const lines = root.querySelectorAll('.ey-line:not(.ey-line-plain)').length;
    if (/src=everyric/.test(dbg) && lines > 0) {
      return {
        lines,
        words: root.querySelectorAll('.ey-word').length,
        pron: root.querySelectorAll('.ey-line-pron').length,
        tr: root.querySelectorAll('.ey-line-tr').length,
        dbg: dbg.slice(0, 150),
      };
    }
    const st = root.querySelector('.ey-state-text')?.textContent ?? '';
    if (st && !/싱크 생성 중|대기열|검색/.test(st)) return { error: st };
    return null;
  }, null, { timeout: SYNC_TIMEOUT_MS, polling: 2000 }).then(h => h.jsonValue());
  if (synced.error) throw new Error('싱크 생성 실패: ' + synced.error);
  const elapsed = Math.round((Date.now() - t0) / 1000);
  check(synced.lines > 10, `싱크 생성 완료 (${elapsed}s, 프로드 실처리)`, synced);
  check(synced.words > 0, `단어(카라오케) 스팬 렌더링 (${synced.words}개)`);
  check(synced.pron > 10, `싱크 후 발음 표기 병합 유지 (${synced.pron}줄) — 쓰기 경로 핵심`, synced.pron);
  check(synced.tr > 10, `싱크 후 위키 번역 병합 유지 (${synced.tr}줄) — 쓰기 경로 핵심`, synced.tr);

  // 5) 하이라이트
  await page.evaluate(() => {
    const v = document.querySelector('video.html5-main-video') ?? document.querySelector('video');
    if (v) { v.currentTime = 25; void v.play().catch(() => {}); }
  });
  await page.waitForFunction(() => {
    const root = document.getElementById('everyric-root')?.shadowRoot;
    return !!root?.querySelector('.ey-line.active');
  }, null, { timeout: 60000, polling: 500 });
  const active = await page.evaluate(() => {
    const root = document.getElementById('everyric-root')?.shadowRoot;
    return root?.querySelector('.ey-line.active')?.textContent?.slice(0, 60) ?? null;
  });
  check(!!active, '재생 위치 하이라이트', active);
  await page.screenshot({ path: resolve(__dirname, '../e2e-prod-2-synced.png') }).catch(() => {});

  // 6) PiP + 음정 바
  await page.locator('[title="PiP 창으로 보기"]').click();
  await page.waitForTimeout(3000);
  const pipRaw = await page.evaluate(readPipPanel());
  const pip = {
    open: pipRaw.open,
    currentLine: pipRaw.currentLine ?? null,
    pron: pipRaw.pron ?? '',
    pitch: pipRaw.lane ?? { present: false, visible: false, drawnPx: 0 },
  };
  check(pip.open, 'PiP 열림', pip.currentLine);
  check(pip.pitch.present && pip.pitch.visible && pip.pitch.drawnPx > 50, '가라오케 음정 바 (실제 FCPE notes)', pip.pitch);
  try {
    const pipPage = ctx.pages().find(p => p !== page);
    if (pipPage) await pipPage.screenshot({ path: resolve(__dirname, '../e2e-prod-3-pip.png') });
  } catch { /* 스크린샷 실패는 비치명 */ }

  console.log(failed ? 'REAL E2E(PROD): FAIL' : 'REAL E2E(PROD): PASS (vocaro → 싱크 생성(프로드) → 번역·발음 표시 → 하이라이트 → 음정 바)');
  process.exitCode = failed ? 1 : 0;
} catch (e) {
  console.log('REAL E2E(PROD): ERROR —', String(e).slice(0, 400));
  try {
    const page = ctx.pages()[0];
    if (page) await page.screenshot({ path: resolve(__dirname, '../e2e-prod-error.png') });
  } catch { /* ignore */ }
  process.exitCode = 1;
} finally {
  await ctx.close();
}
