// 다국어 매트릭스 실브라우저 E2E — **프로드 서버 대상, 읽기 전용**.
//
// 로컬 연산 절대 금지(5090·CPU 모두)라 생성(정렬)은 여기서 하지 않는다 — 쓰기 경로는
// 서버측 verify_regen(3090)으로 이미 검증됐고, 이 스크립트는 «브라우저에서만 보이는
// 것»을 검증한다: 렌더·가드·i18n·실메시징.
//
//   KO-0) 기본 설정(ko) + JW3N → 싱크 라인 + 한글 발음 + 한국어 번역 (오늘자 무회귀)
//   EN-1) translationLanguage=en → 같은 곡의 발음이 romaji로 렌더 (pron dict 경로)
//   EN-2) 영어 번역이 서버 레이어에서 도착(translation_lang=en) — 로컬 캐시 비우고
//         재로드해도 즉시 재수신 (남의 언어 수신 가드의 실동작)
//   EN-3) 옵션 페이지에서 chrome.runtime 실메시지로 MIRAHEZE_LOOKUP → 실 위키
//         검색·파싱 (romaji+영어) — 검색 UI 여정은 프로드에 무싱크 곡이 필요해 제외
//   EN-4) uiLanguage=en → 옵션 페이지 정적 텍스트가 영어로 (i18n 기계 전체 경로)
//
// 실행: EVERYRIC_E2E_KEYFILE=<키파일경로> node scripts/real-e2e-multilang.mjs
//   키 값은 파일로만 전달한다 — 명령행·로그에 값이 남지 않게.
import { chromium } from 'playwright';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import { mkdtempSync, readFileSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { ensureLocalServerPermissionForServerUrl, isLoopbackServerUrl } from './lib/local-server-permission.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const distDir = resolve(__dirname, '../dist');
const SERVER = process.env.EVERYRIC_E2E_SERVER ?? 'https://everyric.moref.co';
const syncedUrl = 'https://www.youtube.com/watch?v=JW3N-HvU0MA'; // 프로드에 pron dict+en 레이어 실존
const wikiTitle = 'ロキ'; // miraheze 실측 검증 곡 (접두 일치로 곡 페이지 히트)
const HAS_HANGUL = '[\\uAC00-\\uD7A3]';

const apiKey = process.env.EVERYRIC_E2E_KEYFILE
  ? readFileSync(process.env.EVERYRIC_E2E_KEYFILE, 'utf8').trim()
  : '';
// 로컬 서버(EVERYRIC_SERVER_API_KEY 미설정이 기본)는 키를 요구하지 않는다(server/main.py
// require_api_key — server.api_key가 없으면 통과). 프로드만 전 /api가 키를 요구하므로
// 키 강제는 SERVER가 실제로 원격일 때만 건다.
if (!apiKey && !isLoopbackServerUrl(SERVER)) {
  console.log('FAIL: EVERYRIC_E2E_KEYFILE 필요 (프로드는 전 /api가 키 요구)');
  process.exit(1);
}

let failed = false;
function check(ok, label, detail) {
  console.log(`${ok ? 'PASS' : 'FAIL'}: ${label}${detail !== undefined ? ' = ' + JSON.stringify(detail).slice(0, 200) : ''}`);
  if (!ok) failed = true;
  return ok;
}

try {
  const health = await (await fetch(`${SERVER}/health`, {
    headers: { 'x-api-key': apiKey }, signal: AbortSignal.timeout(5000),
  })).json();
  if (!check(health.status === 'healthy', 'prod /health', health)) process.exit(1);
} catch (e) {
  console.log('FAIL: prod server unreachable —', String(e).slice(0, 120));
  process.exit(1);
}

const userDataDir = mkdtempSync(join(tmpdir(), 'everyric-e2e-ml-'));
// headless에서 channel을 안 주면 Playwright가 chromium_headless_shell을 쓰는데 그 바이너리는
// 확장을 지원하지 않는다(실측: chrome-extension:// ERR_ABORTED) — 'chromium'을 명시해야
// 풀 크로뮴의 신형 headless로 뜬다. 프로브가 통과하고 본 하네스만 죽던 차이가 이것.
const channel = process.env.EVERYRIC_E2E_CHANNEL
  ?? (process.env.EVERYRIC_E2E_HEADLESS === '1' ? 'chromium' : '');
const ctx = await chromium.launchPersistentContext(userDataDir, {
  ignoreDefaultArgs: ['--disable-extensions'],
  ...(channel ? { channel } : {}),
  headless: process.env.EVERYRIC_E2E_HEADLESS === '1',
  viewport: { width: 1440, height: 900 },
  args: [
    `--disable-extensions-except=${distDir}`,
    `--load-extension=${distDir}`,
    '--mute-audio',
    '--autoplay-policy=no-user-gesture-required',
    '--window-position=40,40',
  ],
});

async function setSettings(sw, patch) {
  await sw.evaluate(async p => {
    const { settings } = await chrome.storage.local.get('settings');
    await chrome.storage.local.set({ settings: { ...(settings ?? {}), ...p } });
  }, patch);
}

const readLines = () => {
  const root = document.getElementById('everyric-root')?.shadowRoot;
  if (!root) return null;
  const lines = root.querySelectorAll('.ey-line:not(.ey-line-plain)').length;
  if (lines === 0) return null;
  const prons = Array.from(root.querySelectorAll('.ey-line-pron')).map(el => (el.textContent ?? '').trim()).filter(Boolean);
  const trs = Array.from(root.querySelectorAll('.ey-line-tr')).map(el => (el.textContent ?? '').trim()).filter(Boolean);
  return {
    lines,
    pron: prons.length,
    pronSample: prons.find(p => p.length > 3) ?? '',
    tr: trs.length,
    trSamples: trs.slice(0, 3),
  };
};

// MV3 SW는 기동 직후 idle로 잠들 수 있어 waitForEvent 한 방은 레이스가 있다(실측:
// 프로브는 통과, 본 하네스는 45s 타임아웃). 폴링하다 안 뜨면 확장 페이지를 열어
// 깨운다 — 언팩 확장 ID는 절대경로 해시라 같은 dist면 고정이다.
const KNOWN_EXT_ID = 'pkhjekjjpccigehfkljffncdhnhfmoob';
async function acquireServiceWorker() {
  for (let i = 0; i < 15; i++) {
    const sw = ctx.serviceWorkers()[0];
    if (sw) return sw;
    await new Promise(r => setTimeout(r, 1000));
  }
  const waker = await ctx.newPage();
  await waker.goto(`chrome-extension://${KNOWN_EXT_ID}/src/options.html`, { timeout: 15000 })
    .catch(e => console.log('waker goto:', String(e).slice(0, 80)));
  for (let i = 0; i < 15; i++) {
    const sw = ctx.serviceWorkers()[0];
    if (sw) { await waker.close(); return sw; }
    await new Promise(r => setTimeout(r, 1000));
  }
  await waker.close();
  return ctx.waitForEvent('serviceworker', { timeout: 15000 });
}

try {
  const sw = await acquireServiceWorker();
  console.log('extension loaded:', sw.url());
  const extId = new URL(sw.url()).host;
  // SERVER가 프로드(기본값)면 무동작 — 필수 host_permissions로 이미 허용됨.
  // EVERYRIC_E2E_SERVER로 로컬을 가리키면(optional_host_permissions) 실제 옵션 페이지
  // "허용" 흐름을 재현해 부여한다(scripts/lib/local-server-permission.mjs).
  await ensureLocalServerPermissionForServerUrl(ctx, sw, extId, SERVER);
  await setSettings(sw, {
    serverUrl: SERVER, apiKey,
    showTranslation: true, showPronunciation: true, debugInfo: true,
    translationLanguage: 'ko', uiLanguage: 'auto',
  });

  const page = ctx.pages()[0] ?? await ctx.newPage();
  page.on('console', msg => {
    if (msg.type() === 'error') console.log('[console:error]', msg.text().slice(0, 200));
  });

  // ── KO-0) 한국어 사용자 무회귀 ──────────────────────────────────────────────
  console.log('navigating to', syncedUrl, '(ko)');
  await page.goto(syncedUrl, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForSelector('#everyric-root', { state: 'attached', timeout: 30000 });
  const ko = await page.waitForFunction(readLines, null, { timeout: 90000, polling: 1000 }).then(h => h.jsonValue())
    .catch(async e => {
      // 실패 시 패널 상태를 덤프한다 — 유튜브 봇 월·서버 오류·검색 스톨을 가른다
      const st = await page.evaluate(() => {
        const root = document.getElementById('everyric-root')?.shadowRoot;
        return {
          state: root?.querySelector('.ey-state-text')?.textContent ?? null,
          banner: root?.querySelector('.ey-banner')?.textContent ?? null,
          rootText: (root?.textContent ?? '').replace(/\s+/g, ' ').slice(0, 200),
          pageTitle: document.title.slice(0, 80),
        };
      }).catch(() => null);
      console.log('KO-0 진단 덤프:', JSON.stringify(st));
      await page.screenshot({ path: resolve(__dirname, '../e2e-ml-fail.png') }).catch(() => {});
      throw e;
    });
  check(ko.lines > 10, 'KO-0 싱크 라인 렌더', ko.lines);
  check(ko.pron > 10 && new RegExp(HAS_HANGUL).test(ko.pronSample), 'KO-0 한글 발음 렌더', ko.pronSample.slice(0, 40));
  check(ko.tr > 10 && ko.trSamples.some(t => new RegExp(HAS_HANGUL).test(t)), 'KO-0 한국어 번역 렌더', ko.trSamples[0]?.slice(0, 40));
  await page.screenshot({ path: resolve(__dirname, '../e2e-ml-0-ko.png') });
  console.log('screenshot: e2e-ml-0-ko.png');

  // ── EN-1·2) 영어권 사용자: romaji + 레이어 번역 ────────────────────────────
  await setSettings(sw, { translationLanguage: 'en' });
  // 언어 전환 시 확장이 로컬 번역 캐시로 응답하지 못하게 비운다 — 서버 레이어 경로를 가른다
  const cleared = await sw.evaluate(async () => {
    const all = await chrome.storage.local.get(null);
    const victims = Object.keys(all).filter(k => k !== 'settings' && /translation/i.test(k));
    await chrome.storage.local.remove(victims);
    return victims.length;
  });
  console.log(`switch to en (${cleared} cached translation keys cleared) — reloading`);
  const t0 = Date.now();
  await page.reload({ waitUntil: 'domcontentloaded', timeout: 60000 });
  const en = await page.waitForFunction(hangulRe => {
    const root = document.getElementById('everyric-root')?.shadowRoot;
    if (!root) return null;
    const prons = Array.from(root.querySelectorAll('.ey-line-pron')).map(el => (el.textContent ?? '').trim()).filter(Boolean);
    const trs = Array.from(root.querySelectorAll('.ey-line-tr')).map(el => (el.textContent ?? '').trim()).filter(Boolean);
    const romaji = prons.filter(p => /[a-z]{2,}/.test(p) && !new RegExp(hangulRe).test(p));
    const english = trs.filter(t => /[A-Za-z]{3,}/.test(t) && !new RegExp(hangulRe).test(t));
    if (romaji.length < 10 || english.length < 10) return null;
    return { romaji: romaji.length, english: english.length, pronSample: romaji[0], trSample: english[0] };
  }, HAS_HANGUL, { timeout: 60000, polling: 1000 }).then(h => h.jsonValue());
  const enSec = Math.round((Date.now() - t0) / 1000);
  check(en.romaji > 10, `EN-1 발음이 romaji로 렌더 (${en.romaji}줄)`, en.pronSample?.slice(0, 50));
  // 레이어에서 오면 수 초, LLM 재생성이면 수십 초 — 25초 안이면 레이어 경로다
  check(en.english > 10 && enSec < 25, `EN-2 영어 번역 — 서버 레이어에서 수신 (${enSec}s)`, en.trSample?.slice(0, 50));
  await page.screenshot({ path: resolve(__dirname, '../e2e-ml-1-en-romaji.png') });
  console.log('screenshot: e2e-ml-1-en-romaji.png');

  // ── EN-3) miraheze 실호출 — 옵션 페이지에서 background로 실메시지 ───────────
  const opts = await ctx.newPage();
  await opts.goto(`chrome-extension://${extId}/src/options.html`, { waitUntil: 'domcontentloaded', timeout: 20000 });
  const miraheze = await opts.evaluate(
    title => chrome.runtime.sendMessage({ type: 'MIRAHEZE_LOOKUP', payload: { title } }),
    wikiTitle,
  );
  const mLines = miraheze?.data?.lines ?? [];
  check(mLines.length > 10, 'EN-3 miraheze 실검색·파싱', {
    pageTitle: miraheze?.data?.pageTitle, lines: mLines.length,
    first: mLines[0], pronLang: miraheze?.data?.pronLang, translationLang: miraheze?.data?.translationLang,
  });
  check(miraheze?.data?.pronLang === 'romaji' && miraheze?.data?.translationLang === 'en', 'EN-3 언어 태그 (romaji/en)');

  // ── EN-4) uiLanguage=en → 옵션 페이지 정적 텍스트 영어 ─────────────────────
  await setSettings(sw, { uiLanguage: 'en' });
  await opts.reload({ waitUntil: 'domcontentloaded', timeout: 20000 });
  await opts.waitForTimeout(1500); // applyStaticI18n 적용 대기
  const optState = await opts.evaluate(() => ({
    sample: document.body.innerText.replace(/\s+/g, ' ').slice(0, 120),
    hasEnglish: /permission|server|allow|host/i.test(document.body.innerText),
    hasHangul: /[가-힣]/.test(document.body.innerText),
  }));
  check(optState.hasEnglish && !optState.hasHangul, 'EN-4 옵션 페이지 uiLanguage=en 렌더', optState.sample);
  await opts.screenshot({ path: resolve(__dirname, '../e2e-ml-3-options-en.png') });
  console.log('screenshot: e2e-ml-3-options-en.png');
} catch (e) {
  failed = true;
  console.log('FAIL (exception):', String(e).slice(0, 500));
} finally {
  await ctx.close();
}
console.log(failed ? '\n=== MULTILANG E2E: FAILED ===' : '\n=== MULTILANG E2E: ALL PASS ===');
process.exit(failed ? 1 : 0);
