// 공지사항 다국어화 클라이언트 검증 — GET /api/notices 응답을 라우트 가로채기로 통제해
// (a) 구서버(translations 필드 자체 없음) 폴백, (b) 신서버(translations 있음)의 언어별
// 선택을 실브라우저로 확인한다. 서버 재기동 불필요(네트워크 계층만 통제).
// taskkill 없음, 고정 프로필.
import { chromium } from 'playwright';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import { mkdirSync } from 'fs';
import { ensureLocalServerPermissionForServerUrl } from './lib/local-server-permission.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const distDir = resolve(__dirname, '../dist');
const SERVER = 'http://127.0.0.1:8000';
const VIDEO = 'arX83q0oJhM'; // 실곡, 로컬 DB에 싱크 있음(패널이 정상적으로 열려야 헤더 버튼 접근 가능)

let failed = false;
function check(ok, label, detail) {
  console.log(`${ok ? 'PASS' : 'FAIL'}: ${label}${detail !== undefined ? ' = ' + JSON.stringify(detail) : ''}`);
  if (!ok) failed = true;
}
function info(label, detail) { console.log(`INFO: ${label} = ${JSON.stringify(detail)}`); }

const profileDir = 'C:\\Users\\user\\AppData\\Local\\Temp\\claude\\C--DevAT-Everyric2\\ccf2cc19-4a2e-4cc6-9f6d-e92f242a5aa3\\scratchpad\\ey-notices-i18n-profile';
mkdirSync(profileDir, { recursive: true });

const ctx = await chromium.launchPersistentContext(profileDir, {
  ignoreDefaultArgs: ['--disable-extensions'],
  headless: false,
  viewport: { width: 1400, height: 900 },
  args: [
    `--disable-extensions-except=${distDir}`, `--load-extension=${distDir}`,
    '--mute-audio', '--autoplay-policy=no-user-gesture-required',
  ],
});
const sw = ctx.serviceWorkers()[0] ?? await ctx.waitForEvent('serviceworker', { timeout: 20000 });
const extId = new URL(sw.url()).host;
await ensureLocalServerPermissionForServerUrl(ctx, sw, extId, SERVER);

// 신서버 응답(translations 포함)을 흉내낸다 — 실제 마이그레이션·API 코드는 pytest로
// 이미 검증했으니, 여기서는 "이 정확한 JSON 모양을 클라이언트가 어떻게 소비하는가"만 잰다.
const NEW_SERVER_NOTICE = {
  id: 999, title: '점검 안내', body: '오늘 자정 점검', level: 'info',
  created_at: new Date().toISOString(), ends_at: null,
  translations: {
    en: { title: 'Maintenance notice', body: 'Maintenance tonight at midnight' },
    ja: { title: 'メンテナンスのお知らせ', body: '今夜0時にメンテナンス' },
  },
};
// 구서버 응답 — translations 필드 자체가 없다(존재하지 않는 키, undefined가 아니라
// JSON에 아예 없는 것과 같다 — JSON.stringify가 undefined 필드를 자동으로 뺀다는
// 점에 기대지 않고 명시적으로 필드를 안 넣은 객체를 쓴다).
const OLD_SERVER_NOTICE = {
  id: 998, title: '구서버 공지', body: '구서버 본문', level: 'info',
  created_at: new Date().toISOString(), ends_at: null,
};

let currentMock = null;
await ctx.route('**/api/notices', async (route) => {
  if (route.request().method() !== 'GET' || !currentMock) { await route.continue(); return; }
  await route.fulfill({
    status: 200, contentType: 'application/json',
    body: JSON.stringify({ notices: [currentMock] }),
  });
});

const page = await ctx.newPage();
const consoleErrors = [];
page.on('console', (msg) => { if (msg.type() === 'error') consoleErrors.push(msg.text()); });

async function setUiLanguage(lang) {
  await sw.evaluate(async (l) => {
    const cur = (await chrome.storage.local.get('settings')).settings ?? {};
    await chrome.storage.local.set({ settings: { ...cur, uiLanguage: l } });
  }, lang);
}
async function openNoticesAndRead() {
  // 매번 새로 열기 위해 패널을 리로드해 헤더부터 다시 접근한다(언어 변경을 확실히 반영)
  await page.reload({ waitUntil: 'domcontentloaded' });
  await page.waitForFunction(
    () => !!document.getElementById('everyric-root')?.shadowRoot?.querySelector('.ey-line'),
    { timeout: 60000 },
  );
  await page.waitForTimeout(1000);
  const clicked = await page.evaluate(() => {
    const root = document.getElementById('everyric-root')?.shadowRoot;
    const btn = root?.querySelector('.ey-notices-btn');
    if (!btn || btn.style.display === 'none') return false;
    btn.click();
    return true;
  });
  if (!clicked) return { opened: false };
  await page.waitForTimeout(1000);
  return page.evaluate(() => {
    const root = document.getElementById('everyric-root')?.shadowRoot;
    const item = root?.querySelector('.ey-notice-item');
    return {
      opened: true,
      title: item?.querySelector('.ey-notice-title')?.textContent ?? null,
      body: item?.querySelector('.ey-notice-body')?.textContent ?? null,
    };
  });
}

console.log('navigating to', VIDEO);
await page.goto(`https://www.youtube.com/watch?v=${VIDEO}`, { waitUntil: 'domcontentloaded', timeout: 60000 });
await page.waitForFunction(
  () => !!document.getElementById('everyric-root')?.shadowRoot?.querySelector('.ey-line'),
  { timeout: 60000 },
);

// ── (a) 구서버 폴백: translations 필드 자체가 없어도 title/body가 정상 표시되는가 ──
currentMock = OLD_SERVER_NOTICE;
await setUiLanguage('en'); // 구서버는 언어 설정과 무관하게 항상 title/body만 준다
const oldServerResult = await openNoticesAndRead();
info('(a) 구서버 응답(translations 없음, uiLanguage=en)', oldServerResult);
check(oldServerResult.opened, '(a) 공지 시트가 열림', oldServerResult);
check(oldServerResult.title === '구서버 공지' && oldServerResult.body === '구서버 본문',
  '(a) 구서버 폴백 — translations 없어도 title/body 그대로 표시(크래시·빈 화면 없음)', oldServerResult);

// ── (b) 신서버 + uiLanguage=en → en 번역 선택 ──
currentMock = NEW_SERVER_NOTICE;
await setUiLanguage('en');
const enResult = await openNoticesAndRead();
info('(b) 신서버 + uiLanguage=en', enResult);
check(enResult.title === 'Maintenance notice' && enResult.body === 'Maintenance tonight at midnight',
  '(b) uiLanguage=en이면 en 번역이 선택된다', enResult);

// ── (c) 신서버 + uiLanguage=ja → ja 번역 선택 ──
await setUiLanguage('ja');
const jaResult = await openNoticesAndRead();
info('(c) 신서버 + uiLanguage=ja', jaResult);
check(jaResult.title === 'メンテナンスのお知らせ' && jaResult.body === '今夜0時にメンテナンス',
  '(c) uiLanguage=ja이면 ja 번역이 선택된다', jaResult);

// ── (d) 신서버 + uiLanguage=ko → translations에 ko가 없으므로 기본(title/body)로 폴백 ──
await setUiLanguage('ko');
const koResult = await openNoticesAndRead();
info('(d) 신서버 + uiLanguage=ko(translations에 ko 없음)', koResult);
check(koResult.title === '점검 안내' && koResult.body === '오늘 자정 점검',
  '(d) translations에 없는 언어는 기본(한국어) title/body로 폴백', koResult);

console.log(`\n총 콘솔 예외: ${consoleErrors.length}건`);
consoleErrors.forEach((e, i) => console.log(`  [${i}]`, e.slice(0, 200)));

await ctx.close().catch(() => {});
console.log(failed ? '\n== 결과: FAIL 있음' : '\n== 결과: 전부 PASS');
process.exit(failed ? 1 : 0);
