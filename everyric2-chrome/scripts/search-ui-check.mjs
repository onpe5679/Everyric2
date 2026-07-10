// 수동 검색 UI 검증: 헤더 검색 버튼 → 후보 리스트(보카로 위키/LRCLIB) 표시 → 후보 선택 시 가사 교체
// 사전 조건: 실서버(:8000, vocaro 인덱스 포함). 실행: node scripts/search-ui-check.mjs [videoUrl]
import { chromium } from 'playwright';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import { mkdtempSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const distDir = resolve(__dirname, '../dist');
const videoUrl = process.argv[2] ?? 'https://www.youtube.com/watch?v=Xg-qfsKN2_E';
const userDataDir = mkdtempSync(join(tmpdir(), 'everyric-search-'));

let failed = false;
function check(ok, label, detail) {
  console.log(`${ok ? 'PASS' : 'FAIL'}: ${label}${detail !== undefined ? ' = ' + JSON.stringify(detail) : ''}`);
  if (!ok) failed = true;
  return ok;
}

const ctx = await chromium.launchPersistentContext(userDataDir, {
  channel: process.env.EVERYRIC_E2E_CHANNEL ?? 'msedge',
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

try {
  const page = ctx.pages()[0] ?? await ctx.newPage();
  await page.goto(videoUrl, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForSelector('#everyric-root', { state: 'attached', timeout: 30000 });
  await page.waitForFunction(() => {
    const root = document.getElementById('everyric-root')?.shadowRoot;
    return (root?.querySelectorAll('.ey-line').length ?? 0) > 0;
  }, null, { timeout: 45000, polling: 1000 });

  // 헤더 검색 버튼 클릭 → 검색 상태 진입 + 초기값(현재 곡)으로 자동 검색
  await page.evaluate(() => {
    const root = document.getElementById('everyric-root').shadowRoot;
    root.querySelector('[title^="가사 다시 검색"]').click();
  });
  await page.waitForFunction(() => {
    const root = document.getElementById('everyric-root')?.shadowRoot;
    return (root?.querySelectorAll('.ey-result-item').length ?? 0) > 0;
  }, null, { timeout: 20000, polling: 500 });

  const results = await page.evaluate(() => {
    const root = document.getElementById('everyric-root').shadowRoot;
    return [...root.querySelectorAll('.ey-result-item')].map(el => ({
      src: el.querySelector('.ey-result-src')?.textContent ?? '',
      title: el.querySelector('.ey-result-title')?.textContent ?? '',
      meta: el.querySelector('.ey-result-meta')?.textContent ?? '',
    }));
  });
  check(results.length > 0, `후보 리스트 표시 (${results.length}건)`, results.slice(0, 4));
  check(results.some(r => r.src.includes('보카로')), '보카로 위키 후보 포함');

  await page.screenshot({ path: resolve(__dirname, '../search-ui.png') });
  console.log('screenshot: search-ui.png');

  // 보카로 위키 후보 선택 → 가사 교체 확인
  await page.evaluate(() => {
    const root = document.getElementById('everyric-root').shadowRoot;
    const items = [...root.querySelectorAll('.ey-result-item')];
    (items.find(el => el.querySelector('.ey-result-src')?.textContent?.includes('보카로')) ?? items[0]).click();
  });
  await page.waitForFunction(() => {
    const root = document.getElementById('everyric-root')?.shadowRoot;
    return (root?.querySelectorAll('.ey-line').length ?? 0) > 0;
  }, null, { timeout: 20000, polling: 500 });
  const after = await page.evaluate(() => {
    const root = document.getElementById('everyric-root').shadowRoot;
    return {
      lines: root.querySelectorAll('.ey-line').length,
      badge: root.querySelector('.ey-source')?.textContent ?? '',
    };
  });
  check(after.lines > 10 && after.badge.includes('보카로'), '후보 선택으로 가사 교체', after);

  console.log(failed ? 'SEARCH UI CHECK: FAIL' : 'SEARCH UI CHECK: PASS');
  process.exitCode = failed ? 1 : 0;
} catch (e) {
  console.log('SEARCH UI CHECK: ERROR —', String(e).slice(0, 300));
  process.exitCode = 1;
} finally {
  await ctx.close();
}
