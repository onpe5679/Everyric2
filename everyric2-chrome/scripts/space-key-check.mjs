// 패널 입력창에서 스페이스바가 유튜브 재생/정지로 새지 않는지 검증.
// 검색 시트를 열고 제목 입력창에 "a b c"를 타이핑 → 값에 공백 포함 + 영상 재생 유지 확인.
import { chromium } from 'playwright';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import { mkdtempSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const distDir = resolve(__dirname, '../dist');
const videoUrl = process.argv[2] ?? 'https://www.youtube.com/watch?v=Xg-qfsKN2_E';
const userDataDir = mkdtempSync(join(tmpdir(), 'everyric-space-'));

let failed = false;
function check(ok, label, detail) {
  console.log(`${ok ? 'PASS' : 'FAIL'}: ${label}${detail !== undefined ? ' = ' + JSON.stringify(detail) : ''}`);
  if (!ok) failed = true;
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

  await page.evaluate(() => {
    const v = document.querySelector('video.html5-main-video') ?? document.querySelector('video');
    if (v) void v.play().catch(() => {});
  });
  await page.waitForTimeout(800);

  // 검색 시트 열고 제목 입력창에 포커스 → 공백 포함 타이핑
  await page.evaluate(() => {
    const root = document.getElementById('everyric-root').shadowRoot;
    root.querySelector('[title^="가사 다시 검색"]').click();
  });
  await page.waitForTimeout(500);
  await page.evaluate(() => {
    const root = document.getElementById('everyric-root').shadowRoot;
    const input = root.querySelector('.ey-search-form .ey-input');
    input.value = '';
    input.focus();
  });
  await page.keyboard.type('a b c', { delay: 60 });
  await page.waitForTimeout(400);

  const result = await page.evaluate(() => {
    const root = document.getElementById('everyric-root').shadowRoot;
    const input = root.querySelector('.ey-search-form .ey-input');
    const v = document.querySelector('video.html5-main-video') ?? document.querySelector('video');
    return { value: input?.value ?? '', paused: v ? v.paused : null };
  });
  check(result.value === 'a b c', '입력창에 공백 포함 타이핑', result.value);
  check(result.paused === false, '스페이스가 유튜브 재생/정지로 새지 않음 (재생 유지)', result.paused);

  console.log(failed ? 'SPACE KEY CHECK: FAIL' : 'SPACE KEY CHECK: PASS');
  process.exitCode = failed ? 1 : 0;
} catch (e) {
  console.log('SPACE KEY CHECK: ERROR —', String(e).slice(0, 300));
  process.exitCode = 1;
} finally {
  await ctx.close();
}
