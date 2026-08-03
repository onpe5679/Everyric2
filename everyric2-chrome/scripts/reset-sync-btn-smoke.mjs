// #46 항목2 스모크 — resetSyncBtn(구 regenBtn) 위치·라벨·동작 실브라우저 확인.
// 실행: node scripts/reset-sync-btn-smoke.mjs [videoId]
import { chromium } from 'playwright';
import { fileURLToPath } from 'url';
import { dirname, resolve, join } from 'path';
import { mkdtempSync, cpSync } from 'fs';
import { tmpdir } from 'os';
import { execFileSync } from 'child_process';
import { ensureLocalServerPermissionForServerUrl } from './lib/local-server-permission.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const distSrc = resolve(__dirname, '../dist');
const distDir = join(mkdtempSync(join(tmpdir(), 'ey-dist-')), 'dist');
cpSync(distSrc, distDir, { recursive: true });

const SERVER = 'http://127.0.0.1:8000';
const VIDEO = process.argv[2] ?? 'arX83q0oJhM'; // fast+everyric 실곡(이전 라운드 검증에서 확인)
let failed = false;
function check(ok, label, detail) {
  console.log(`${ok ? 'PASS' : 'FAIL'}: ${label}${detail !== undefined ? ' = ' + JSON.stringify(detail) : ''}`);
  if (!ok) failed = true;
}

try {
  execFileSync('taskkill', ['/F', '/IM', 'chrome.exe', '/T'], { stdio: 'ignore' });
} catch { /* 잔류 없음 — 정상 */ }

const userDataDir = mkdtempSync(join(tmpdir(), 'ey-resetbtn-'));
const ctx = await chromium.launchPersistentContext(userDataDir, {
  ignoreDefaultArgs: ['--disable-extensions'],
  headless: false,
  viewport: { width: 1600, height: 950 },
  args: [
    `--disable-extensions-except=${distDir}`, `--load-extension=${distDir}`,
    '--mute-audio', '--autoplay-policy=no-user-gesture-required',
  ],
});
const sw = ctx.serviceWorkers()[0] ?? await ctx.waitForEvent('serviceworker', { timeout: 20000 });
const extId = new URL(sw.url()).host;
await ensureLocalServerPermissionForServerUrl(ctx, sw, extId, SERVER);
await sw.evaluate(async (url) => {
  const cur = (await chrome.storage.local.get('settings')).settings ?? {};
  await chrome.storage.local.set({ settings: { ...cur, serverUrl: url, uiLanguage: 'ko' } });
}, SERVER);

const page = await ctx.newPage();
await page.goto(`https://www.youtube.com/watch?v=${VIDEO}`, { waitUntil: 'domcontentloaded', timeout: 90000 });
await page.waitForFunction(
  () => !!document.getElementById('everyric-root')?.shadowRoot?.querySelector('.ey-line'),
  { timeout: 120000 },
);
await page.waitForTimeout(2500);

// R1: 헤더 액션 줄엔 더 이상 없음
const headerProbe = await page.evaluate(() => {
  const sr = document.getElementById('everyric-root').shadowRoot;
  const actions = sr.querySelector('.ey-actions');
  return {
    headerBtnCount: actions ? actions.querySelectorAll('button').length : -1,
    hasResetInHeader: actions ? !!actions.querySelector('.ey-reset-sync-btn') : null,
  };
});
check(headerProbe.hasResetInHeader === false, 'R1 헤더 액션 줄에 초기화 버튼 없음', headerProbe);

// R2: 풋터에 별점(★) 바로 오른쪽에 존재 + 라벨이 "초기화" 계열
const footerProbe = await page.evaluate(() => {
  const sr = document.getElementById('everyric-root').shadowRoot;
  const footer = sr.querySelector('.ey-footer');
  const feedback = footer?.querySelector('.ey-feedback-btn');
  const resetBtn = footer?.querySelector('.ey-reset-sync-btn');
  if (!footer || !feedback || !resetBtn) {
    return { footer: !!footer, feedback: !!feedback, resetBtn: !!resetBtn };
  }
  // DOM 순서상 feedback 바로 다음 형제인지(사이에 감춰진 요소가 있을 수 있어 nextElementSibling로 확인)
  const isRightAfter = feedback.nextElementSibling === resetBtn;
  return {
    footer: true, feedback: true, resetBtn: true,
    isRightAfter,
    visible: resetBtn.getClientRects().length > 0,
    title: resetBtn.title,
  };
});
check(footerProbe.resetBtn === true && footerProbe.isRightAfter === true && footerProbe.visible === true,
  'R2 풋터의 별점 바로 오른쪽에 초기화 버튼 존재·표시됨', footerProbe);
check(/초기화/.test(footerProbe.title ?? ''), 'R2b 버튼 title이 "초기화" 계열', footerProbe.title);

// R3: 클릭 시 confirmTwice 무장(두 번 눌러야 실제 동작) — 실제 삭제까지는 안 가고 무장 상태만 확인
const armProbe = await page.evaluate(() => {
  const sr = document.getElementById('everyric-root').shadowRoot;
  const btn = sr.querySelector('.ey-reset-sync-btn');
  const beforeArmed = btn.classList.contains('ey-confirm-armed');
  const beforeTitle = btn.title;
  btn.click();
  return { beforeArmed, beforeTitle };
});
await page.waitForTimeout(300);
const afterArm = await page.evaluate(() => {
  const sr = document.getElementById('everyric-root').shadowRoot;
  const btn = sr.querySelector('.ey-reset-sync-btn');
  return { armed: btn.classList.contains('ey-confirm-armed'), title: btn.title };
});
check(armProbe.beforeArmed === false && afterArm.armed === true,
  'R3a 첫 클릭 후 무장(armed) 상태로 전환', { armProbe, afterArm });
check(/삭제|초기화/.test(afterArm.title), 'R3b 무장 중 title이 삭제 확인 문구로 바뀜', afterArm.title);

// 실제로 지우지 않도록 여기서 멈춘다 — 무장 해제만 하고(같은 버튼 다시 클릭하면 실행되므로),
// 대신 다른 곳을 클릭해 confirmTimer가 자연 만료되게 두거나 페이지를 벗어난다.
console.log('INFO: 실제 삭제 실행은 하지 않음(파괴적 동작) — 무장 전환까지만 확인');

await page.screenshot({
  path: 'C:\\Users\\user\\AppData\\Local\\Temp\\claude\\C--DevAT-Everyric2\\ccf2cc19-4a2e-4cc6-9f6d-e92f242a5aa3\\scratchpad\\reset-sync-btn-smoke.png',
}).catch(() => {});
await ctx.close();
console.log(failed ? '\n== 결과: FAIL 있음' : '\n== 결과: 전부 PASS');
process.exit(failed ? 1 : 0);
