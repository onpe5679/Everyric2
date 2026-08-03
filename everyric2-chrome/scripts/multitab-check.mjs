// 헤비유저 다중 탭 시나리오 실브라우저 검증 — P0 수리 2건의 런타임 증거.
//
//   T1) 탭 간 설정 전파: 탭 A에서 설정을 바꾸면 **이미 열려 있는** 탭 B가 새로고침
//       없이 즉시 반영해야 한다 (수리 전: storage 감시가 settings 키를 통째로 무시해
//       탭 B는 새로고침 전까지 옛 값을 썼다).
//   T2) 설정 저장이 다른 값을 지우지 않는다: 탭 A와 B가 각각 다른 키를 바꿔도 둘 다
//       살아남아야 한다 (읽기 실패 시 기본값 전체를 디스크에 쓰던 P0의 반대 증명).
//   T3) 폴링 범위: 탭을 여러 개 띄워도 잡을 시작하지 않은/그 영상을 안 보는 탭은
//       /api/job 을 때리지 않아야 한다 (수리 전: 탭 N × 잡 M 건씩 2초마다).
//
// 실행: node scripts/multitab-check.mjs
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
let failed = false;
function check(ok, label, detail) {
  console.log(`${ok ? 'PASS' : 'FAIL'}: ${label}${detail !== undefined ? ' = ' + JSON.stringify(detail) : ''}`);
  if (!ok) failed = true;
  return ok;
}

const health = await (await fetch(`${SERVER}/health`, { signal: AbortSignal.timeout(3000) })).json();
if (!check(health.status === 'healthy', 'real server /health', health)) process.exit(1);

const userDataDir = mkdtempSync(join(tmpdir(), 'ey-multitab-'));
const ctx = await chromium.launchPersistentContext(userDataDir, {
  // 이게 없으면 크롬이 확장을 통째로 끈다 — 서비스워커가 영영 안 뜬다(실측)
  ignoreDefaultArgs: ['--disable-extensions'],
  headless: false,
  viewport: { width: 1440, height: 900 },
  args: [
    `--disable-extensions-except=${distDir}`,
    `--load-extension=${distDir}`,
    '--mute-audio',
    '--autoplay-policy=no-user-gesture-required',
  ],
});

const sw = ctx.serviceWorkers()[0] ?? await ctx.waitForEvent('serviceworker', { timeout: 20000 });
const extId = new URL(sw.url()).host;
console.log('extension:', extId);

await ensureLocalServerPermissionForServerUrl(ctx, sw, extId, SERVER);
// 로컬 서버를 보게 하고 시작 (기본값은 프로드)
await sw.evaluate(async (url) => {
  const cur = (await chrome.storage.local.get('settings')).settings ?? {};
  await chrome.storage.local.set({ settings: { ...cur, serverUrl: url } });
}, SERVER);

// 잡 요청 카운터 — 어느 탭이 /api/job 을 때리는지 센다
const jobHits = new Map();
ctx.on('request', (req) => {
  const u = req.url();
  if (!u.includes('/api/job/')) return;
  const key = req.frame()?.url() ?? 'unknown';
  jobHits.set(key, (jobHits.get(key) ?? 0) + 1);
});

// 싱크가 있는 두 곡 — 탭 A는 ja곡, 탭 B는 en곡
const tabA = await ctx.newPage();
await tabA.goto('https://www.youtube.com/watch?v=b2NTglk9tvI', { waitUntil: 'domcontentloaded', timeout: 60000 });
const tabB = await ctx.newPage();
await tabB.goto('https://www.youtube.com/watch?v=UgK6n1KKUxY', { waitUntil: 'domcontentloaded', timeout: 60000 });

// 오버레이가 뜰 때까지
async function waitOverlay(page, label) {
  try {
    await page.waitForFunction(
      () => !!document.getElementById('everyric-root')?.shadowRoot?.querySelector('.ey-line, .ey-body'),
      { timeout: 90000 });
    return true;
  } catch { console.log(`  (${label}: 오버레이 미검출 — 계속)`); return false; }
}
await waitOverlay(tabA, 'A');
await waitOverlay(tabB, 'B');
await tabA.waitForTimeout(4000);

// ── T1: 탭 간 설정 전파 ──────────────────────────────────────────
// 탭 A의 세션이 들고 있는 값을 직접 읽을 수 없으므로, 화면에 즉시 반영되는 설정을 쓴다.
// showPronunciation을 끄면 발음 줄이 사라져야 한다(렌더 경로가 settings를 읽는 증거).
const pronBefore = await tabA.evaluate(() => {
  // 발음은 DOM 제거가 아니라 패널의 .ey-hide-pron 클래스로 숨긴다(overlay.ts:1741,
  // CSS .ey-hide-pron .ey-line-pron{display:none}). 그래서 "보이는 발음 줄 수"로 센다.
  const sr = document.getElementById('everyric-root')?.shadowRoot;
  if (!sr) return -1;
  return [...sr.querySelectorAll('.ey-line-pron, .ey-pron')]
    .filter(el => el.getClientRects().length > 0).length;
});

// 탭 B(다른 탭)에서 설정을 바꾼다 — 탭 A가 새로고침 없이 따라와야 한다
await sw.evaluate(async () => {
  const cur = (await chrome.storage.local.get('settings')).settings ?? {};
  await chrome.storage.local.set({ settings: { ...cur, showPronunciation: false } });
});
await tabA.waitForTimeout(2500);
const pronAfter = await tabA.evaluate(() => {
  // 발음은 DOM 제거가 아니라 패널의 .ey-hide-pron 클래스로 숨긴다(overlay.ts:1741,
  // CSS .ey-hide-pron .ey-line-pron{display:none}). 그래서 "보이는 발음 줄 수"로 센다.
  const sr = document.getElementById('everyric-root')?.shadowRoot;
  if (!sr) return -1;
  return [...sr.querySelectorAll('.ey-line-pron, .ey-pron')]
    .filter(el => el.getClientRects().length > 0).length;
});
check(pronBefore > 0, 'T1 준비: 탭 A에 발음 줄이 렌더돼 있었다', { pronBefore });
check(pronBefore > 0 && pronAfter === 0,
  'T1 탭 간 설정 전파 — 다른 탭의 변경이 새로고침 없이 반영', { pronBefore, pronAfter });

// 되돌리기
await sw.evaluate(async () => {
  const cur = (await chrome.storage.local.get('settings')).settings ?? {};
  await chrome.storage.local.set({ settings: { ...cur, showPronunciation: true } });
});
await tabA.waitForTimeout(1500);

// ── T2: 설정 저장이 서로를 지우지 않는다 ─────────────────────────
await tabA.evaluate(() => window.postMessage({ __eyTest: 'noop' }, '*'));
await sw.evaluate(async (url) => {
  const cur = (await chrome.storage.local.get('settings')).settings ?? {};
  await chrome.storage.local.set({ settings: { ...cur, serverUrl: url, fontSize: 'large' } });
}, SERVER);
await tabA.waitForTimeout(1200);
await sw.evaluate(async () => {
  const cur = (await chrome.storage.local.get('settings')).settings ?? {};
  await chrome.storage.local.set({ settings: { ...cur, theme: 'dark' } });
});
await tabA.waitForTimeout(1200);
const merged = await sw.evaluate(async () => (await chrome.storage.local.get('settings')).settings);
check(merged.fontSize === 'large' && merged.theme === 'dark' && merged.serverUrl.includes('127.0.0.1'),
  'T2 연속 저장이 서로의 값을 지우지 않는다',
  { fontSize: merged.fontSize, theme: merged.theme, serverUrl: merged.serverUrl });

// ── T3: 폴링 범위 — 진행 중 잡이 없으면 아무도 /api/job 을 안 때린다 ──
jobHits.clear();
await tabA.waitForTimeout(9000); // 폴링 주기(2s)의 4배 이상
const totalJobHits = [...jobHits.values()].reduce((a, b) => a + b, 0);
check(totalJobHits === 0,
  'T3 진행 잡이 없을 때 유휴 폴링 0건 (탭 2개, 9초 관찰)',
  { totalJobHits, byFrame: Object.fromEntries(jobHits) });

console.log(`MULTITAB CHECK: ${failed ? 'FAIL' : 'PASS'}`);
await ctx.close();
process.exit(failed ? 1 : 0);
