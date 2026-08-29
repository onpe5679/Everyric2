// 영상 자막 모듈이 «번역 데이터 변화»를 따라 다시 그리는지 — 실사용 제보 2026-08-04
// ("유튜브 자막 오버레이 번역이 안 보인다")의 재현·수리 검증.
//
// 기전: content는 번역을 라인 객체에 **제자리로**(mutation) 싣고 지운다. 패널은
// refreshTranslations 방송을 받아 다시 그렸지만 영상 자막 모듈은 그 방송의 수신자가
// 아니었고, 자기 render는 `index !== currentIndex`(줄이 넘어갈 때)에만 돈다. 그래서
// **일시정지 중에는 현재 줄의 번역이 영영 갱신되지 않았다.**
//
//   T0) 기전 확인 — 자막 호스트가 플레이어 안에 있고 번역 줄이 실제로 그려져 있다
//   T1) 일시정지한 채 번역 언어를 바꾸면 자막의 번역 줄이 따라 바뀐다  ← 본 검사
//
// T1은 «일시정지»가 핵심이다. 재생 중이면 줄이 넘어가며 저절로 갱신돼 수리 전에도
// 통과해 버린다 — 결함이 살아 있을 때 반드시 실패하도록 시간축을 멈춰 놓고 잰다.
//
// 실행: node scripts/video-caption-refresh-check.mjs [videoId]
// 사전 조건: 실서버 127.0.0.1:8000 (localhost는 IPv6 스톨로 요청당 2초),
//            그 서버에 **한국어 번역이 실린** 싱크가 있는 곡.
import { chromium } from 'playwright';
import { fileURLToPath } from 'url';
import { dirname, resolve, join } from 'path';
import { mkdtempSync, mkdirSync, cpSync } from 'fs';
import { tmpdir } from 'os';
import { ensureLocalServerPermissionForServerUrl } from './lib/local-server-permission.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const distSrc = resolve(__dirname, '../dist');
const distDir = join(mkdtempSync(join(tmpdir(), 'ey-dist-')), 'dist');
cpSync(distSrc, distDir, { recursive: true });

const SERVER = 'http://127.0.0.1:8000';
const VIDEO = process.argv[2] ?? 'arX83q0oJhM';
let failed = false;
function check(ok, label, detail) {
  console.log(`${ok ? 'PASS' : 'FAIL'}: ${label}${detail !== undefined ? ' = ' + JSON.stringify(detail) : ''}`);
  if (!ok) failed = true;
  return ok;
}
function info(label, detail) { console.log(`INFO: ${label} = ${JSON.stringify(detail)}`); }

const health = await (await fetch(`${SERVER}/health`, { signal: AbortSignal.timeout(3000) })).json();
if (!check(health.status === 'healthy', 'real server /health', health)) process.exit(1);

// taskkill 금지(팀리드 공지 2026-08-04) — 이 스크립트 전용 고정 프로필로 병렬 검수와 격리한다
const userDataDir = join(tmpdir(), 'ey-vc-refresh-persist');
mkdirSync(userDataDir, { recursive: true });
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
  await chrome.storage.local.set({
    settings: {
      ...cur, serverUrl: url, uiLanguage: 'ko',
      videoCaptions: true, showTranslation: true, translationLanguage: 'ko',
    },
  });
}, SERVER);

const page = await ctx.newPage();
await page.goto(`https://www.youtube.com/watch?v=${VIDEO}`, { waitUntil: 'domcontentloaded', timeout: 90000 });
await page.waitForFunction(
  () => !!document.getElementById('everyric-root')?.shadowRoot?.querySelector('.ey-line'),
  { timeout: 120000 });

// 자막은 시간 구동이라 재생해야 첫 줄이 그려진다. 번역이 서버에서 늦게 붙는 경우까지
// 감안해 **번역 줄이 실제로 채워질 때까지** 조건 폴링한다(고정 대기는 환경 실패를
// 제품 실패로 둔갑시킨다 — ui-wave-check의 같은 교훈).
const readCaption = () => page.evaluate(() => {
  const host = document.querySelector('#movie_player .ey-video-caption');
  const v = document.querySelector('video');
  const media = v ? { paused: v.paused, currentTime: Number(v.currentTime.toFixed(2)) } : null;
  if (!host) return { present: false, media };
  const tr = host.querySelector('.ey-vc-tr');
  return {
    present: true,
    inPlayer: !!host.closest('#movie_player'),
    line: (host.querySelector('.ey-vc-line')?.textContent ?? '').trim().slice(0, 40),
    trShown: !!tr && tr.style.display !== 'none',
    tr: (tr?.textContent ?? '').trim().slice(0, 60),
    media,
  };
});

await page.evaluate(() => {
  const v = document.querySelector('video');
  if (v) { v.muted = true; v.currentTime = Math.max(0, (v.currentTime || 0) + 30); void v.play(); }
});
const DEADLINE = Date.now() + 25000;
let cap = await readCaption();
while (Date.now() < DEADLINE && !(cap.present && cap.trShown && cap.tr)) {
  await page.waitForTimeout(500);
  cap = await readCaption();
}

check(cap.present && cap.inPlayer, 'T0a 자막 호스트가 플레이어 안에 있음', cap);
if (!check(cap.trShown && !!cap.tr, 'T0b 번역 줄이 실제로 그려져 있음(기전 확인)', cap)) {
  console.log('INFO: 이 곡에 한국어 번역이 없으면 T1은 원리적으로 성립하지 않는다 — 곡을 바꿔 재실행할 것');
  await ctx.close();
  process.exit(1);
}

// ── 시간축을 멈춘다. 여기서부터 줄 전환은 일어나지 않는다 ────────────────
await page.evaluate(() => { document.querySelector('video')?.pause(); });
await page.waitForTimeout(500);
const before = await readCaption();
info('T1 기준(일시정지 직후)', before);
check(before.media?.paused === true, 'T1a 일시정지 상태 확보', before.media);

// 번역 언어 전환 → content가 clearTranslations로 실린 번역을 지우고 새 언어를 요청한다.
// 둘 중 무엇이 오든 **번역 줄의 내용은 달라져야** 한다(비워지거나 새 언어로 교체).
await sw.evaluate(async () => {
  const cur = (await chrome.storage.local.get('settings')).settings ?? {};
  await chrome.storage.local.set({ settings: { ...cur, translationLanguage: 'en' } });
});
await page.waitForTimeout(4000);
const after = await readCaption();
info('T1 전환 후', after);

check(after.media?.paused === true, 'T1b 측정 내내 일시정지 유지(줄 전환 없음)', after.media);
check(before.line === after.line, 'T1c 같은 줄을 보고 있음(비교 대상 동일)', { before: before.line, after: after.line });

// **«비워짐»은 통과가 아니다.** 언어 전환 그 자체(설정 patch)는 applyDisplay를 태우므로
// 수리 없이도 옛 언어 번역이 지워지기까지는 된다 — 역검증에서 실제로 그렇게 관측됐다
// (수리 무력화 상태: "사라지질 않아" → ""). 갭은 그 **다음 단계**, 새 언어 번역이
// 서버에서 도착했을 때 화면을 채우는 지점이다. 그때는 설정이 안 바뀌므로 applyDisplay가
// 불리지 않고, refreshTranslations 방송만 온다. 그래서 «새 내용으로 채워졌는가»를 묻는다.
check(after.tr !== before.tr && !!after.tr,
  'T1 일시정지 중 도착한 새 언어 번역이 자막에 채워짐',
  { before: before.tr, after: after.tr });

await page.screenshot({ path: join(__dirname, '..', 'video-caption-refresh-check.png') }).catch(() => {});
await ctx.close();
console.log(failed ? '\n== 결과: FAIL 있음' : '\n== 결과: 전부 PASS');
process.exit(failed ? 1 : 0);
