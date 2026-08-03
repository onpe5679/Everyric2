// 노트 위 음절 텍스트가 **설정과 무관하게 항상** 그려지는지 실측.
//
// 운영자 실사용 제보: "가라오케 노트 하나하나마다는 절대 아무것도 표시가 안 되면 안 돼".
// 원인은 두 겹이었다 — (a) 코너 버튼 순환 함정으로 pitchPronPosition='off'가 **저장**됐고,
// (b) 'off'면 노트 위 텍스트까지 함께 꺼졌다. (a)는 이미 고쳤지만 이미 저장된 사용자는
// 구제되지 않는다. 그래서 (b)를 설계로 끊었다: 노트 텍스트는 설정에서 분리해 항상 표시.
//
// 재는 것: pitchPronPosition의 **모든 값**에서 노트 영역에 글자 픽셀이 있는가.
// 캔버스라 DOM으로 못 보므로 오선 위쪽(노트 바 + 그 아래 음절) 영역의 잉크를 센다.
//
// 실행: node scripts/lane-note-text-check.mjs <syncedVideoId>
import { chromium } from 'playwright';
import { fileURLToPath } from 'url';
import { dirname, resolve, join } from 'path';
import { cpSync, mkdtempSync, readFileSync } from 'fs';
import { tmpdir } from 'os';
import { ensureLocalServerPermissionForServerUrl } from './lib/local-server-permission.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const distDir = mkdtempSync(join(tmpdir(), 'everyric-dist-snap-'));
cpSync(resolve(__dirname, '../dist'), distDir, { recursive: true });
JSON.parse(readFileSync(join(distDir, 'manifest.json'), 'utf8'));

const videoId = process.argv[2];
// 서버는 인자로 바꿀 수 있다 — 프로드 읽기 검증에 쓴다(생성 없이 «이미 있는 싱크»만 본다).
// 프로드는 manifest의 **필수** host_permission이라 별도 승인 흐름이 필요 없다.
const SERVER_URL = process.argv[3] ?? 'http://127.0.0.1:8000';
const IS_LOCAL = /127\.0\.0\.1|localhost/.test(SERVER_URL);
if (!videoId) {
  console.log('사용법: node scripts/lane-note-text-check.mjs <syncedVideoId> [serverUrl]');
  process.exit(2);
}

let failed = false;
const check = (ok, label, detail) => {
  console.log(`${ok ? 'PASS' : 'FAIL'}: ${label}${detail !== undefined ? ' = ' + JSON.stringify(detail) : ''}`);
  if (!ok) failed = true;
};

/** 레인 캔버스에서 «노트 영역»(위 55%)의 잉크 픽셀 수 — 음절 글자가 있으면 확 는다 */
const MEASURE = `(() => {
  const c = document.querySelector('.ey-main-lane')
    ?? document.getElementById('everyric-root')?.shadowRoot?.querySelector('.ey-main-lane');
  if (!c || !c.width) return null;
  const g = c.getContext('2d');
  const h = Math.floor(c.height * 0.55);
  const d = g.getImageData(0, 0, c.width, h).data;
  let ink = 0;
  for (let i = 3; i < d.length; i += 4) if (d[i] > 40) ink++;
  return { w: c.width, h: c.height, noteInk: ink };
})()`;

const userDataDir = mkdtempSync(join(tmpdir(), 'everyric-note-text-'));
const ctx = await chromium.launchPersistentContext(userDataDir, {
  ignoreDefaultArgs: ['--disable-extensions'],
  headless: false,
  viewport: { width: 1440, height: 900 },
  args: [
    `--disable-extensions-except=${distDir}`, `--load-extension=${distDir}`,
    '--mute-audio', '--autoplay-policy=no-user-gesture-required', '--window-position=30,30',
  ],
});

try {
  const sw = ctx.serviceWorkers()[0] ?? await ctx.waitForEvent('serviceworker', { timeout: 15000 });
  // 로컬 주소만 optional_host_permissions라 승인 흐름이 필요하다
  if (IS_LOCAL) {
    await ensureLocalServerPermissionForServerUrl(ctx, sw, new URL(sw.url()).host, SERVER_URL);
  }
  const setSettings = (patch) => sw.evaluate(async (p) => {
    const cur = (await chrome.storage.local.get('settings')).settings ?? {};
    await chrome.storage.local.set({ settings: { ...cur, ...p } });
  }, patch);

  // **메인 창 레인**으로 잰다 — PiP를 열지 않아도 같은 렌더러·같은 캔버스다.
  await setSettings({
    serverUrl: SERVER_URL, uiLanguage: 'ko', theme: 'dark',
    modMainLane: true, mainLanePos: 'bottom', pitchPronPosition: 'off',
  });

  const page = ctx.pages()[0] ?? await ctx.newPage();
  await page.goto(`https://www.youtube.com/watch?v=${videoId}`, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForSelector('#everyric-root', { state: 'attached', timeout: 30000 });
  await page.waitForFunction(() => {
    const r = document.getElementById('everyric-root')?.shadowRoot;
    return (r?.querySelectorAll('.ey-line').length ?? 0) > 0;
  }, null, { timeout: 45000, polling: 1000 });
  await page.evaluate(() => {
    const v = document.querySelector('video.html5-main-video') ?? document.querySelector('video');
    if (v) { v.currentTime = 30; void v.play().catch(() => {}); }
  });
  await page.waitForTimeout(2500);

  // 노트 텍스트의 **출처**를 판정한다 — 잉크 양만으로는 «원문 음절»인지 «발음»인지
  // 구분되지 않는다. 레인이 실제로 쓴 세그를 패널에서 직접 물어본다.
  const source = await page.evaluate(() => {
    const root = document.getElementById('everyric-root')?.shadowRoot;
    const lines = [...(root?.querySelectorAll('.ey-line') ?? [])].slice(0, 6);
    return lines.map(el => ({
      text: (el.querySelector('.ey-line-text')?.textContent ?? el.textContent ?? '').slice(0, 40),
      pron: (el.querySelector('.ey-line-pron')?.textContent ?? '').slice(0, 40),
    }));
  });
  console.log('INFO: 화면 줄 샘플 =', JSON.stringify(source.slice(0, 3)));

  const results = {};
  for (const pos of ['off', 'bottom', 'center', 'both']) {
    await setSettings({ pitchPronPosition: pos });
    await page.waitForTimeout(1500);
    const m = await page.evaluate(MEASURE);
    results[pos] = m;
    check(m !== null && m.noteInk > 500,
      `pitchPronPosition='${pos}' 에서도 노트 영역에 글자가 그려짐`, m);
  }

  // ── 노트 텍스트가 설정을 안 탄다는 것을 «잉크 양»으로 확인한다.
  //
  // 비교 상대는 'both'가 **아니라 'center'다**. 'bottom'·'both'는 발음 줄이 세로 공간을
  // 차지해 오선이 줄고, 그만큼 아래 가사 줄(굵은 원문 토큰)이 이 측정 밴드(위 55%) 안으로
  // 밀려 올라온다 — 노트 텍스트와 무관한 잉크가 섞이는 것이다. 'center'는 반투명 중앙
  // 오버레이라 staffH 계산에 관여하지 않아(pitch-lane.ts renderPronFallback 호출부 주석)
  // 'off'와 **레이아웃이 완전히 같다**. 차이는 오버레이 잉크 하나뿐이라 정규화가 성립한다.
  //
  // 예전 문턱("off 대비 both < 1.7")은 그 레이아웃 밀림 때문에 못 쓴다. 실측(2026-08-04,
  // 5곡: K0WRNxEgnoE·UnIhRpIT7nc·M7VSEZOQIlg·m1A-S-PzU6E·BiQs9ABhT7U)에서 정상 동작인데도
  // both/off가 1.13~1.92까지 벌어져 1.7을 넘겼다. 게다가 이 지표는 «잡으려던 결함»과
  // 구분도 못 한다 — 노트 텍스트를 off/bottom에서 안 그리던 옛 동작을 되살려 같은 5곡을
  // 재면 both/off가 1.80~2.23이라 정상 범위와 겹친다. 어떤 값을 넣어도 오탐 아니면 미탐이다.
  //
  // off/center는 갈린다: 정상 0.796·0.841·0.878·0.932·0.979(실브라우저 5곡) vs
  // 결함 0.493~0.689(같은 5곡, 결함 재현 렌더). 문턱 0.72는 그 사이에 있다 —
  // 정상 최저(0.796)와 0.076, 결함 최고(0.689)와 0.031 여유.
  // 위쪽 한계는 두지 않는다 — center는 구조적으로 off + 오버레이라 1을 넘을 수 없고,
  // 넘는다면 그건 오버레이 쪽 결함이지 이 검사의 대상이 아니다.
  const a = results.off?.noteInk ?? 0;
  const c = results.center?.noteInk ?? 0;
  const ratio = c > 0 ? a / c : 0;
  check(ratio >= 0.72,
    "노트 영역 잉크가 설정값에 좌우되지 않음('off'가 'center' 대비 안 줄어듦)",
    { off: a, center: c, ratio: +ratio.toFixed(3) });

  await page.screenshot({ path: resolve(__dirname, '../lane-note-text-check.png') });
  console.log('screenshot: lane-note-text-check.png');
  console.log(failed ? 'LANE NOTE TEXT CHECK: FAIL' : 'LANE NOTE TEXT CHECK: PASS');
  process.exitCode = failed ? 1 : 0;
} catch (e) {
  console.log('LANE NOTE TEXT CHECK: ERROR —', String(e).slice(0, 400));
  process.exitCode = 1;
} finally {
  await ctx.close();
}
