// PiP 글자 외곽선(streamTextOutline) 실측 — 운영자 제보 "크로마키+외곽선을 켰는데
// PiP 안 글자에 외곽선이 안 보인다".
//
// PiP 창에는 글자가 세 종류로 그려진다. 셋을 따로 재야 «어디가 빠졌는지»가 나온다:
//   1) 가사창 줄        — 패널 Shadow DOM 안의 .ey-line (CSS text-shadow)
//   2) 단축 표시        — PiP 문서 light DOM의 .ey-pip-stage (CSS text-shadow)
//   3) 가라오케 레인    — 캔버스 fillText (CSS가 원리상 안 닿는다 → strokeText 필요)
//
// 재는 법: 1·2는 getComputedStyle의 textShadow 문자열, 3은 캔버스 픽셀에서 «외곽선 색»
// (다크 테마 = 거의 검정) 잉크를 센다. 레인 글자·오선은 밝은 회색/주황이라 검정 잉크는
// 외곽선이 그려졌을 때만 나온다.
//
// 실행: node scripts/pip-text-outline-check.mjs <syncedVideoId> [--dist dir] [--profile dir]
import { chromium } from 'playwright';
import { fileURLToPath } from 'url';
import { dirname, resolve, join } from 'path';
import { cpSync, mkdirSync, mkdtempSync, readFileSync } from 'fs';
import { tmpdir } from 'os';
import { ensureLocalServerPermissionForServerUrl } from './lib/local-server-permission.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const arg = (n, d) => { const i = process.argv.indexOf(`--${n}`); return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : d; };
const videoId = process.argv[2];
const LOCAL_SERVER_URL = 'http://127.0.0.1:8000';
if (!videoId) {
  console.log('사용법: node scripts/pip-text-outline-check.mjs <syncedVideoId> [--dist dir] [--profile dir]');
  process.exit(2);
}
// dist는 스냅샷해서 쓴다 — 공유 dist/를 다시 빌드하지 않고 격리 빌드를 가리킬 수 있다
const distDir = mkdtempSync(join(tmpdir(), 'everyric-outline-dist-'));
cpSync(resolve(__dirname, '..', arg('dist', 'dist')), distDir, { recursive: true });
JSON.parse(readFileSync(join(distDir, 'manifest.json'), 'utf8'));
const userDataDir = arg('profile', join(tmpdir(), 'exec-lane2-outline-profile'));
mkdirSync(userDataDir, { recursive: true });

let failed = false;
const check = (ok, label, detail) => {
  console.log(`${ok ? 'PASS' : 'FAIL'}: ${label}${detail !== undefined ? ' = ' + JSON.stringify(detail) : ''}`);
  if (!ok) failed = true;
};

/** PiP 창 안의 세 글자 종류를 한 번에 잰다 */
const MEASURE = `(() => {
  const sr = document.getElementById('everyric-root')?.shadowRoot;
  const shadow = (s) => sr?.querySelector(s) ?? null;
  const ts = (el) => {
    if (!el) return null;
    const v = getComputedStyle(el).textShadow;
    return !v || v === 'none' ? 'none' : v;
  };
  // 캔버스 외곽선 잉크 — 알파가 있는 픽셀 중 «거의 검정»(다크 테마 외곽선 색) 비율.
  // 레인 글자(#f1f1f2 계열)·노트(#ffb02e)·오선(흐린 회색)은 전부 밝아서 섞이지 않는다.
  // 라이트 테마에서는 외곽선이 흰색이고 글자가 검정이라 방향이 뒤집힌다 — 둘 다 세고
  // 호출부가 테마에 맞는 값을 본다(외곽선 색은 CSS 변수 --ey-outline 한 곳에서 온다).
  const canvas = document.querySelector('.ey-main-lane') ?? shadow('.ey-main-lane');
  let lane = null;
  if (canvas && canvas.width) {
    const d = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
    let ink = 0, dark = 0, bright = 0;
    for (let i = 0; i < d.length; i += 4) {
      if (d[i + 3] < 60) continue;
      ink++;
      if (Math.max(d[i], d[i + 1], d[i + 2]) < 60) dark++;
      if (Math.min(d[i], d[i + 1], d[i + 2]) > 225) bright++;
    }
    lane = { w: canvas.width, h: canvas.height, ink, dark, bright,
      darkRatio: ink ? +(dark / ink).toFixed(4) : 0,
      brightRatio: ink ? +(bright / ink).toFixed(4) : 0 };
  }
  // 크로마키가 «가사창 열»까지 닿는가 — 문서 블랭킷(:root.ey-chroma body.ey-pip *)은
  // Shadow DOM 경계를 못 넘는다. 패널 배경이 불투명하게 남으면 스트리머 화면에서
  // 오른쪽 열만 검은 박스로 남는다.
  const panelEl = shadow('.ey-panel');
  const bg = panelEl ? getComputedStyle(panelEl).backgroundColor : null;
  const opaque = (v) => {
    if (!v) return null;
    const m = /rgba?\(([^)]+)\)/.exec(v);
    if (!m) return null;
    const p = m[1].split(',').map(s => parseFloat(s));
    return (p[3] === undefined ? 1 : p[3]) > 0.05;
  };
  return {
    panelBg: bg,
    panelBgOpaque: opaque(bg),
    // 블랭킷의 «예외» 규칙이 겨냥하는 요소들이 실제로 어디 있는가(문서 vs 그림자)
    chipInLightDom: !!document.querySelector('.ey-gen-chip'),
    chipInShadow: !!shadow('.ey-gen-chip'),
    panelLine: ts(shadow('.ey-line')),
    panelTitle: ts(shadow('.ey-song-title')),
    shortLine: ts(document.querySelector('.ey-pip-stage .ey-pip-line.current .ey-line')),
    shortSide: ts(document.querySelector('.ey-pip-stage .ey-pip-line.prev')),
    shortStageFound: !!document.querySelector('.ey-pip-stage'),
    lane,
  };
})()`;

const ctx = await chromium.launchPersistentContext(userDataDir, {
  ignoreDefaultArgs: ['--disable-extensions'],
  headless: false,
  viewport: { width: 1600, height: 1000 },
  args: [
    `--disable-extensions-except=${distDir}`, `--load-extension=${distDir}`,
    '--mute-audio', '--autoplay-policy=no-user-gesture-required', '--window-position=20,20',
  ],
});

try {
  const sw = ctx.serviceWorkers()[0] ?? await ctx.waitForEvent('serviceworker', { timeout: 15000 });
  await ensureLocalServerPermissionForServerUrl(ctx, sw, new URL(sw.url()).host, LOCAL_SERVER_URL);
  const setSettings = (patch) => sw.evaluate(async (p) => {
    const cur = (await chrome.storage.local.get('settings')).settings ?? {};
    await chrome.storage.local.set({ settings: { ...cur, ...p } });
  }, patch);

  await setSettings({
    serverUrl: LOCAL_SERVER_URL, uiLanguage: 'ko', theme: 'dark',
    pitchGuide: true, pipShortLyrics: true, pipShowVideo: true, pipPlaylist: false,
    pipLaneWidth: 320, pipPanelWidth: 380,
    pitchPronPosition: 'both', showPronunciation: true,
    streamTextOutline: false, pipChromaKey: 'off',
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
  await page.waitForTimeout(1200);
  await page.locator('#everyric-root [title="PiP 창으로 보기"]').first().click();
  await page.waitForTimeout(3500);
  const pip = ctx.pages().find(p => p !== page);
  if (!pip) throw new Error('PiP 창이 페이지로 노출되지 않음');

  const sample = async (label, patch, shot) => {
    await setSettings(patch);
    await pip.waitForTimeout(1800);
    const m = await pip.evaluate(MEASURE);
    console.log(`\n── ${label}`);
    console.log(`   가사창 줄   text-shadow: ${m.panelLine}`);
    console.log(`   단축 표시   text-shadow: ${m.shortLine} (앞뒤 줄: ${m.shortSide})`);
    console.log(`   레인 캔버스 외곽선 잉크: ${m.lane ? `${m.lane.dark}/${m.lane.ink} (${(m.lane.darkRatio * 100).toFixed(2)}%)` : '캔버스 없음'}`);
    console.log(`   가사창 배경: ${m.panelBg} (불투명=${m.panelBgOpaque})`);
    if (shot) {
      await pip.screenshot({ path: resolve(__dirname, `../pip-outline-${shot}.png`) });
      console.log(`   screenshot: pip-outline-${shot}.png`);
    }
    return m;
  };

  // ── 1) 외곽선 off (기준선) — 여기서 외곽선이 보이면 그 자체가 결함이다
  const base = await sample('외곽선 off / 크로마키 off (기준선)',
    { streamTextOutline: false, pipChromaKey: 'off' }, 'base');
  check(base.shortStageFound, '단축 표시가 PiP에 있다(측정 대상 존재)');
  check(base.panelLine === 'none' && base.shortLine === 'none',
    '외곽선 off면 DOM 글자에 그림자가 없다', { panel: base.panelLine, short: base.shortLine });
  const baseDark = base.lane?.darkRatio ?? 0;

  // ── 2) 외곽선 on, 크로마키 off/green/blue/magenta
  const results = {};
  for (const chroma of ['off', 'green', 'blue', 'magenta']) {
    results[chroma] = await sample(`외곽선 on / 크로마키 ${chroma}`,
      { streamTextOutline: true, pipChromaKey: chroma },
      chroma === 'off' ? 'outline-plain' : `outline-${chroma}`);
  }

  // 크로마키 예외 규칙(:root.ey-chroma body.ey-pip .ey-gen-chip 등)이 겨냥하는 요소가
  // 그림자 안에만 있다면 그 규칙은 애초에 죽은 코드다 — 블랭킷이 패널을 덮을 «의도»였다는 증거.
  console.log(`\n칩 위치: light DOM=${results.green.chipInLightDom} / shadow=${results.green.chipInShadow}`);
  for (const chroma of ['green', 'blue', 'magenta']) {
    check(results[chroma].panelBgOpaque === false,
      `[${chroma}] 가사창 배경도 키 컬러로 비침(검은 박스로 안 남음)`, results[chroma].panelBg);
  }
  // 누출 방지 — 크로마키가 꺼져 있으면 패널 배경은 원래대로 불투명해야 한다
  check(results.off.panelBgOpaque === true,
    '[크로마 off] 가사창 배경은 그대로 불투명(투명화가 새지 않음)', results.off.panelBg);
  // 메인 창(유튜브 페이지)은 크로마키와 무관하다 — :host-context가 PiP 밖으로 새면
  // 크로마키를 켠 사용자의 **일반 오버레이 패널**까지 투명해진다
  const mainBg = await page.evaluate(() => {
    const p = document.getElementById('everyric-root')?.shadowRoot?.querySelector('.ey-panel');
    return p ? getComputedStyle(p).backgroundColor : null;
  });
  check(!!mainBg && !/rgba\([^)]*,\s*0\s*\)/.test(mainBg),
    '메인 창 패널은 크로마키 green 중에도 불투명(PiP 밖으로 안 샘)', mainBg);

  for (const [chroma, m] of Object.entries(results)) {
    check(m.panelLine !== 'none', `[${chroma}] 가사창 줄에 외곽선`, m.panelLine?.slice(0, 60));
    check(m.shortLine !== 'none', `[${chroma}] 단축 표시 줄에 외곽선`, m.shortLine?.slice(0, 60));
    check(m.shortSide !== 'none', `[${chroma}] 단축 표시 앞뒤 줄에 외곽선`, m.shortSide?.slice(0, 60));
    check((m.lane?.darkRatio ?? 0) > baseDark + 0.02,
      `[${chroma}] 레인 캔버스에 외곽선 잉크`, { on: m.lane?.darkRatio, off: baseDark });
  }

  // ── 3) 라이트 테마 — 외곽선 색은 CSS 변수(--ey-outline)가 흰색으로 덮는다.
  //    캔버스도 같은 변수를 읽으므로 «검은 테가 흰 테로» 바뀌어야 한다(방향이 뒤집힌다).
  const lightBase = await sample('라이트 테마 / 외곽선 off (기준선)',
    { theme: 'light', streamTextOutline: false, pipChromaKey: 'off' });
  const light = await sample('라이트 테마 / 외곽선 on', { theme: 'light', streamTextOutline: true }, 'light');
  check(/255, 255, 255/.test(light.panelLine ?? ''), '[라이트] 가사창 줄 외곽선이 흰색', light.panelLine?.slice(0, 50));
  check(/255, 255, 255/.test(light.shortLine ?? ''), '[라이트] 단축 표시 외곽선이 흰색', light.shortLine?.slice(0, 50));
  check((light.lane?.brightRatio ?? 0) > (lightBase.lane?.brightRatio ?? 0) + 0.02,
    '[라이트] 레인 캔버스에 흰 외곽선 잉크',
    { on: light.lane?.brightRatio, off: lightBase.lane?.brightRatio });
  await setSettings({ theme: 'dark' });
  await pip.waitForTimeout(1200);

  // ── 4) 무회귀: 외곽선을 끄면 기준선과 같은 상태로 돌아온다
  const backOff = await sample('외곽선 다시 off (무회귀 확인)',
    { streamTextOutline: false, pipChromaKey: 'off' });
  check(backOff.panelLine === 'none' && backOff.shortLine === 'none',
    '외곽선 off로 되돌리면 DOM 그림자가 사라진다', { panel: backOff.panelLine, short: backOff.shortLine });
  check(Math.abs((backOff.lane?.darkRatio ?? 0) - baseDark) < 0.01,
    '외곽선 off로 되돌리면 레인 캔버스도 기준선과 같다',
    { back: backOff.lane?.darkRatio, base: baseDark });

  console.log(failed ? '\nPIP TEXT OUTLINE: FAIL' : '\nPIP TEXT OUTLINE: PASS');
  process.exitCode = failed ? 1 : 0;
} catch (e) {
  console.log('PIP TEXT OUTLINE: ERROR —', String(e).slice(0, 500));
  process.exitCode = 1;
} finally {
  await ctx.close();
}
