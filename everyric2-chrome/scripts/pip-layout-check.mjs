// PiP 3열 레이아웃 실측 — 운영자 인수 기준("구석에 몰려 있는 개차반이 아닐 것").
//
// 구조(2026-08-04 지시):
//   [레인] │ [중앙 열: 영상 ↑ / 가사 단축 표시 / 재생 컨트롤 ↓] │ [가사창] [재생목록]
//
// 재는 것은 «그려졌나»가 아니라 «읽을 수 있게 놓였나»다:
//   1) 열끼리 픽셀 겹침 0 — 절대좌표 배치가 되살아나면 여기서 즉시 잡힌다
//   2) 창 밖 이탈 0
//   3) 보이는 열은 전부 최소 폭 이상 — 짓눌린 채 «있는 척»하지 않는다
//   4) 좁은 창에서는 우선순위대로 접힌다(겹쳐 그리지 않는다)
//   5) 중앙 열 안 세로 순서(영상 → 단축 가사 → 컨트롤)와 겹침 0
//
// 실행: node scripts/pip-layout-check.mjs <syncedVideoId>
// 주소는 반드시 127.0.0.1 — 이 개발 머신은 localhost가 IPv6 폴백 스톨로 요청당 2초.
import { chromium } from 'playwright';
import { fileURLToPath } from 'url';
import { dirname, resolve, join } from 'path';
import { cpSync, mkdtempSync, readFileSync } from 'fs';
import { tmpdir } from 'os';
import { ensureLocalServerPermissionForServerUrl } from './lib/local-server-permission.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const liveDist = resolve(__dirname, '../dist');
const distDir = mkdtempSync(join(tmpdir(), 'everyric-dist-snap-'));
cpSync(liveDist, distDir, { recursive: true });
JSON.parse(readFileSync(join(distDir, 'manifest.json'), 'utf8'));

const videoId = process.argv[2];
const LOCAL_SERVER_URL = 'http://127.0.0.1:8000';
if (!videoId) {
  console.log('사용법: node scripts/pip-layout-check.mjs <syncedVideoId>');
  process.exit(2);
}

/** 열 최소 폭 — pip.ts의 LANE_COL_MIN / CENTER_COL_MIN / PANEL_COL_MIN과 같은 값 */
const MINS = { lane: 140, center: 200, panel: 280, playlist: 200 };

let failed = false;
const check = (ok, label, detail) => {
  console.log(`${ok ? 'PASS' : 'FAIL'}: ${label}${detail !== undefined ? ' = ' + JSON.stringify(detail) : ''}`);
  if (!ok) failed = true;
};

/** PiP 창 안의 열 기하를 읽는다 (PiP 페이지 컨텍스트에서 평가) */
const MEASURE = `(() => {
  const q = (s) => document.querySelector(s);
  const box = (el) => {
    if (!el) return null;
    const cs = getComputedStyle(el);
    if (cs.display === 'none') return null;
    const b = el.getBoundingClientRect();
    if (b.width === 0 && b.height === 0) return null;
    return { x: Math.round(b.x), y: Math.round(b.y), w: Math.round(b.width), h: Math.round(b.height),
             r: Math.round(b.right), b: Math.round(b.bottom) };
  };
  const root = document.getElementById('everyric-root');
  const sr = root?.shadowRoot;
  return {
    win: { w: innerWidth, h: innerHeight },
    cols: {
      lane: box(q('.ey-pip-lane-col')),
      center: box(q('.ey-pip-center')),
      panel: box(root),
      playlist: box(q('.ey-pip-playlist-col')),
    },
    center: {
      video: box(q('.ey-pip-video')),
      stage: box(q('.ey-pip-stage')),
      footer: box(q('.ey-pip-footer')),
    },
    // 각 열이 실제로 내용을 갖고 있는가 — 빈 상자만 놓고 PASS 하지 않기 위해
    laneCanvasW: Math.round(sr?.querySelector('.ey-main-lane')?.getBoundingClientRect().width ?? 0),
    panelLines: sr?.querySelectorAll('.ey-line').length ?? 0,
    panelHasHeader: !!sr?.querySelector('.ey-header'),
    playlistRows: sr?.querySelectorAll('.ey-pl-list *').length ?? 0,
    stageLine: q('.ey-pip-stage .ey-line') ? 1 : 0,
    stageUsesSharedLine: !!q('.ey-pip-stage .ey-pip-line.current .ey-line'),
    // 절대좌표 배치가 되살아났는지 감시 — 부착 패널에 left/top이 박히면 즉시 잡힌다
    strayCoords: [...document.querySelectorAll('.ey-attach-lane, .ey-attach-playlist')]
      .filter(el => el.style.left !== '' || el.style.top !== '').length,
  };
})()`;

function verify(name, m) {
  const shown = Object.entries(m.cols).filter(([, v]) => v);
  const names = shown.map(([k]) => k);
  console.log(`\n── ${name} (창 ${m.win.w}×${m.win.h}) — 보이는 열: ${names.join(' | ') || '(없음)'}`);

  // 1) 열끼리 겹침 0
  let overlap = null;
  for (let i = 0; i < shown.length; i++) {
    for (let j = i + 1; j < shown.length; j++) {
      const [an, a] = shown[i]; const [bn, b] = shown[j];
      if (!(a.r <= b.x || b.r <= a.x || a.b <= b.y || b.b <= a.y)) overlap = `${an}×${bn}`;
    }
  }
  check(!overlap, `${name}: 열끼리 겹침 없음`, overlap ?? '겹침 0');

  // 2) 창 밖 이탈 0
  const stray = shown.filter(([, v]) => v.r > m.win.w + 1 || v.b > m.win.h + 1).map(([k]) => k);
  check(stray.length === 0, `${name}: 창 밖으로 나간 열 없음`, stray);

  // 3) 보이는 열은 최소 폭 이상
  const squished = shown.filter(([k, v]) => v.w < MINS[k] - 1).map(([k, v]) => `${k}=${v.w}<${MINS[k]}`);
  check(squished.length === 0, `${name}: 보이는 열이 짓눌리지 않음`, squished.length ? squished : shown.map(([k, v]) => `${k}=${v.w}`));

  // 4) **중앙 열은 절대 접히지 않는다** (최후 생존 — PiP의 정체성이 영상+현재 줄이다)
  check(!!m.cols.center, `${name}: 중앙 열이 살아 있음(최후 생존)`);
  // 가사창이 보이면 실제 내용이 있어야 한다(빈 상자만 놓고 PASS 하지 않기)
  if (m.cols.panel) {
    check(m.panelLines > 0 && m.panelHasHeader, `${name}: 가사창에 실제 가사·헤더가 있음`, { lines: m.panelLines, header: m.panelHasHeader });
  }

  // 5) 중앙 열 세로 순서 + 겹침
  if (m.cols.center) {
    const { video, stage, footer } = m.center;
    const seq = [video, stage, footer].filter(Boolean);
    let bad = null;
    for (let i = 1; i < seq.length; i++) if (seq[i].y < seq[i - 1].b - 1) bad = i;
    check(bad === null, `${name}: 중앙 열 세로 순서(영상→단축가사→컨트롤) 유지`, {
      video: video && `${video.y}..${video.b}`, stage: stage && `${stage.y}..${stage.b}`, footer: footer && `${footer.y}..${footer.b}`,
    });
  }

  // 6) 절대좌표 잔재 감시
  check(m.strayCoords === 0, `${name}: 부착 패널에 절대좌표 잔재 없음`, m.strayCoords);
  return m;
}

const userDataDir = mkdtempSync(join(tmpdir(), 'everyric-pip-layout-'));
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
  const extId = new URL(sw.url()).host;
  await ensureLocalServerPermissionForServerUrl(ctx, sw, extId, LOCAL_SERVER_URL);

  const setSettings = (s) => sw.evaluate(async (patch) => {
    const cur = (await chrome.storage.local.get('settings')).settings ?? {};
    await chrome.storage.local.set({ settings: { ...cur, ...patch } });
  }, s);

  await setSettings({
    serverUrl: LOCAL_SERVER_URL, uiLanguage: 'ko', theme: 'dark',
    pitchGuide: true, pipPlaylist: true, pipShortLyrics: true, pipShowVideo: true,
    pipLaneWidth: 300, pipPanelWidth: 360, pipLaneSwapped: false,
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
    if (v) { v.currentTime = 25; void v.play().catch(() => {}); }
  });
  await page.waitForTimeout(1200);
  await page.locator('#everyric-root [title="PiP 창으로 보기"]').first().click();
  await page.waitForTimeout(3500);

  const pipPage = ctx.pages().find(p => p !== page);
  if (!pipPage) throw new Error('PiP 창이 페이지로 노출되지 않음');

  const SIZES = [
    ['작음', 400, 300],
    ['중간', 640, 480],
    ['넓음', 960, 600],
  ];
  // **PiP 표면의 키**로 켜고 끈다 — 메인의 modPlaylist/modMainLane은 PiP에 영향이 없다
  // (표면별 상태 원칙). 그 분리 자체도 아래 «표면 독립» 검사에서 따로 확인한다.
  const MODES = [
    ['부착 ON', { pitchGuide: true, pipPlaylist: true, pipShortLyrics: true }],
    ['부착 OFF', { pitchGuide: false, pipPlaylist: false, pipShortLyrics: false }],
  ];

  const shots = [];
  for (const [modeName, patch] of MODES) {
    await setSettings(patch);
    await page.waitForTimeout(1200); // storage.onChanged → applySettingsPatch 왑복
    for (const [sizeName, w, h] of SIZES) {
      await pipPage.setViewportSize({ width: w, height: h });
      await pipPage.waitForTimeout(700);
      const m = await pipPage.evaluate(MEASURE);
      verify(`${modeName}/${sizeName}`, m);
      const file = resolve(__dirname, `../pip-layout-${modeName === '부착 ON' ? 'on' : 'off'}-${w}x${h}-check.png`);
      await pipPage.screenshot({ path: file });
      shots.push(file);
    }
  }

  // ── 상태 영속: 열 상태·폭·창 크기를 바꾼 뒤 close → open ──────────
  //
  // 운영자 원칙("닫아도 상태 유지"). 창 크기는 ResizeObserver+500ms 디바운스로,
  // 열 상태·폭은 확정 시점 1회 저장으로 남는다 — 다시 열었을 때 그대로여야 한다.
  await setSettings({
    pitchGuide: true, pipPlaylist: true, pipShortLyrics: true, pipShowPanel: true,
    pipLaneWidth: 240, pipPanelWidth: 420,
  });
  await page.waitForTimeout(1000);
  await pipPage.setViewportSize({ width: 1280, height: 700 });
  await pipPage.waitForTimeout(1500); // 디바운스(500ms) 통과 대기
  const beforeClose = await pipPage.evaluate(MEASURE);

  await page.evaluate(() => window.documentPictureInPicture?.window?.close());
  await page.waitForTimeout(1500);
  await page.locator('#everyric-root [title="PiP 창으로 보기"]').first().click();
  await page.waitForTimeout(3500);
  const reopened = ctx.pages().find(p => p !== page);
  if (!reopened) throw new Error('다시 연 PiP 창이 페이지로 노출되지 않음');
  await reopened.waitForTimeout(800);
  const afterOpen = await reopened.evaluate(MEASURE);

  console.log('\n── 상태 영속 (close → open)');
  const near = (a, b, tol) => Math.abs(a - b) <= tol;
  // 창 «크기»는 렌더된 뷰포트로 잴 수 없다: Playwright의 setViewportSize는 페이지마다
  // 뷰포트를 에뮬레이션하고, 다시 연 PiP 페이지에는 컨텍스트 기본값(1600×1000)이 다시
  // 걸려 requestWindow가 실제로 무엇을 받았는지를 가린다(실측: before 1280×700 →
  // after 1600×1000). 그래서 **저장소에 남은 값**으로 잰다 — 디바운스 저장이 실제로
  // 돌았는지가 이 요구("닫아도 크기 유지")의 검증 가능한 절반이고, 그 값을 다음
  // requestWindow에 그대로 넘기는 것은 코드 경로가 한 줄이라 눈으로 확인된다.
  const savedSize = await sw.evaluate(async () => {
    const st = (await chrome.storage.local.get('settings')).settings ?? {};
    return { w: st.pipWidth, h: st.pipHeight };
  });
  // 비교 대상은 **content script가 보는 실제 PiP 창 크기**다. 위 MEASURE의 innerWidth는
  // Playwright가 그 페이지에만 씌운 에뮬레이션 값이라 실제 창과 다르다(실측: 페이지
  // 1280×700 / 실제 창 1600×1000). 디바운스 저장이 안 돌았다면 저장값은 열 때 요청한
  // 크기(기본 440×500)에 머물러 있으므로, 실제 창 크기와 일치한다는 것이 곧 «저장 경로가
  // 돌았다»는 증거다.
  const realWin = await page.evaluate(() => {
    const w = window.documentPictureInPicture?.window;
    return w ? { w: w.innerWidth, h: w.innerHeight } : null;
  });
  check(realWin !== null && near(savedSize.w ?? 0, realWin.w, 24) && near(savedSize.h ?? 0, realWin.h, 40),
    '창 크기가 저장됨(ResizeObserver+500ms 디바운스)', { saved: savedSize, realWindow: realWin });
  check(near(afterOpen.cols.lane?.w ?? 0, beforeClose.cols.lane?.w ?? 0, 2),
    '레인 열 폭이 복원됨', { before: beforeClose.cols.lane?.w, after: afterOpen.cols.lane?.w });
  check(near(afterOpen.cols.panel?.w ?? 0, beforeClose.cols.panel?.w ?? 0, 2),
    '가사창 열 폭이 복원됨', { before: beforeClose.cols.panel?.w, after: afterOpen.cols.panel?.w });
  check(!!afterOpen.cols.playlist === !!beforeClose.cols.playlist,
    '재생목록 열 표시 상태가 복원됨', { before: !!beforeClose.cols.playlist, after: !!afterOpen.cols.playlist });
  check(!!afterOpen.center.stage === !!beforeClose.center.stage,
    '가사 단축 표시 상태가 복원됨', { before: !!beforeClose.center.stage, after: !!afterOpen.center.stage });
  check(afterOpen.stageUsesSharedLine === true, '단축 표시가 공용 줄 렌더러(.ey-line)를 씀', afterOpen.stageUsesSharedLine);

  // 사용자가 **명시적으로** 가사창 열을 접으면 그 선택도 살아남는다
  await setSettings({ pipShowPanel: false });
  await page.waitForTimeout(1000);
  const folded = await reopened.evaluate(MEASURE);
  check(!folded.cols.panel, '가사창 열을 명시적으로 접으면 즉시 사라짐', !!folded.cols.panel);
  check(!!folded.cols.center, '접은 뒤에도 중앙 열은 남음', !!folded.cols.center);
  await page.evaluate(() => window.documentPictureInPicture?.window?.close());
  await page.waitForTimeout(1200);
  await page.locator('#everyric-root [title="PiP 창으로 보기"]').first().click();
  await page.waitForTimeout(3500);
  const reopened2 = ctx.pages().find(p => p !== page);
  const afterFold = reopened2 ? await reopened2.evaluate(MEASURE) : null;
  check(afterFold !== null && !afterFold.cols.panel,
    '접은 상태가 close → open 뒤에도 유지됨', afterFold ? !!afterFold.cols.panel : 'no window');
  if (reopened2) {
    await reopened2.screenshot({ path: resolve(__dirname, '../pip-layout-folded-check.png') });
    shots.push(resolve(__dirname, '../pip-layout-folded-check.png'));
  }

  // ── 열 스왑: [레인][중앙] ⇄ [중앙][레인] ────────────────────────
  const swapTarget = reopened2 ?? reopened;
  await setSettings({ pipShowPanel: true, pitchGuide: true, pipLaneSwapped: false });
  await page.waitForTimeout(1200);
  await swapTarget.setViewportSize({ width: 1280, height: 700 });
  await swapTarget.waitForTimeout(700);
  const normal = await swapTarget.evaluate(MEASURE);
  await setSettings({ pipLaneSwapped: true });
  await page.waitForTimeout(1200);
  const swapped = await swapTarget.evaluate(MEASURE);
  console.log('\n\u2500\u2500 \ub808\uc778\u00b7\uc911\uc559 \uc5f4 \uc2a4\uc651');
  const bothShown = normal.cols.lane && normal.cols.center && swapped.cols.lane && swapped.cols.center;
  check(bothShown, '\uc2a4\uc651 \uac80\uc0ac \uc804\uc81c: \ub808\uc778\u00b7\uc911\uc559 \uc5f4\uc774 \ub458 \ub2e4 \ubcf4\uc784',
    { normal: !!normal.cols.lane, swapped: !!swapped.cols.lane });
  if (bothShown) {
    check(normal.cols.lane.x < normal.cols.center.x, '\uae30\ubcf8 \ubc30\uce58: \ub808\uc778\uc774 \uc911\uc559 \uc5f4 \uc67c\ucabd',
      { lane: normal.cols.lane.x, center: normal.cols.center.x });
    check(swapped.cols.lane.x > swapped.cols.center.x, '\uc2a4\uc651 \ud6c4: \ub808\uc778\uc774 \uc911\uc559 \uc5f4 \uc624\ub978\ucabd',
      { lane: swapped.cols.lane.x, center: swapped.cols.center.x });
    check(!!swapped.cols.panel && swapped.cols.lane.r <= swapped.cols.panel.x + 1,
      '\uc2a4\uc651\ud574\ub3c4 \uac00\uc0ac\ucc3d\uc740 \uacc4\uc18d \ub9e8 \uc624\ub978\ucabd',
      { laneRight: swapped.cols.lane.r, panelLeft: swapped.cols.panel?.x });
  }
  await page.evaluate(() => window.documentPictureInPicture?.window?.close());
  await page.waitForTimeout(1200);
  await page.locator('#everyric-root [title="PiP \ucc3d\uc73c\ub85c \ubcf4\uae30"]').first().click();
  await page.waitForTimeout(3500);
  const afterSwapWin = ctx.pages().find(p => p !== page);
  const afterSwap = afterSwapWin ? await afterSwapWin.evaluate(MEASURE) : null;
  check(afterSwap !== null && !!afterSwap.cols.lane && !!afterSwap.cols.center
    && afterSwap.cols.lane.x > afterSwap.cols.center.x,
    '\uc2a4\uc651 \uc0c1\ud0dc\uac00 close \u2192 open \ub4a4\uc5d0\ub3c4 \uc720\uc9c0\ub428',
    afterSwap ? { lane: afterSwap.cols.lane?.x, center: afterSwap.cols.center?.x } : 'no window');
  if (afterSwapWin) {
    await afterSwapWin.screenshot({ path: resolve(__dirname, '../pip-layout-swapped-check.png') });
    shots.push(resolve(__dirname, '../pip-layout-swapped-check.png'));
  }

  // \uba54\uc778 \ud0a4\ub97c \uaebc\ub3c4 PiP \uc5f4\uc740 \uadf8\ub300\ub85c
  await setSettings({ modPlaylist: false, modMainLane: false });
  await page.waitForTimeout(1200);
  const surfIndep = afterSwapWin ? await afterSwapWin.evaluate(MEASURE) : null;
  console.log('\n\u2500\u2500 \ud45c\uba74 \ub3c5\ub9bd');
  check(surfIndep !== null && !!surfIndep.cols.lane,
    '\uba54\uc778 \ub808\uc778\uc744 \uaebc\ub3c4 PiP \ub808\uc778 \uc5f4\uc740 \ub0a8\ub294\ub2e4', surfIndep ? !!surfIndep.cols.lane : 'no window');
  check(surfIndep !== null && !!surfIndep.cols.playlist,
    '\uba54\uc778 \uc7ac\uc0dd\ubaa9\ub85d\uc744 \uaebc\ub3c4 PiP \uc7ac\uc0dd\ubaa9\ub85d \uc5f4\uc740 \ub0a8\ub294\ub2e4', surfIndep ? !!surfIndep.cols.playlist : 'no window');

  console.log('\nscreenshots:');
  for (const f of shots) console.log('  ' + f);
  console.log(failed ? '\nPIP LAYOUT CHECK: FAIL' : '\nPIP LAYOUT CHECK: PASS');
  process.exitCode = failed ? 1 : 0;
} catch (e) {
  console.log('PIP LAYOUT CHECK: ERROR —', String(e).slice(0, 400));
  process.exitCode = 1;
} finally {
  await ctx.close();
}
