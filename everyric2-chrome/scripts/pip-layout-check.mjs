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
import { cpSync, mkdirSync, readFileSync } from 'fs';
import { tmpdir } from 'os';
import { ensureLocalServerPermissionForServerUrl } from './lib/local-server-permission.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
// 고정 dist 스냅샷 + 고정 프로필(병렬 검수 규약, a28540e): 무작위 경로는 언팩 확장 ID가
// 매번 달라져 host permission이 안 남고, 그 버블을 지우려 taskkill //IM chrome.exe를 쓰면
// 다른 에이전트의 브라우저까지 죽는다. 고정 경로에 매번 덮어써 최신 빌드는 그대로 반영한다.
const distDir = process.env.EVERYRIC_E2E_DIST_DIR
  ?? join(tmpdir(), 'everyric-e2e-profiles', 'pip-layout-dist');
mkdirSync(distDir, { recursive: true });
cpSync(resolve(__dirname, '../dist'), distDir, { recursive: true });
JSON.parse(readFileSync(join(distDir, 'manifest.json'), 'utf8')); // 빌드 도중 깨진 스냅샷이면 즉사

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
    // 레인 캔버스도 filled에서는 Shadow DOM 밖(laneSlot)이다 — 한 곳만 보면 PiP에서 늘 0이 되어
    // 「열은 있는데 캔버스가 없다」를 구분하지 못한다
    laneCanvasW: Math.round((q('.ey-main-lane') ?? sr?.querySelector('.ey-main-lane'))
      ?.getBoundingClientRect().width ?? 0),
    panelLines: sr?.querySelectorAll('.ey-line').length ?? 0,
    panelHasHeader: !!sr?.querySelector('.ey-header'),
    // 재생목록은 **부착 패널 자체**를 봐야 한다. filled에서는 그 패널이 Shadow DOM 밖의
    // 열(playlistSlot)로 나가므로 두 곳을 다 뒤지고, 표시 여부는 computed display로,
    // 내용은 목록 행(또는 「다음 영상」 대체 카드) 수로 센다 — 아래 «표면 독립» 검사가
    // 이 값을 쓴다(판정 기준을 상자에서 내용으로 옮긴 이유는 그쪽 주석 참조).
    playlist: (() => {
      const el = q('.ey-attach-playlist') ?? sr?.querySelector('.ey-attach-playlist');
      if (!el) return { present: false, visible: false, rows: 0 };
      const b = el.getBoundingClientRect();
      return {
        present: true,
        visible: getComputedStyle(el).display !== 'none' && b.width > 0 && b.height > 0,
        rows: el.querySelectorAll('.ey-pl-row-title, .ey-nextup-card').length,
      };
    })(),
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

const userDataDir = process.env.EVERYRIC_E2E_PROFILE_DIR
  ?? join(tmpdir(), 'everyric-e2e-profiles', 'pip-layout-check');
mkdirSync(userDataDir, { recursive: true });
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
  console.log('\n── 레인·중앙 열 스왑');
  const bothShown = normal.cols.lane && normal.cols.center && swapped.cols.lane && swapped.cols.center;
  check(bothShown, '스왑 검사 전제: 레인·중앙 열이 둘 다 보임',
    { normal: !!normal.cols.lane, swapped: !!swapped.cols.lane });
  if (bothShown) {
    check(normal.cols.lane.x < normal.cols.center.x, '기본 배치: 레인이 중앙 열 왼쪽',
      { lane: normal.cols.lane.x, center: normal.cols.center.x });
    check(swapped.cols.lane.x > swapped.cols.center.x, '스왑 후: 레인이 중앙 열 오른쪽',
      { lane: swapped.cols.lane.x, center: swapped.cols.center.x });
    check(!!swapped.cols.panel && swapped.cols.lane.r <= swapped.cols.panel.x + 1,
      '스왑해도 가사창은 계속 맨 오른쪽',
      { laneRight: swapped.cols.lane.r, panelLeft: swapped.cols.panel?.x });
  }
  await page.evaluate(() => window.documentPictureInPicture?.window?.close());
  await page.waitForTimeout(1200);
  await page.locator('#everyric-root [title="PiP 창으로 보기"]').first().click();
  await page.waitForTimeout(3500);
  const afterSwapWin = ctx.pages().find(p => p !== page);
  const afterSwap = afterSwapWin ? await afterSwapWin.evaluate(MEASURE) : null;
  check(afterSwap !== null && !!afterSwap.cols.lane && !!afterSwap.cols.center
    && afterSwap.cols.lane.x > afterSwap.cols.center.x,
    '스왑 상태가 close → open 뒤에도 유지됨',
    afterSwap ? { lane: afterSwap.cols.lane?.x, center: afterSwap.cols.center?.x } : 'no window');
  if (afterSwapWin) {
    await afterSwapWin.screenshot({ path: resolve(__dirname, '../pip-layout-swapped-check.png') });
    shots.push(resolve(__dirname, '../pip-layout-swapped-check.png'));
  }

  // ── 표면 독립: 메인 키를 꺼도 PiP 쪽은 그대로 ─────────────────────
  //
  // **판정 기준을 «열 상자»에서 «열 안의 내용»으로 옮겼다(2026-08-04).** 실제로 난
  // 결함(운영자 실제보 P1)의 모양이 바로 그 틈이었다: 열 상자(.ey-pip-playlist-col)는
  // 그대로 남고 **그 안의 부착 패널만 display:none**이 되어, 상자 존재로 세는 이 검사가
  // 거짓 통과했다. 원인은 overlay.ts의 updatePlaylistPlacement()/renderPlaylistPanel()이
  // playlistVisible() 대신 settings.modPlaylist를 직접 읽어 filled 인스턴스가 메인 키를
  // 따라간 것(방송된 settings는 두 인스턴스가 공유한다).
  // 그래서 아래는 표시 여부와 **행 수**를 함께 본다 — 빈 상자는 PASS가 아니다.
  // modNextUp도 함께 끈다 — 이래야 PiP 쪽에 남는 내용이 «PiP 표면 키로 조달된 것»임이
  // 확실해진다(다음 영상 카드 모듈이 켜져 있으면 행 수가 그쪽 덕분일 수 있어 증거가 흐려진다)
  await setSettings({ modPlaylist: false, modMainLane: false, modNextUp: false });
  await page.waitForTimeout(1500);
  const surfIndep = afterSwapWin ? await afterSwapWin.evaluate(MEASURE) : null;
  console.log('\n── 표면 독립');
  check(surfIndep !== null && !!surfIndep.cols.lane && surfIndep.laneCanvasW > 0,
    '메인 레인을 꺼도 PiP 레인 열은 남는다(캔버스까지)',
    surfIndep ? { col: !!surfIndep.cols.lane, canvasW: surfIndep.laneCanvasW } : 'no window');
  check(surfIndep !== null && !!surfIndep.cols.playlist && surfIndep.playlist.visible,
    '메인 재생목록을 꺼도 PiP 재생목록 패널이 보인다',
    surfIndep ? { col: !!surfIndep.cols.playlist, panel: surfIndep.playlist } : 'no window');
  check(surfIndep !== null && surfIndep.playlist.rows > 0,
    '그 패널이 **내용까지** 남아 있다(빈 상자 거짓 통과 방지)',
    surfIndep ? surfIndep.playlist : 'no window');

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
