// 재생목록 모듈의 **표면 분리** 회귀 검사 — 메인 창과 PiP 창은 서로의 on/off를 따라가면 안 된다.
//
// 왜 이 하네스가 따로 필요한가(2026-08-04 실사고, 운영자 실제보 P1):
// 토글(쓰기) 경로는 처음부터 표면별 키를 쓰고 있었는데, **적용(렌더) 경로**가 전역 키를
// 직접 읽고 있었다 — overlay.ts의 updatePlaylistPlacement()/renderPlaylistPanel()이
// `settings.modPlaylist`를 보는 바람에, applySettings가 두 인스턴스에 같은 settings를
// 방송할 때 filled(PiP) 인스턴스가 메인 키를 따라갔다(방향: 메인 → PiP). 여기에 데이터
// 조달 게이트(content.ts refreshPlaylist/refreshNextUp)도 modPlaylist만 봐서, 메인을 끄고
// PiP만 켜면 «열은 뜨는데 내용이 없는» 두 번째 증상이 났다.
//
// **판정은 열 상자가 아니라 내용으로 한다.** 이 결함의 모양이 «열 상자는 남고 그 안의
// 부착 패널만 display:none»이라, 상자 존재로 세면 거짓 통과한다(1차 검수가 정확히 그렇게
// 놓쳤다). 그래서 아래 readSurface는 .ey-attach-playlist의 computed display와 **실제 행 수**를
// 같이 읽고, "켜져 있다"고 주장하는 쪽은 행이 0이면 실패로 본다.
//
// 또 하나의 교훈: **양방향을 다 눌러야 한다.** 한쪽(PiP에서 켜기)만 재면 반대편이 내내
// 켜져 있어 "상대 키 그대로" 어서션이 공허하게 통과한다.
//
// 실행: node scripts/pip-playlist-surface-check.mjs [videoIdA] [videoIdB] [serverUrl]
//   videoIdB는 «분리된 상태에서 곡 전환» 구간에 쓴다(렌더·조달이 다시 도는 경로).
//   주소는 반드시 127.0.0.1 — 이 개발 머신은 localhost가 IPv6 폴백 스톨로 요청당 2초.
import { chromium } from 'playwright';
import { fileURLToPath } from 'url';
import { dirname, resolve, join } from 'path';
import { mkdirSync, cpSync, readFileSync } from 'fs';
import { tmpdir } from 'os';
import { ensureLocalServerPermissionForServerUrl } from './lib/local-server-permission.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
// 고정 dist 스냅샷 + 고정 프로필(병렬 검수 규약, a28540e): 무작위 경로는 확장 ID가 매번
// 달라져 host permission이 안 남고, 그걸 지우려 taskkill //IM chrome.exe를 쓰면 다른
// 에이전트의 브라우저까지 죽인다. 고정 경로에 매번 덮어써서 최신 빌드는 그대로 반영한다.
const distDir = process.env.EVERYRIC_E2E_DIST_DIR
  ?? join(tmpdir(), 'everyric-e2e-profiles', 'pip-playlist-surface-dist');
mkdirSync(distDir, { recursive: true });
cpSync(resolve(__dirname, '../dist'), distDir, { recursive: true });
JSON.parse(readFileSync(join(distDir, 'manifest.json'), 'utf8')); // 빌드 도중 깨진 스냅샷이면 여기서 즉사

const VIDEO_A = process.argv[2] ?? 'f6TytcA47rI';
const VIDEO_B = process.argv[3] ?? 'qcI4QHKtk0E';
const SERVER = process.argv[4] ?? 'http://127.0.0.1:8000';
const IS_LOCAL = /^https?:\/\/(127\.0\.0\.1|localhost)/.test(SERVER);

let failed = false;
const check = (ok, label, detail) => {
  console.log(`${ok ? 'PASS' : 'FAIL'}: ${label}${detail !== undefined ? ' = ' + JSON.stringify(detail) : ''}`);
  if (!ok) failed = true;
  return ok;
};

/**
 * 한 문서(메인 탭 / PiP 창)의 재생목록 상태 — **상자가 아니라 내용**으로 읽는다.
 * 부착 패널은 filled에서 light DOM의 열로 나가고 floating에서는 Shadow DOM 안에 있으므로
 * 두 곳을 다 본다(scripts/lib/pip-panel.mjs의 레인 판정과 같은 규약).
 */
const READ_ONE = `((doc, win) => {
  const root = doc.getElementById('everyric-root')?.shadowRoot;
  const el = doc.querySelector('.ey-attach-playlist') ?? root?.querySelector('.ey-attach-playlist');
  const visible = !!el && win.getComputedStyle(el).display !== 'none'
    && el.getBoundingClientRect().width > 0 && el.getBoundingClientRect().height > 0;
  return {
    visible,
    // 목록 행 또는 «다음 영상» 대체 카드 — 둘 중 하나라도 있어야 «채워졌다»고 본다
    rows: el ? el.querySelectorAll('.ey-pl-row-title, .ey-nextup-card').length : 0,
    text: el ? (el.textContent || '').replace(/\\s+/g, ' ').trim().slice(0, 50) : '',
    quickOn: !!root?.querySelector('.ey-quick-row .ey-mini:nth-child(4)')?.classList.contains('on'),
    lyricWords: root ? root.querySelectorAll('.ey-word').length : 0,
  };
})`;

const READ = `(() => {
  const readOne = ${READ_ONE};
  const w = window.documentPictureInPicture?.window;
  return {
    main: readOne(document, window),
    pipOpen: !!w,
    pip: w ? readOne(w.document, w) : null,
  };
})()`;

const userDataDir = process.env.EVERYRIC_E2E_PROFILE_DIR
  ?? join(tmpdir(), 'everyric-e2e-profiles', 'pip-playlist-surface-check');
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
  const sw = ctx.serviceWorkers()[0] ?? await ctx.waitForEvent('serviceworker', { timeout: 20000 });
  const extId = new URL(sw.url()).host;
  // 프로드는 manifest의 필수 권한이라 부여 절차가 없다 — 로컬(옵셔널 권한)일 때만 확보한다
  if (IS_LOCAL) await ensureLocalServerPermissionForServerUrl(ctx, sw, extId, SERVER);

  const setSettings = (patch) => sw.evaluate(async (p) => {
    const cur = (await chrome.storage.local.get('settings')).settings ?? {};
    await chrome.storage.local.set({ settings: { ...cur, ...p } });
  }, patch);
  const keys = () => sw.evaluate(async () => {
    const s = (await chrome.storage.local.get('settings')).settings ?? {};
    return { modPlaylist: s.modPlaylist, pipPlaylist: s.pipPlaylist };
  });

  await setSettings({
    serverUrl: SERVER, uiLanguage: 'ko', theme: 'dark',
    // 프로드에서는 번역을 끈다 — /api/translate는 POST라 읽기 전용 회차에서 서버에 쓰게 된다
    showTranslation: IS_LOCAL, showPronunciation: true,
    modPlaylist: true, pipPlaylist: true, modMainLane: false, pitchGuide: true,
    // 다음 영상 카드 모듈을 꺼 둔다 — 이래야 조달 게이트(refreshNextUp)가 **표면 키에만**
    // 의존하게 되어, 「메인 꺼짐·PiP 켜짐」에서 내용이 채워지는 것이 분리의 증거가 된다
    modNextUp: false,
    pipShortLyrics: true, pipShowPanel: true, pipShowCenter: true,
    pipLaneWidth: 260, pipPanelWidth: 380, pipLaneSwapped: false,
    // 두 표면을 **동시에** 띄워야 «따로따로»를 한 스냅샷에서 잰다. pipKeepPanel 기본값은
    // false이고 마커 없는 저장본은 1회 마이그레이션이 강제로 내리므로 마커를 함께 심는다
    // (src/lib/settings.ts — 안 심으면 true가 무시된다).
    pipKeepPanel: true, settingsMigrations: ['pipKeepPanel-default-false-2026-08'],
    pipWidth: 1200, pipHeight: 700,
  });

  const page = ctx.pages()[0] ?? await ctx.newPage();
  const waitLyrics = () => page.waitForFunction(() => {
    const r = document.getElementById('everyric-root')?.shadowRoot;
    return (r?.querySelectorAll('.ey-word').length ?? 0) > 0;
  }, null, { timeout: 90000, polling: 1000 });
  const playFrom = (sec) => page.evaluate((s) => {
    const v = document.querySelector('video.html5-main-video') ?? document.querySelector('video');
    if (v) { v.currentTime = s; void v.play().catch(() => {}); }
  }, sec);
  /** 진짜 SPA 전환 — 하드 내비게이션은 opener 문서를 갈아 PiP 창을 함께 닫는다 */
  const spaGoto = async (vid) => {
    await page.evaluate((v) => {
      document.querySelector('ytd-app')?.dispatchEvent(new CustomEvent('yt-navigate', {
        bubbles: true, composed: true,
        detail: { endpoint: {
          commandMetadata: { webCommandMetadata: { url: `/watch?v=${v}`, webPageType: 'WEB_PAGE_TYPE_WATCH', rootVe: 3832 } },
          watchEndpoint: { videoId: v } } },
      }));
    }, vid);
    await page.waitForFunction((v) => location.href.includes(v), vid, { timeout: 20000, polling: 500 });
    await page.waitForTimeout(1500);
    await playFrom(30);
    await page.waitForTimeout(5000);
  };

  await page.goto(`https://www.youtube.com/watch?v=${VIDEO_A}`, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForSelector('#everyric-root', { state: 'attached', timeout: 30000 });
  await waitLyrics();
  await playFrom(30);
  await page.waitForTimeout(1500);
  await page.locator('#everyric-root [title="PiP 창으로 보기"]').first().click();
  await page.waitForTimeout(3500);
  let pip = ctx.pages().find(p => p !== page);
  if (!pip) throw new Error('PiP 창이 페이지로 노출되지 않음');
  await pip.setViewportSize({ width: 1200, height: 700 });
  await pip.waitForTimeout(1500);

  /**
   * 기대 상태가 될 때까지 폴링한다 — 고정 sleep으로 재면 안 되는 이유가 있다:
   * 목록 데이터는 유튜브 DOM 스크랩이라 곡 전환 직후 빈손일 수 있고, content.ts가
   * 백오프로 몇 번 더 시도한다(PLAYLIST_EMPTY_RETRY_DELAYS_MS). 그 사이에 재면
   * «켜져 있는데 행 0»이 뜨는데, 그건 표면 분리 결함이 아니라 아직 안 채워진 것이다.
   * 마지막 스냅샷을 돌려주므로 실패해도 실제 관측값으로 보고된다.
   *
   * 바로 아래 전제 체크도 이 헬퍼를 쓴다 — PiP를 막 연 직후도 같은 백오프 창 안이라
   * 1회성 조회로 재면 똑같이 거짓 FAIL이 났다(2026-08-04 회귀 게이트 실측 — 이 헬퍼가
   * 이미 있는데 정작 최초 전제 체크만 안 쓰고 있었다).
   */
  const settle = async (pred, ms = 20000) => {
    const t0 = Date.now();
    let s = await page.evaluate(READ);
    while (!pred(s) && Date.now() - t0 < ms) {
      await page.waitForTimeout(800);
      s = await page.evaluate(READ);
    }
    return s;
  };
  /** 기대대로 «보이는 쪽은 내용까지 있고, 꺼진 쪽은 안 보인다»가 될 때까지 기다린다 */
  const settleExpect = (expect) => settle(s =>
    s.main.visible === expect.main && (s.pip?.visible ?? false) === expect.pip
    && (!expect.main || s.main.rows > 0) && (!expect.pip || (s.pip?.rows ?? 0) > 0));

  const base = await settleExpect({ main: true, pip: true });
  check(base.pipOpen, '전제: PiP가 열렸다');
  check(base.main.visible && base.main.rows > 0,
    '전제: 메인 재생목록이 내용까지 보인다', { visible: base.main.visible, rows: base.main.rows });
  check(base.pip?.visible && base.pip?.rows > 0,
    '전제: PiP 재생목록도 내용까지 보인다', { visible: base.pip?.visible, rows: base.pip?.rows });

  /**
   * 한 조작 경로를 눌러 보고 «누른 표면만 바뀌었는가»를 판정한다.
   * 화면(표시 여부) · 저장소 키 · 켜져 있다는 쪽의 내용까지 셋 다 본다.
   */
  const trial = async (label, clickFn, expect) => {
    console.log(`\n── ${label}`);
    await clickFn();
    await page.waitForTimeout(1200); // 설정 왕복(storage.onChanged → applySettings)
    const s = await settleExpect(expect);
    const k = await keys();
    check(s.main.visible === expect.main,
      `${label}: 메인 재생목록 표시 = ${expect.main}`, { got: s.main.visible, rows: s.main.rows });
    check((s.pip?.visible ?? false) === expect.pip,
      `${label}: PiP 재생목록 표시 = ${expect.pip}`, { got: s.pip?.visible, rows: s.pip?.rows });
    check(k.modPlaylist === expect.main && k.pipPlaylist === expect.pip,
      `${label}: 저장소 키도 그 표면만 움직였다`, k);
    if (expect.main) check(s.main.rows > 0, `${label}: 메인 쪽 목록 내용이 남아 있다`, s.main.rows);
    if (expect.pip) check(s.pip.rows > 0, `${label}: PiP 쪽 목록 내용이 남아 있다`, s.pip?.rows);
    return s;
  };

  // 퀵 줄의 재생목록 버튼은 4번째다(레인 / 레인 위치 / 자막 / 재생목록 — overlay.ts quickRow)
  const quickOf = (target) => () => target.evaluate(() => {
    document.getElementById('everyric-root').shadowRoot
      .querySelectorAll('.ey-quick-row .ey-mini')[3].click();
  });
  // 설정 시트 행은 **키가 modPlaylist 하나**다 — filled 인스턴스에서는 같은 행의 onChange가
  // pipPlaylist를 쓴다("이 창에서 보이는 토글은 전부 이 창의 것"). 그래서 검색어도 하나다.
  const sheetOf = (target) => () => target.evaluate(() => new Promise(res => {
    const root = document.getElementById('everyric-root').shadowRoot;
    root.querySelectorAll('.ey-actions .ey-btn')[5].click(); // 톱니(설정) 열기
    setTimeout(() => {
      const filter = root.querySelector('.ey-settings-filter');
      filter.value = 'modPlaylist';
      filter.dispatchEvent(new Event('input', { bubbles: true }));
      setTimeout(() => {
        const rows = [...root.querySelectorAll('.ey-settings-sections .ey-settings-row')]
          .filter(r => getComputedStyle(r).display !== 'none');
        rows.map(r => r.querySelector('input[type=checkbox]')).find(Boolean)?.click();
        setTimeout(() => { root.querySelectorAll('.ey-actions .ey-btn')[5].click(); res(true); }, 400);
      }, 500);
    }, 700);
  }));
  // 코너 툴바 순서: 레인 / 중앙 / 영상 / 단축 / 가사창 / 재생목록 / 이중표시 (pip.ts)
  const cornerPlaylist = () => pip.evaluate(() => {
    document.querySelectorAll('.ey-pip-corner .ey-pip-mini')[5].click();
  });

  // ── 5경로 × 양방향 ────────────────────────────────────────────────
  await trial('경로1 메인 퀵 토글 OFF', quickOf(page), { main: false, pip: true });
  await trial('경로1 메인 퀵 토글 ON 복귀', quickOf(page), { main: true, pip: true });
  await trial('경로2 메인 설정 시트 OFF', sheetOf(page), { main: false, pip: true });
  await trial('경로2 메인 설정 시트 ON 복귀', sheetOf(page), { main: true, pip: true });
  await trial('경로3 PiP 퀵 토글 OFF', quickOf(pip), { main: true, pip: false });
  await trial('경로3 PiP 퀵 토글 ON 복귀', quickOf(pip), { main: true, pip: true });
  await trial('경로4 PiP 설정 시트 OFF', sheetOf(pip), { main: true, pip: false });
  await trial('경로4 PiP 설정 시트 ON 복귀', sheetOf(pip), { main: true, pip: true });
  await trial('경로5 PiP 코너 툴바 OFF', cornerPlaylist, { main: true, pip: false });
  await trial('경로5 PiP 코너 툴바 ON 복귀', cornerPlaylist, { main: true, pip: true });

  // ── 메인만 끈 채 PiP 여닫기 ───────────────────────────────────────
  console.log('\n── 메인 OFF 상태로 PiP 여닫기');
  await quickOf(page)();
  await page.waitForTimeout(1800);
  await page.evaluate(() => window.documentPictureInPicture?.window?.close());
  await page.waitForTimeout(2000);
  const closed = await page.evaluate(READ);
  check(closed.main.visible === false, '메인만 끈 상태가 PiP를 닫아도 유지된다', closed.main.visible);
  await page.locator('#everyric-root [title="PiP 창으로 보기"]').first().click();
  await page.waitForTimeout(3500);
  pip = ctx.pages().find(p => p !== page) ?? pip;
  await pip.setViewportSize({ width: 1200, height: 700 });
  await pip.waitForTimeout(1500);
  const reopened = await settleExpect({ main: false, pip: true });
  check(reopened.pip?.visible === true && reopened.pip?.rows > 0,
    '메인이 꺼진 채여도 다시 연 PiP의 재생목록은 내용까지 살아 있다',
    { visible: reopened.pip?.visible, rows: reopened.pip?.rows });
  check(reopened.main.visible === false, '그래도 메인은 꺼진 그대로다', reopened.main.visible);

  // ── 분리된 상태에서 **곡 전환** ───────────────────────────────────
  //
  // 재생목록 패널은 곡이 바뀔 때 setPlaylist → renderPlaylistPanel로 다시 그려지고,
  // 데이터 조달도 그때 다시 돈다 — 표면 분리가 그 두 곳까지 지켜지는지는 곡을 실제로
  // 넘겨 봐야 안다(토글 순간만 보면 통과하는 결함이 여기서 잡힌다).
  console.log('\n── 갈래 A: 메인 OFF · PiP ON 으로 곡 전환');
  await setSettings({ modPlaylist: false, pipPlaylist: true });
  await page.waitForTimeout(2000);
  await spaGoto(VIDEO_B);
  await waitLyrics();
  const a = await settleExpect({ main: false, pip: true });
  check(a.pip?.visible === true, 'A: 곡을 바꿔도 PiP 재생목록이 그대로 보인다', a.pip?.visible);
  check((a.pip?.rows ?? 0) > 0,
    'A: 곡을 바꾼 뒤에도 PiP 목록이 내용까지 채워진다(조달 게이트가 메인 키에 안 묶임)',
    { rows: a.pip?.rows, text: a.pip?.text });
  check(a.main.visible === false, 'A: 메인은 여전히 꺼져 있다', a.main.visible);

  console.log('\n── 갈래 B: 메인 ON · PiP OFF 로 곡 전환');
  await setSettings({ modPlaylist: true, pipPlaylist: false });
  await page.waitForTimeout(2000);
  await spaGoto(VIDEO_A);
  await waitLyrics();
  const b = await settleExpect({ main: true, pip: false });
  check(b.pip?.visible === false, 'B: 곡을 바꿔도 PiP 재생목록이 되살아나지 않는다', b.pip?.visible);
  check(b.main.visible === true && (b.main.rows ?? 0) > 0,
    'B: 메인 쪽은 내용까지 정상', { visible: b.main.visible, rows: b.main.rows });

  console.log('\n── 갈래 C: 둘 다 OFF → PiP만 ON (조달 재개)');
  await setSettings({ modPlaylist: false, pipPlaylist: false });
  await page.waitForTimeout(2500);
  const c0 = await page.evaluate(READ);
  check(c0.main.visible === false && c0.pip?.visible === false, 'C 전제: 양쪽 다 꺼짐',
    { main: c0.main.visible, pip: c0.pip?.visible });
  await setSettings({ pipPlaylist: true });
  await page.waitForTimeout(1200);
  const c1 = await settleExpect({ main: false, pip: true });
  check(c1.pip?.visible === true && (c1.pip?.rows ?? 0) > 0,
    'C: 메인이 꺼진 채 PiP만 켜도 목록이 내용까지 채워진다(refreshPlaylist가 pipPlaylist로도 깨어난다)',
    { visible: c1.pip?.visible, rows: c1.pip?.rows });
  check(c1.main.visible === false, 'C: 메인은 꺼진 그대로', c1.main.visible);

  // *-check.png는 .gitignore 대상 — 리포에 산출물을 남기지 않는다
  await pip.screenshot({ path: resolve(__dirname, '../pip-playlist-surface-check.png') });
  console.log('screenshot: pip-playlist-surface-check.png');
  console.log(failed ? '\nPIP PLAYLIST SURFACE CHECK: FAIL' : '\nPIP PLAYLIST SURFACE CHECK: PASS');
  process.exitCode = failed ? 1 : 0;
} catch (e) {
  console.log('PIP PLAYLIST SURFACE CHECK: ERROR —', String(e?.stack ?? e).slice(0, 500));
  process.exitCode = 1;
} finally {
  await ctx.close();
}
