// PiP 열 조작 «경합» 실측 — 연타·자동접힘·드래그 중 리사이즈.
//
// 무엇을 왜 재는가:
//   기존 pip-layout-check.mjs는 크기별 **정적 스냅샷**만 잰다(그 크기에서 열이 겹치지 않는가).
//   실제 사고는 정적인 자리가 아니라 «두 조작이 겹치는 순간»에 난다 — 접기와 스왑이 같은
//   프레임에 들어오거나, 자동 접힘이 도는 중에 사용자가 그 열을 명시적으로 누르거나,
//   디바이더를 잡은 채 창이 줄어드는 때다. 그 구간이 어떤 하네스에도 없었다.
//
// 재는 것(정지 상태가 아니라 조작 직후):
//   1) 창 최소 크기에서 스왑 → 접기 → 펴기 연타(간격 없이 동시 발사)
//   2) 자동 접힘 중에 그 열의 코너 버튼을 명시적으로 누름 → «사용자가 끔»으로 내려가고,
//      창을 넓혀도 되살아나지 않아야 한다(자동 접힘과 사용자 의사가 섞이면 여기서 드러난다)
//   3) 디바이더 드래그를 놓지 않은 채 창 크기 변경 → 저장 폭이 NaN·음수·극단값이 되지 않아야 한다
//   4) 좁힘·확대를 반복해 자동 접힘 전이를 여러 번 → 알림 칩이 중복 생성되지 않아야 한다
//   공통 불변식: 열 겹침 0 / 창 밖 이탈 0 / 0폭 생존 0 / 인라인 폭에 NaN·음수 없음 / 중앙 열 생존.
//
// 지키는 커밋: 0b17853(PiP 3열 재작업 — 열·자동 접힘 구조의 출처),
//              3f91df5(표면 분리), 7d5a883(레이아웃 판정 교정).
//
// 서버는 읽기만 한다. 실행:
//   node scripts/pip-columns-stress-check.mjs [videoId] [serverUrl]
import { launchE2E, settingsIO, makeCheck, shotPath, LOCAL_SERVER, isLocalServer } from './lib/e2e-launch.mjs';
import { resolveVideoId } from './lib/pick-song.mjs';

const picked = resolveVideoId(process.argv[2] ?? 'kYwB-kZyNU4', { minLines: 20 });
const VID = picked.videoId;
const SERVER = process.argv[3] ?? LOCAL_SERVER;
// 프로드에는 쓰기를 태우지 않는다 — 번역 표시는 없는 언어면 생성 요청으로 이어질 수 있다
const WRITE_SAFE = isLocalServer(SERVER);
const { check, failures } = makeCheck();
console.log(`[songs] ${VID} (${picked.source}) — ${picked.note}`);

/** 코너 툴바 버튼 순서(pip.ts의 코너 미니 버튼 배열과 같은 순서) */
const CORNER = { lane: 0, center: 1, video: 2, short: 3, panel: 4, playlist: 5, dual: 6 };

const MEASURE = `(() => {
  const box = (el) => {
    if (!el) return null;
    const cs = getComputedStyle(el);
    if (cs.display === 'none') return null;
    const b = el.getBoundingClientRect();
    if (b.width === 0 && b.height === 0) return null;
    return { x: Math.round(b.x), y: Math.round(b.y), w: Math.round(b.width), h: Math.round(b.height),
             r: Math.round(b.right), bt: Math.round(b.bottom),
             // 인라인 폭에 NaN/음수가 박히면 여기서 문자열째 잡힌다
             styleW: el.style.width || el.style.flexBasis || '' };
  };
  const cols = {
    lane: box(document.querySelector('.ey-pip-lane-col')),
    center: box(document.querySelector('.ey-pip-center')),
    panel: box(document.getElementById('everyric-root')),
    playlist: box(document.querySelector('.ey-pip-playlist-col')),
  };
  const shown = Object.entries(cols).filter(([, v]) => v);
  let overlap = null;
  for (let i = 0; i < shown.length; i++) for (let j = i + 1; j < shown.length; j++) {
    const [an, a] = shown[i], [bn, b] = shown[j];
    if (!(a.r <= b.x || b.r <= a.x || a.bt <= b.y || b.bt <= a.y)) overlap = an + '×' + bn;
  }
  const root = document.getElementById('everyric-root')?.shadowRoot;
  return {
    win: { w: innerWidth, h: innerHeight },
    cols, shownNames: shown.map(([k]) => k), overlap,
    outside: shown.filter(([, v]) => v.r > innerWidth + 1 || v.x < -1).map(([k]) => k),
    zeroWidth: shown.filter(([, v]) => v.w <= 0).map(([k]) => k),
    badStyle: Object.entries(cols).filter(([, v]) => v && /NaN|-\\d/.test(v.styleW)).map(([k, v]) => k + '=' + v.styleW),
    // 알림 칩은 **하나뿐이어야** 한다(중복 생성 감시)
    noticeChips: root ? root.querySelectorAll('.ey-notice-chip').length : -1,
    noticeText: (() => {
      const c = root?.querySelector('.ey-notice-chip');
      return c && c.style.display !== 'none' ? c.textContent.trim().slice(0, 40) : '';
    })(),
    cornerStates: [...document.querySelectorAll('.ey-pip-corner .ey-pip-mini')]
      .map(b => (b.classList.contains('on') ? '1' : '0') + (b.classList.contains('ey-pip-mini-auto') ? 'A' : '')).join(','),
  };
})()`;

const { ctx, sw } = await launchE2E({ name: 'pip-columns-stress', serverUrl: SERVER, width: 1600, height: 1000 });
const io = settingsIO(sw);
const shots = [];
try {
  await io.patch({
    serverUrl: SERVER, uiLanguage: 'ko', theme: 'dark',
    showTranslation: WRITE_SAFE, translationLanguage: 'ko', showPronunciation: true,
    pitchGuide: true, pipPlaylist: true, pipShortLyrics: true, pipShowPanel: true, pipShowCenter: true,
    pipLaneWidth: 300, pipPanelWidth: 360, pipLaneSwapped: false,
    pipWidth: 1200, pipHeight: 700,
  });

  const page = ctx.pages()[0] ?? await ctx.newPage();
  await page.goto(`https://www.youtube.com/watch?v=${VID}`, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForSelector('#everyric-root', { state: 'attached', timeout: 30000 });
  await page.waitForFunction(() => {
    const r = document.getElementById('everyric-root')?.shadowRoot;
    return (r?.querySelectorAll('.ey-word').length ?? 0) > 0;
  }, null, { timeout: 90000, polling: 1000 });
  await page.evaluate(() => {
    const v = document.querySelector('video.html5-main-video') ?? document.querySelector('video');
    if (v) { v.currentTime = 40; void v.play().catch(() => {}); }
  });
  await page.waitForTimeout(1200);
  await page.locator('#everyric-root [title="PiP 창으로 보기"]').first().click();
  await page.waitForTimeout(3500);
  const pp = ctx.pages().find(p => p !== page);
  if (!pp) throw new Error('PiP 창을 페이지로 못 잡음');

  const clickCorner = (i) => pp.evaluate((idx) => {
    document.querySelectorAll('.ey-pip-corner .ey-pip-mini')[idx]?.click();
  }, i);
  const verify = (label, m) => {
    check(m.overlap === null, `${label}: 열끼리 겹침 없음`, m.overlap ?? `보이는 열 ${m.shownNames.join('|')}`);
    check(m.outside.length === 0, `${label}: 창 밖으로 나간 열 없음`, m.outside);
    check(m.zeroWidth.length === 0, `${label}: 0폭으로 살아 있는 열 없음`, m.zeroWidth);
    check(m.badStyle.length === 0, `${label}: 인라인 폭에 NaN·음수 없음`, m.badStyle);
    check(m.cols.center !== null, `${label}: 중앙 열 생존(최후 보루)`, m.shownNames);
    check(m.noticeChips === 1, `${label}: 알림 칩이 하나뿐(중복 생성 없음)`, m.noticeChips);
  };

  // ── 1) 최소 크기에서 스왑·접기·펴기 연타 ─────────────────────────
  console.log('\n══ 1) 창 최소 크기에서 연타');
  await pp.setViewportSize({ width: 320, height: 260 });
  await pp.waitForTimeout(900);
  const mSmall = await pp.evaluate(MEASURE);
  verify('최소크기 정지상태', mSmall);
  console.log(`  (자동 접힘 결과: 보이는 열 ${mSmall.shownNames.join('|')}, 코너 ${mSmall.cornerStates})`);

  for (let round = 0; round < 3; round++) {
    await io.patch({ pipLaneSwapped: round % 2 === 0 });
    await Promise.all([clickCorner(CORNER.lane), clickCorner(CORNER.panel), clickCorner(CORNER.playlist)]);
    await pp.waitForTimeout(120);
    await Promise.all([clickCorner(CORNER.lane), clickCorner(CORNER.panel), clickCorner(CORNER.playlist)]);
    await pp.waitForTimeout(120);
  }
  await pp.waitForTimeout(1500);
  verify('연타 직후(최소크기)', await pp.evaluate(MEASURE));
  const shot1 = shotPath('pip-columns-burst-small'); await pp.screenshot({ path: shot1 }); shots.push(shot1);

  // 넓히면 접혔던 것이 되살아나는가 — 연타가 저장값을 오염시켰다면 여기서 드러난다
  await pp.setViewportSize({ width: 1280, height: 720 });
  await pp.waitForTimeout(1500);
  verify('연타 후 창 확대', await pp.evaluate(MEASURE));
  const st1 = await io.read();
  check(Number.isFinite(st1.pipLaneWidth) && st1.pipLaneWidth > 0
    && Number.isFinite(st1.pipPanelWidth) && st1.pipPanelWidth > 0,
    '연타가 저장된 열 폭을 망가뜨리지 않았다', { lane: st1.pipLaneWidth, panel: st1.pipPanelWidth });

  // ── 2) 자동 접힘 중에 그 열을 명시적으로 토글 ────────────────────
  console.log('\n══ 2) 자동 접힘 중 명시 토글');
  await io.patch({ pitchGuide: true, pipPlaylist: true, pipShowPanel: true });
  await pp.waitForTimeout(1200);
  await pp.setViewportSize({ width: 420, height: 320 }); // 좁혀서 자동 접힘 유발
  await pp.waitForTimeout(1200);
  const mAuto = await pp.evaluate(MEASURE);
  check(mAuto.cornerStates.includes('A'), '전제: 좁은 창에서 자동 접힘 배지가 켜졌다', mAuto.cornerStates);
  const stBeforeAuto = await io.read();
  check(stBeforeAuto.pipPlaylist === true && stBeforeAuto.pitchGuide === true,
    '자동 접힘은 설정을 건드리지 않는다', { pipPlaylist: stBeforeAuto.pipPlaylist, pitchGuide: stBeforeAuto.pitchGuide });

  await clickCorner(CORNER.playlist);
  await pp.waitForTimeout(1200);
  const stAfterClick = await io.read();
  check(stAfterClick.pipPlaylist === false,
    '자동 접힘 중 코너를 누르면 «사용자가 끈 것»으로 저장된다', stAfterClick.pipPlaylist);
  await pp.setViewportSize({ width: 1280, height: 720 });
  await pp.waitForTimeout(1500);
  const mReopen = await pp.evaluate(MEASURE);
  verify('자동 접힘 → 명시 OFF → 확대', mReopen);
  check(mReopen.cols.playlist === null,
    '사용자가 끈 열은 창을 넓혀도 되살아나지 않는다', mReopen.shownNames);
  await clickCorner(CORNER.playlist);
  await pp.waitForTimeout(1200);
  const mBack = await pp.evaluate(MEASURE);
  check(mBack.cols.playlist !== null, '다시 누르면 돌아온다', mBack.shownNames);
  verify('재점등', mBack);

  // ── 3) 디바이더 드래그 도중 창 크기 변경 ────────────────────────
  console.log('\n══ 3) 디바이더 드래그 중 창 리사이즈');
  await pp.setViewportSize({ width: 1280, height: 720 });
  await pp.waitForTimeout(1000);
  const divBox = await pp.evaluate(() => {
    const d = document.querySelector('.ey-pip-vdivider');
    if (!d) return null;
    const b = d.getBoundingClientRect();
    return { x: b.x + b.width / 2, y: b.y + b.height / 2 };
  });
  check(!!divBox, '전제: 세로 디바이더가 있다', divBox);
  if (divBox) {
    await pp.mouse.move(divBox.x, divBox.y);
    await pp.mouse.down();
    await pp.mouse.move(divBox.x - 60, divBox.y, { steps: 5 });
    // **드래그를 놓지 않은 채** 창을 줄인다 — 포인터 좌표계와 창 폭이 동시에 바뀐다
    await pp.setViewportSize({ width: 760, height: 560 });
    await pp.waitForTimeout(300);
    await pp.mouse.move(divBox.x - 200, divBox.y, { steps: 5 });
    await pp.waitForTimeout(200);
    await pp.mouse.up();
    await pp.waitForTimeout(1500);
    verify('드래그 중 리사이즈 후', await pp.evaluate(MEASURE));
    const st3 = await io.read();
    check(Number.isFinite(st3.pipLaneWidth) && st3.pipLaneWidth >= 100 && st3.pipLaneWidth <= 2000,
      '드래그·리사이즈 경합 뒤에도 저장된 레인 폭이 제정신 범위', st3.pipLaneWidth);
    check(Number.isFinite(st3.pipPanelWidth) && st3.pipPanelWidth >= 100 && st3.pipPanelWidth <= 2000,
      '저장된 가사창 폭도 제정신 범위', st3.pipPanelWidth);
    const shot2 = shotPath('pip-columns-drag-resize'); await pp.screenshot({ path: shot2 }); shots.push(shot2);
  }

  // ── 4) 자동 접힘 전이 반복 — 알림 칩 중복 ────────────────────────
  console.log('\n══ 4) 자동 접힘 알림 중복 여부');
  for (let i = 0; i < 3; i++) {
    await pp.setViewportSize({ width: 380, height: 300 });
    await pp.waitForTimeout(900);
    await pp.setViewportSize({ width: 1280, height: 720 });
    await pp.waitForTimeout(900);
  }
  await pp.setViewportSize({ width: 380, height: 300 });
  await pp.waitForTimeout(1200);
  const mChip = await pp.evaluate(MEASURE);
  check(mChip.noticeChips === 1, '반복 전이 후에도 알림 칩 엘리먼트는 하나', mChip.noticeChips);
  console.log(`  (알림 문구: "${mChip.noticeText}")`);
  verify('반복 전이 후', mChip);
  const shot3 = shotPath('pip-columns-chip'); await pp.screenshot({ path: shot3 }); shots.push(shot3);

  console.log('\nscreenshots:');
  for (const f of shots) console.log('  ' + f);
  if (failures.length) { console.log('\n실패 항목:'); for (const r of failures) console.log('  - ' + r); }
  console.log(failures.length ? '\nPIP COLUMNS STRESS: FAIL' : '\nPIP COLUMNS STRESS: PASS');
  process.exitCode = failures.length ? 1 : 0;
} catch (e) {
  console.log('PIP COLUMNS STRESS: ERROR —', String(e?.stack ?? e).slice(0, 600));
  process.exitCode = 1;
} finally {
  await ctx.close();
}
