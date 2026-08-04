// 시트를 열 때 본문 스크롤이 처음으로 돌아가는가 — 3f91df5(감사 A1 D1)의 회귀 감시.
//
// 무엇을 왜 재는가:
//   시트(검색·설정)는 본문(.ey-body)을 replaceChildren으로 갈아끼워 연다. 예전에는 그때
//   body.scrollTop을 되돌리지 않아, 노래를 듣는 동안 카라오케 자동 스크롤로 내려가 있던
//   위치 그대로 시트가 열렸다. 실측 당시 시트 머리가 y=-338px, 즉 창 위로 밀려 나갔고
//   사용자는 **검색 입력칸과 «보던 가사로 돌아가기» 버튼**을 동시에 잃었다 — 돌아갈 유일한
//   길이 화면 밖이라 스크롤을 올리기 전에는 빠져나올 수 없었다.
//   «시트가 열린다»는 DOM 존재만으로는 이 결함이 안 잡힌다. 그래서 여는 것이 아니라
//   **열린 뒤 그 조각이 화면 안에 있는가**를 잰다.
//
// 재는 것: 자동 스크롤로 본문을 충분히 내려 놓은 상태에서 시트를 열고
//   (a) body.scrollTop === 0  (b) «돌아가기» 버튼이 가시 영역 안  (c) 검색 폼이 가시 영역 안.
//   메인 창과 PiP 창 **양쪽**에서 본다(같은 overlay 코드가 두 표면을 그리므로, 한쪽만
//   고쳐지는 회귀를 여기서 잡는다). 설정 시트도 같은 resetBody 경로라 함께 확인한다.
//
// 실행: node scripts/sheet-scroll-reset-check.mjs [videoId] [serverUrl]
import { launchE2E, settingsIO, makeCheck, shotPath, LOCAL_SERVER, isLocalServer } from './lib/e2e-launch.mjs';
import { resolveVideoId } from './lib/pick-song.mjs';

const picked = resolveVideoId(process.argv[2] ?? 'kYwB-kZyNU4', { minLines: 20 });
const VID = picked.videoId;
const SERVER = process.argv[3] ?? LOCAL_SERVER;
const WRITE_SAFE = isLocalServer(SERVER);
const { check, failures } = makeCheck();
console.log(`[songs] ${VID} (${picked.source}) — ${picked.note}`);

/** 시트 조각이 «본문 가시 영역 안»에 걸쳐 있는가 — 존재 여부가 아니라 보이는지를 잰다 */
const PROBE = `(() => {
  const root = document.getElementById('everyric-root').shadowRoot;
  const body = root.querySelector('.ey-body');
  const bb = body.getBoundingClientRect();
  const back = root.querySelector('.ey-search-back, .ey-sheet-back');
  const form = root.querySelector('.ey-search-form');
  const r = (el) => { if (!el) return null; const b = el.getBoundingClientRect();
    return { y: Math.round(b.y), bottom: Math.round(b.bottom) }; };
  return {
    scrollTop: Math.round(body.scrollTop),
    scrollHeight: Math.round(body.scrollHeight),
    bodyTop: Math.round(bb.y), bodyBottom: Math.round(bb.bottom),
    back: r(back), form: r(form),
    backVisible: back ? (r(back).bottom > bb.y && r(back).y < bb.bottom) : null,
    formVisible: form ? (r(form).bottom > bb.y && r(form).y < bb.bottom) : null,
  };
})()`;

const SCROLL_TOP = `(() => {
  const b = document.getElementById('everyric-root').shadowRoot.querySelector('.ey-body');
  return Math.round(b.scrollTop);
})()`;

const { ctx, sw } = await launchE2E({ name: 'sheet-scroll-reset', serverUrl: SERVER, width: 1500, height: 950 });
const io = settingsIO(sw);
const shots = [];
try {
  await io.patch({
    serverUrl: SERVER, uiLanguage: 'ko', theme: 'dark',
    showTranslation: WRITE_SAFE, showPronunciation: true,
    pitchGuide: true, pipPlaylist: false, pipShortLyrics: true, pipShowPanel: true, pipShowCenter: true,
    pipKeepPanel: true, pipWidth: 1100, pipHeight: 700,
  });

  const page = ctx.pages()[0] ?? await ctx.newPage();
  await page.goto(`https://www.youtube.com/watch?v=${VID}`, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForSelector('#everyric-root', { state: 'attached', timeout: 30000 });
  await page.waitForFunction(() => {
    const r = document.getElementById('everyric-root')?.shadowRoot;
    return (r?.querySelectorAll('.ey-word').length ?? 0) > 0;
  }, null, { timeout: 90000, polling: 1000 });

  // 곡을 뒤쪽으로 보낸다 — 카라오케 자동 스크롤이 본문을 내려 놓아야 이 검사가 성립한다
  await page.evaluate(() => {
    const v = document.querySelector('video.html5-main-video') ?? document.querySelector('video');
    if (v) { v.currentTime = Math.max(0, (v.duration || 200) - 30); void v.play().catch(() => {}); }
  });
  await page.waitForTimeout(6000);

  const openSearchIn = (target) => target.evaluate(() => {
    document.getElementById('everyric-root').shadowRoot
      .querySelectorAll('.ey-actions .ey-btn')[2].click(); // 검색
  });

  // ── 메인 창 ────────────────────────────────────────────────────
  const beforeMain = await page.evaluate(SCROLL_TOP);
  check(beforeMain > 0, '전제: 자동 스크롤로 본문이 내려가 있다(메인)', beforeMain);
  await openSearchIn(page);
  await page.waitForTimeout(1200);
  const mainSheet = await page.evaluate(PROBE);
  console.log('  메인 시트:', JSON.stringify(mainSheet));
  check(mainSheet.scrollTop === 0, '메인: 시트를 열면 본문 스크롤이 처음으로 돌아간다', mainSheet.scrollTop);
  check(mainSheet.backVisible === true, '메인: «보던 가사로 돌아가기» 버튼이 화면 안에 있다', mainSheet.back);
  const f0 = shotPath('sheet-scroll-main'); await page.screenshot({ path: f0 }); shots.push(f0);

  // ── PiP 창 ─────────────────────────────────────────────────────
  // 시트가 열려 있는 동안은 PiP 버튼이 숨는다 — 뒤로 눌러 가사 화면으로 되돌린 뒤 연다.
  await page.evaluate(() => {
    const root = document.getElementById('everyric-root').shadowRoot;
    root.querySelector('.ey-search-back, .ey-sheet-back')?.click();
    const b = root.querySelector('.ey-body');
    if (b) b.scrollTop = 0;
  });
  await page.waitForTimeout(1500);
  await page.locator('#everyric-root [title="PiP 창으로 보기"]').first().click();
  await page.waitForTimeout(3500);
  const pp = ctx.pages().find(p => p !== page);
  if (!pp) throw new Error('PiP 창 없음');
  await pp.setViewportSize({ width: 1100, height: 700 });
  await pp.waitForTimeout(4000); // 재생 중 대기 — PiP 본문도 자동 스크롤로 내려가게 한다

  const beforePip = await pp.evaluate(SCROLL_TOP);
  check(beforePip > 0, '전제: PiP 본문도 내려가 있다', beforePip);
  await openSearchIn(pp);
  await pp.waitForTimeout(1200);
  const pipSheet = await pp.evaluate(PROBE);
  console.log('  PiP 시트:', JSON.stringify(pipSheet));
  check(pipSheet.scrollTop === 0, 'PiP: 시트를 열면 본문 스크롤이 처음으로 돌아간다', pipSheet.scrollTop);
  check(pipSheet.backVisible === true, 'PiP: «보던 가사로 돌아가기» 버튼이 화면 안에 있다', pipSheet.back);
  check(pipSheet.formVisible === true, 'PiP: 검색 입력 폼이 화면 안에 있다', pipSheet.form);
  const f1 = shotPath('sheet-scroll-pip'); await pp.screenshot({ path: f1 }); shots.push(f1);

  // ── 설정 시트도 같은 resetBody 경로다 ───────────────────────────
  await pp.evaluate(() => {
    document.getElementById('everyric-root').shadowRoot
      .querySelector('.ey-search-back, .ey-sheet-back')?.click();
  });
  await pp.waitForTimeout(2500); // 자동 스크롤이 다시 내려가도록
  const beforeSet = await pp.evaluate(SCROLL_TOP);
  await pp.evaluate(() => {
    document.getElementById('everyric-root').shadowRoot
      .querySelectorAll('.ey-actions .ey-btn')[5].click(); // 설정
  });
  await pp.waitForTimeout(1200);
  const setProbe = await pp.evaluate(`(() => {
    const root = document.getElementById('everyric-root').shadowRoot;
    const body = root.querySelector('.ey-body');
    const bb = body.getBoundingClientRect();
    const filter = root.querySelector('.ey-settings-filter');
    const fb = filter?.getBoundingClientRect();
    return {
      scrollTop: Math.round(body.scrollTop),
      filterY: fb ? Math.round(fb.y) : null,
      filterVisible: fb ? (fb.bottom > bb.y && fb.y < bb.bottom) : null,
    };
  })()`);
  console.log(`  설정 시트: 열기 전 scrollTop=${beforeSet} → ${JSON.stringify(setProbe)}`);
  check(setProbe.filterVisible === true, 'PiP: 설정 시트의 검색칸(맨 위)이 화면 안에 있다', setProbe);
  const f2 = shotPath('sheet-scroll-pip-settings'); await pp.screenshot({ path: f2 }); shots.push(f2);

  console.log('\nscreenshots:');
  for (const f of shots) console.log('  ' + f);
  if (failures.length) { console.log('\n실패 항목:'); for (const r of failures) console.log('  - ' + r); }
  console.log(failures.length ? '\nSHEET SCROLL RESET: FAIL' : '\nSHEET SCROLL RESET: PASS');
  process.exitCode = failures.length ? 1 : 0;
} catch (e) {
  console.log('SHEET SCROLL RESET: ERROR —', String(e?.stack ?? e).slice(0, 500));
  process.exitCode = 1;
} finally {
  await ctx.close();
}
