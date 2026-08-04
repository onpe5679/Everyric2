// 1회성 설정 마이그레이션의 «실브라우저 왕복» + 새 진입에서 표면 상태 복원.
//
// 무엇을 왜 재는가:
//   settings-migration.test.mjs는 마이그레이션 함수의 **시간 성질**(한 번만 돈다, 마커가
//   남는다)을 순수 로직으로 잰다. 실제 확장에서는 그 사이에 저장소가 낀다 —
//   storage.local → getSettings → 되쓰기 → storage.onChanged 가 한 바퀴 돌고, 그 왕복이
//   사용자의 재설정을 덮어쓸 수 있다. 순수 테스트가 절대 못 보는 자리가 거기다:
//   «마이그레이션이 내린 값을 사용자가 다시 켰는데, 다음 로드가 도로 내리는가».
//
// 재는 것:
//   1) 마커 없는 옛 프로필을 심고 페이지를 열면 → 값이 내려가고 마커가 남는다
//   2) 사용자가 **설정 시트에서 실제로** 다시 켠다 → 저장소에 반영된다
//   3) 페이지 리로드 → 사용자가 켠 값이 유지되고 마커가 중복 누적되지 않는다
//   4) 새 탭으로 재진입(콘텐츠 스크립트 새 주입) → 표면별 상태(메인 레인·재생목록 /
//      PiP 스왑·재생목록)가 저장값과 화면 양쪽에서 그대로다. 확장 업데이트 직후 사용자가
//      겪는 것이 «새로 주입된 화면»이라 그 자리를 대신 잰다(아래 주석에 근거).
//
// **다음 마이그레이션 때 이 파일을 재사용한다.** 곡·서버처럼 마이그레이션도 인자다:
//   node scripts/settings-migration-live-check.mjs [videoId] [serverUrl] \
//        [--marker <id>] [--key <설정키>] [--from <옛값>] [--to <마이그레이션 후 값>]
// 기본값은 현행 1건(pipKeepPanel). 마이그레이션이 바뀌면 인자만 갈아 끼우면 된다.
//
// 지키는 커밋: 0b17853(표면별 상태 + settingsMigrations 도입), 4589c9a(pipKeepPanel 신 UX).
import { launchE2E, settingsIO, makeCheck, shotPath, LOCAL_SERVER, isLocalServer } from './lib/e2e-launch.mjs';
import { resolveVideoId } from './lib/pick-song.mjs';

const FLAGS = ['marker', 'key', 'from', 'to'];
const argv = process.argv.slice(2);
const arg = (n, d) => { const i = argv.indexOf(`--${n}`); return i >= 0 && argv[i + 1] ? argv[i + 1] : d; };
// 플래그와 그 값을 걷어낸 나머지가 위치 인자(videoId, serverUrl)다
const positional = [];
for (let i = 0; i < argv.length; i++) {
  if (argv[i].startsWith('--')) { if (FLAGS.includes(argv[i].slice(2))) i++; continue; }
  positional.push(argv[i]);
}
const picked = resolveVideoId(positional[0] ?? 'kYwB-kZyNU4', { minLines: 20 });
const VID = picked.videoId;
const SERVER = positional[1] ?? LOCAL_SERVER;
const WRITE_SAFE = isLocalServer(SERVER);

/** 검사 대상 마이그레이션 1건 — 여기만 바꾸면 다음 마이그레이션에 그대로 쓴다 */
const MIGRATION = {
  marker: arg('marker', 'pipKeepPanel-default-false-2026-08'),
  key: arg('key', 'pipKeepPanel'),
  legacy: JSON.parse(arg('from', 'true')),    // 옛 프로필이 갖고 있던 값
  migrated: JSON.parse(arg('to', 'false')),   // 마이그레이션이 내려놓아야 하는 값
};
const { check, failures } = makeCheck();
console.log(`[songs] ${VID} (${picked.source}) — ${picked.note}`);
console.log(`[migration] ${MIGRATION.key}: ${JSON.stringify(MIGRATION.legacy)} → ${JSON.stringify(MIGRATION.migrated)} (marker ${MIGRATION.marker})`);

const { ctx, sw } = await launchE2E({ name: 'settings-migration-live', serverUrl: SERVER, width: 1500, height: 950 });
const io = settingsIO(sw);
const shots = [];
try {
  // ── 마커 없는 «옛 프로필»을 심는다 — 마커가 없다는 것이 옛 프로필의 정의다
  await io.replace({
    serverUrl: SERVER, uiLanguage: 'ko', theme: 'dark',
    showTranslation: WRITE_SAFE, showPronunciation: true,
    [MIGRATION.key]: MIGRATION.legacy,
    pitchGuide: true, pipPlaylist: true, pipLaneSwapped: true, modMainLane: false, modPlaylist: true,
    pipLaneWidth: 280, pipPanelWidth: 400,
  });
  const seeded = await io.read();
  check(JSON.stringify(seeded[MIGRATION.key]) === JSON.stringify(MIGRATION.legacy) && !seeded.settingsMigrations,
    '전제: 마커 없는 옛 프로필을 심었다',
    { [MIGRATION.key]: seeded[MIGRATION.key], marks: seeded.settingsMigrations });

  let page = ctx.pages()[0] ?? await ctx.newPage();
  await page.goto(`https://www.youtube.com/watch?v=${VID}`, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForSelector('#everyric-root', { state: 'attached', timeout: 30000 });
  await page.waitForTimeout(4000);

  const migrated = await io.read();
  check(JSON.stringify(migrated[MIGRATION.key]) === JSON.stringify(MIGRATION.migrated),
    `1회 마이그레이션이 ${MIGRATION.key}를 옮겼다`, migrated[MIGRATION.key]);
  check(Array.isArray(migrated.settingsMigrations) && migrated.settingsMigrations.includes(MIGRATION.marker),
    '마이그레이션 마커가 저장소에 남았다', migrated.settingsMigrations);

  // ── 사용자가 설정 시트에서 **실제로** 되돌린다 ───────────────────
  await page.waitForFunction(() => {
    const r = document.getElementById('everyric-root')?.shadowRoot;
    return (r?.querySelectorAll('.ey-word').length ?? 0) > 0;
  }, null, { timeout: 90000, polling: 1000 }).catch(() => console.log('  (가사 대기 초과 — 설정 조작은 계속)'));
  const toggled = await page.evaluate((key) => {
    const root = document.getElementById('everyric-root').shadowRoot;
    root.querySelectorAll('.ey-actions .ey-btn')[5].click(); // 설정
    return new Promise(res => setTimeout(() => {
      const filter = root.querySelector('.ey-settings-filter');
      if (!filter) return res({ ok: false, why: '설정 시트 없음' });
      filter.value = key; // 설정 검색은 키 이름도 훑는다
      filter.dispatchEvent(new Event('input', { bubbles: true }));
      const rows = [...root.querySelectorAll('.ey-settings-sections .ey-settings-row')]
        .filter(r => getComputedStyle(r).display !== 'none');
      const box = rows.map(r => r.querySelector('input[type=checkbox]')).find(Boolean);
      if (!box) return res({ ok: false, why: '행 없음', rows: rows.length });
      const before = box.checked;
      box.click();
      res({ ok: true, before, after: box.checked });
    }, 900));
  }, MIGRATION.key);
  check(toggled.ok && JSON.stringify(toggled.after) === JSON.stringify(MIGRATION.legacy),
    `사용자가 설정 시트에서 ${MIGRATION.key}를 되돌렸다`, toggled);
  await page.waitForTimeout(1500);
  const reEnabled = await io.read();
  check(JSON.stringify(reEnabled[MIGRATION.key]) === JSON.stringify(MIGRATION.legacy),
    '되돌린 값이 저장소에 반영됐다', reEnabled[MIGRATION.key]);

  // ── 페이지 리로드 — 마이그레이션이 다시 돌아 덮으면 안 된다 ──────
  await page.reload({ waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForSelector('#everyric-root', { state: 'attached', timeout: 30000 });
  await page.waitForTimeout(4000);
  const afterReload = await io.read();
  check(JSON.stringify(afterReload[MIGRATION.key]) === JSON.stringify(MIGRATION.legacy),
    '리로드 뒤에도 사용자가 되돌린 값이 유지된다(마이그레이션 재실행 없음)', afterReload[MIGRATION.key]);
  check((afterReload.settingsMigrations ?? []).filter(m => m === MIGRATION.marker).length === 1,
    '마커가 중복 누적되지 않는다', afterReload.settingsMigrations);

  // ── 표면별 상태를 구분되게 만든 뒤 «새 진입»에서 복원되는지 ──────
  //
  // 원래 이 자리는 chrome.runtime.reload()(확장 업데이트 재현)였다. 그런데 Playwright
  // 지속 컨텍스트에서는 리로드 뒤 확장이 다시 주입되지 않는다 — 실측: about:blank를 거쳐
  // 재진입을 3회(각 25초) 시도해도 #everyric-root가 끝내 안 생겼다. 자동화 환경의 한계이지
  // 제품 결함이 아니므로, **그걸 FAIL로 남기면 이 게이트는 늘 빨간불이 되어 무시당한다.**
  // 대신 같은 성질을 흔들리지 않는 방법으로 잰다: 표면별 상태를 저장한 뒤 **새 탭으로 다시
  // 들어가** 콘텐츠 스크립트가 처음부터 주입되게 하고, 저장값과 «그려진 화면»이 모두
  // 그대로인지 본다. 업데이트 직후 사용자가 겪는 것도 결국 «새로 주입된 화면»이다.
  console.log('\n══ 새 진입에서 표면 상태 복원');
  await io.patch({
    modMainLane: false, modPlaylist: true,
    pitchGuide: true, pipPlaylist: false, pipLaneSwapped: true,
    pipLaneWidth: 260, pipPanelWidth: 420, pipShortLyrics: true, pipShowPanel: true,
  });
  await page.waitForTimeout(1200);
  const beforeReload = await io.read();

  await page.close();
  page = await ctx.newPage();
  await page.goto(`https://www.youtube.com/watch?v=${VID}`, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForSelector('#everyric-root', { state: 'attached', timeout: 40000 });
  await page.waitForFunction(() => {
    const r = document.getElementById('everyric-root')?.shadowRoot;
    return (r?.querySelectorAll('.ey-word').length ?? 0) > 0;
  }, null, { timeout: 90000, polling: 1000 }).catch(() => console.log('  (새 탭 가사 대기 초과)'));
  await page.waitForTimeout(2000);

  const afterExtReload = await io.read();
  const keys = [MIGRATION.key, 'modMainLane', 'modPlaylist', 'pitchGuide', 'pipPlaylist',
    'pipLaneSwapped', 'pipLaneWidth', 'pipPanelWidth'];
  const diff = keys.filter(k => JSON.stringify(afterExtReload[k]) !== JSON.stringify(beforeReload[k]))
    .map(k => `${k}: ${JSON.stringify(beforeReload[k])}→${JSON.stringify(afterExtReload[k])}`);
  check(diff.length === 0, '새 진입 뒤에도 표면별 설정이 그대로다', diff);

  // 저장값만이 아니라 화면도 본다 — 메인 레인 꺼짐 / 재생목록 켜짐
  const mainState = await page.evaluate(() => {
    const root = document.getElementById('everyric-root')?.shadowRoot;
    const vis = (sel) => {
      const el = document.querySelector(sel) ?? root?.querySelector(sel);
      return el ? getComputedStyle(el).display !== 'none' : false;
    };
    return { lane: vis('.ey-lane-wrap'), playlist: vis('.ey-attach-playlist') };
  });
  check(mainState.lane === false && mainState.playlist === true,
    '새 진입 후 메인 표면 화면 상태도 그대로(레인 off / 재생목록 on)', mainState);

  await page.evaluate(() => {
    const v = document.querySelector('video.html5-main-video') ?? document.querySelector('video');
    if (v) { v.currentTime = 30; void v.play().catch(() => {}); }
  });
  await page.waitForTimeout(1200);
  await page.locator('#everyric-root [title="PiP 창으로 보기"]').first().click();
  await page.waitForTimeout(4000);
  const pp = ctx.pages().find(p => p !== page);
  if (pp) {
    await pp.setViewportSize({ width: 1200, height: 700 });
    await pp.waitForTimeout(1200);
    const pipState = await pp.evaluate(() => {
      const shown = (sel) => {
        const el = document.querySelector(sel);
        if (!el || getComputedStyle(el).display === 'none') return null;
        const b = el.getBoundingClientRect();
        return b.width > 0 ? Math.round(b.x) : null;
      };
      return { laneX: shown('.ey-pip-lane-col'), centerX: shown('.ey-pip-center'),
               playlistX: shown('.ey-pip-playlist-col') };
    });
    check(pipState.playlistX === null, '새 진입 후 PiP 재생목록은 여전히 꺼져 있다', pipState);
    check(pipState.laneX !== null && pipState.centerX !== null && pipState.laneX > pipState.centerX,
      '새 진입 후 PiP 레인 스왑(레인이 중앙 오른쪽)이 유지된다', pipState);
    const f = shotPath('settings-migration-after-reload'); await pp.screenshot({ path: f }); shots.push(f);
  } else {
    check(false, '새 진입 후 PiP를 다시 열 수 있다', 'PiP 창 없음');
  }

  console.log('\nscreenshots:');
  for (const f of shots) console.log('  ' + f);
  if (failures.length) { console.log('\n실패 항목:'); for (const r of failures) console.log('  - ' + r); }
  console.log(failures.length ? '\nSETTINGS MIGRATION LIVE: FAIL' : '\nSETTINGS MIGRATION LIVE: PASS');
  process.exitCode = failures.length ? 1 : 0;
} catch (e) {
  console.log('SETTINGS MIGRATION LIVE: ERROR —', String(e?.stack ?? e).slice(0, 600));
  process.exitCode = 1;
} finally {
  await ctx.close();
}
