// 사용자 시나리오 재현: vocaro를 거친 적 없는 "새 프로필"로 실곡 영상을 열었을 때
//   - 서버 싱크에 저장된 발음/사람 번역이 그대로 표시되는지
//   - 번역 표시 ON이어도 [NO API KEY]가 나타나지 않는지 (사람 번역 우선)
//   - PiP 음정 바(신규 UI)가 그려지고, pitchGuide 설정으로 껐다 켤 수 있는지
// 사전 조건: 실서버(:8000) + 해당 곡 싱크(발음/번역/notes 포함)가 DB에 있어야 한다.
// 실행: node scripts/fresh-profile-check.mjs [videoUrl]
//   videoUrl 생략 시 everyric2.db를 조회해 조건(발음·번역 40줄 이상 + tempo 있음)에 맞는
//   실곡을 자동 선택한다(2026-08-04, 하드코딩 로키 영상 제거 — DB에 없어 ERROR였다).
import { chromium } from 'playwright';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import { mkdtempSync, mkdirSync, cpSync, readFileSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { DatabaseSync } from 'node:sqlite';
import { ensureLocalServerPermissionForServerUrl } from './lib/local-server-permission.mjs';
import { readPipPanel } from './lib/pip-panel.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const distSrc = resolve(__dirname, '../dist');
// 2026-08-04 갱신: 예전엔 ../dist를 직접 로드했다 — 다른 에이전트의 npm run build와
// 겹치면 manifest.json이 갈리는 중이라 크롬이 "매니페스트 없음" 모달을 띄우고 Playwright
// 연결을 영영 막는 위험이 있었다(display-values-check.mjs가 이미 봉인한 함정). 고정
// 목적지에 스냅샷하면 그 위험도 막고, 확장 ID도 고정 userDataDir과 짝이 맞아 host
// permission이 재사용된다(무작위 경로였다면 매번 새 확장 ID라 권한 버블이 되살아난다).
const distDir = process.env.EVERYRIC_E2E_DIST_DIR
  ?? join(tmpdir(), 'everyric-e2e-profiles', 'exec-pron-fresh-profile-dist');
mkdirSync(distDir, { recursive: true });
cpSync(distSrc, distDir, { recursive: true });
JSON.parse(readFileSync(join(distDir, 'manifest.json'), 'utf8')); // 깨진 스냅샷이면 여기서 즉사
const DB_PATH = resolve(__dirname, '../../everyric2.db');

// 조건: 발음 40줄↑ + 번역(세그 내장) 40줄↑ + tempo 있음(PiP 음정 바 렌더 전제) —
// feedback-round3-check.mjs의 DB 조회 패턴과 동일한 방식(생성 아님, 조회만).
function pickFreshSong() {
  const db = new DatabaseSync(DB_PATH, { readOnly: true });
  try {
    const rows = db.prepare(
      'SELECT video_id, language, title, timestamps FROM sync_results ORDER BY id DESC',
    ).all();
    for (const r of rows) {
      let d;
      try { d = JSON.parse(r.timestamps); } catch { continue; }
      const segs = d.segments ?? [];
      if (segs.length < 40 || !d.tempo) continue;
      const pronCount = segs.filter(s => s.pron && Object.keys(s.pron).length > 0).length;
      const trCount = segs.filter(s => s.translation).length;
      if (pronCount < 40 || trCount < 40) continue;
      return { videoId: r.video_id, language: r.language, title: r.title, lines: segs.length };
    }
    return null;
  } finally {
    db.close();
  }
}

let videoUrl = process.argv[2];
let pickedSong = null;
if (!videoUrl) {
  pickedSong = pickFreshSong();
  if (!pickedSong) {
    console.log('FRESH PROFILE CHECK: ERROR — DB 조회 조건(발음·번역 40줄↑ + tempo)에 맞는 실곡이 없음. videoUrl을 인자로 직접 지정하세요.');
    process.exit(1);
  }
  console.log(`INFO: 자동 선택된 실곡 = ${JSON.stringify(pickedSong)}`);
  videoUrl = `https://www.youtube.com/watch?v=${pickedSong.videoId}`;
}
// 병렬 검수 공지(2026-08-04, team-lead): 무작위 임시 프로필은 매번 host permission
// 버블을 다시 띄우고, 그걸 지우려던 taskkill //IM chrome.exe가 다른 에이전트 브라우저까지
// 죽인 실사고가 있었다 — 에이전트 고유의 **고정** user-data-dir을 재사용하면 확장 ID가
// 유지돼 permission이 그대로 살아있고, 버블 자체가 다시 뜨지 않아 taskkill이 필요 없다.
// ("새 프로필" 시나리오 자체는 디스크상 pristine 여부가 아니라 이 videoId의 vocaro
// 이력이 없는 상태를 재현하는 게 목적이라 — 매 실행이 필요한 settings를 명시적으로
// 덮어쓰므로 프로필 재사용과 충돌하지 않는다.)
const userDataDir = process.env.EVERYRIC_E2E_PROFILE_DIR
  ?? join(tmpdir(), 'everyric-e2e-profiles', 'exec-pron-fresh-profile-check');
mkdirSync(userDataDir, { recursive: true });

let failed = false;
function check(ok, label, detail) {
  console.log(`${ok ? 'PASS' : 'FAIL'}: ${label}${detail !== undefined ? ' = ' + JSON.stringify(detail) : ''}`);
  if (!ok) failed = true;
  return ok;
}

const ctx = await chromium.launchPersistentContext(userDataDir, {
  channel: process.env.EVERYRIC_E2E_CHANNEL ?? 'msedge',
  headless: false,
  viewport: { width: 1440, height: 900 },
  args: [
    `--disable-extensions-except=${distDir}`,
    `--load-extension=${distDir}`,
    '--mute-audio',
    '--autoplay-policy=no-user-gesture-required',
    '--window-position=40,40',
  ],
});

try {
  const sw = ctx.serviceWorkers()[0] ?? await ctx.waitForEvent('serviceworker', { timeout: 15000 });
  const extId = new URL(sw.url()).host;
  const localServerUrl = 'http://127.0.0.1:8000';
  // serverUrl 기본값이 프로드로 바뀐 뒤로는(host-permissions.ts) 여기서 로컬을 명시하고
  // optional_host_permissions도 실제 흐름으로 부여해야 "서버 싱크에 저장된 발음/번역"
  // 전제(로컬 DB)가 성립한다.
  await ensureLocalServerPermissionForServerUrl(ctx, sw, extId, localServerUrl);
  // 사용자 설정 재현: 번역 표시 ON (문제가 발생했던 조건)
  await sw.evaluate(s => chrome.storage.local.set({ settings: s }),
    { showTranslation: true, translationLanguage: 'ko', serverUrl: localServerUrl });

  const page = ctx.pages()[0] ?? await ctx.newPage();
  await page.goto(videoUrl, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForSelector('#everyric-root', { state: 'attached', timeout: 30000 });

  await page.waitForFunction(() => {
    const root = document.getElementById('everyric-root')?.shadowRoot;
    return (root?.querySelectorAll('.ey-line:not(.ey-line-plain)').length ?? 0) > 0;
  }, null, { timeout: 45000, polling: 1000 });
  await page.waitForTimeout(2500); // 번역 로드가 시도된다면 그 뒤 상태를 봐야 한다

  const panel = await page.evaluate(() => {
    const root = document.getElementById('everyric-root')?.shadowRoot;
    const text = root?.textContent ?? '';
    return {
      lines: root?.querySelectorAll('.ey-line').length ?? 0,
      pron: root?.querySelectorAll('.ey-line-pron').length ?? 0,
      tr: root?.querySelectorAll('.ey-line-tr').length ?? 0,
      noApiKey: text.includes('[NO API KEY]'),
      firstPron: root?.querySelector('.ey-line-pron')?.textContent?.slice(0, 40) ?? '',
    };
  });
  check(panel.pron > 40, `새 프로필에서 발음 표기 (서버 저장분, ${panel.pron}줄)`, panel);
  check(panel.tr > 40, `새 프로필에서 사람 번역 (${panel.tr}줄)`);
  check(!panel.noApiKey, '[NO API KEY] 없음');

  // 재생 보장 후 PiP 열기
  await page.evaluate(() => {
    const v = document.querySelector('video.html5-main-video') ?? document.querySelector('video');
    if (v) { v.currentTime = 30; void v.play().catch(() => {}); }
  });
  await page.waitForTimeout(1500);
  await page.locator('[title="PiP 창으로 보기"]').click();
  await page.waitForTimeout(3000);

  // PiP 안 가사 UI는 메인 창과 같은 인스턴스다 — 읽는 법은 lib/pip-panel.mjs 한 곳에 있다
  const pipRaw = await page.evaluate(readPipPanel());
  const pip = {
    open: pipRaw.open,
    pron: pipRaw.pron ?? '',
    pitch: pipRaw.lane ?? { present: false, visible: false, drawnPx: 0 },
  };
  check(pip.open, 'PiP 열림');
  check(pip.pitch.present && pip.pitch.visible && pip.pitch.drawnPx > 50, '음정 바 (신규 UI) 렌더링', pip.pitch);
  if (pip.pron) console.log('PASS: PiP 발음 =', JSON.stringify(pip.pron));

  const pipPage = ctx.pages().find(p => p !== page);
  if (pipPage) {
    await pipPage.screenshot({ path: resolve(__dirname, '../fresh-pip.png') });
    console.log('screenshot: fresh-pip.png');
  }
  // PiP 닫기 — pipKeepPanel 기본값이 2026-08 이후 false라(운영자 지시, settings.ts:103),
  // PiP를 열면 메인 패널이 "패널로 되돌리기" 플레이스홀더로 접히고 헤더의 PiP 아이콘은
  // stateKind!=='synced'라 의도적으로 숨는다(pip-dual-instance-architecture 메모). 헤더
  // 아이콘 재클릭이 아니라 플레이스홀더 자체의 버튼으로 닫아야 실제 UX와 맞는다.
  const placeholder = await page.evaluate(() => {
    const sr = document.getElementById('everyric-root')?.shadowRoot;
    const btn = sr?.querySelector('.ey-state button');
    return { present: !!btn, text: btn?.textContent?.trim() ?? '' };
  });
  check(placeholder.present, '메인 패널이 PiP 플레이스홀더("패널로 되돌리기")로 접힘', placeholder);
  if (placeholder.present) {
    await page.locator('.ey-state button').click();
  } else {
    // pipKeepPanel이 true로 바뀌어 있는 등 플레이스홀더가 없는 경우의 폴백 — 헤더 아이콘이
    // 여전히 보일 것이므로 옛 경로로 닫는다.
    await page.locator('[title="PiP 창으로 보기"]').click();
  }
  await page.waitForTimeout(1000);

  // pitchGuide OFF → 리로드 → PiP에서 음정 바 숨김 확인
  // storage.local.set({settings: s})는 병합이 아니라 통째로 교체다 — serverUrl을 다시
  // 안 넣으면 getSettings()가 DEFAULT_SETTINGS(프로드)로 되돌린다.
  await sw.evaluate(s => chrome.storage.local.set({ settings: s }),
    { showTranslation: true, translationLanguage: 'ko', pitchGuide: false, serverUrl: localServerUrl });
  await page.reload({ waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => {
    const root = document.getElementById('everyric-root')?.shadowRoot;
    return (root?.querySelectorAll('.ey-line').length ?? 0) > 0;
  }, null, { timeout: 45000, polling: 1000 });
  await page.evaluate(() => {
    const v = document.querySelector('video.html5-main-video') ?? document.querySelector('video');
    if (v) { v.currentTime = 30; void v.play().catch(() => {}); }
  });
  await page.waitForTimeout(1000);
  await page.locator('[title="PiP 창으로 보기"]').click();
  await page.waitForTimeout(2500);
  const offRaw = await page.evaluate(readPipPanel());
  const off = { open: offRaw.open, pitchVisible: offRaw.lane?.visible ?? false };
  check(off.open && !off.pitchVisible, 'pitchGuide OFF → 음정 바 숨김', off);

  console.log(failed ? 'FRESH PROFILE CHECK: FAIL' : 'FRESH PROFILE CHECK: PASS');
  process.exitCode = failed ? 1 : 0;
} catch (e) {
  console.log('FRESH PROFILE CHECK: ERROR —', String(e).slice(0, 300));
  process.exitCode = 1;
} finally {
  await ctx.close();
}
