// 실서버(:8000) + 실제 YouTube 영상으로 전체 사용자 여정을 검증하는 E2E.
//
//   빈 상태(가사 없음) → "다시 검색"에 한국어 곡명 입력 → 보카로 가사 위키 히트(원문+번역)
//   → "싱크 생성"(위키 가사) → 실서버 yt-dlp 다운로드 + CTC 정렬 + FCPE 멜로디
//   → 싱크 가사 + 단어 하이라이트 → PiP 음정 바 픽셀 검증
//
// 사전 조건: 실서버가 :8000에 떠 있어야 한다 (mock-server 아님!).
//   .venv\Scripts\python.exe -m uvicorn everyric2.server.main:app --port 8000
// 주소는 반드시 127.0.0.1 — 이 개발 머신은 localhost가 IPv6 폴백 스톨로 요청당 2초를
// 먹는다. 확장 기본 serverUrl이 프로드로 바뀐 뒤로는 설정에 로컬 주소를 명시해야 한다.
// 실행: node scripts/real-e2e.mjs [videoUrl] [koreanTitle]
//   기본: 로키 공식 MV (Xg-qfsKN2_E) + "로키" — LRCLIB 미수록이라 전체 폴백 체인이 재현됨
import { chromium } from 'playwright';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import { mkdtempSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { ensureLocalServerPermissionForServerUrl } from './lib/local-server-permission.mjs';
import { readPipPanel } from './lib/pip-panel.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const distDir = resolve(__dirname, '../dist');
const videoUrl = process.argv[2] ?? 'https://www.youtube.com/watch?v=Xg-qfsKN2_E';
const koreanTitle = process.argv[3] ?? '로키';
const SYNC_TIMEOUT_MS = 12 * 60 * 1000; // 첫 잡은 모델 로드 포함 — CPU에서 수 분 걸린다

let failed = false;
function check(ok, label, detail) {
  console.log(`${ok ? 'PASS' : 'FAIL'}: ${label}${detail !== undefined ? ' = ' + JSON.stringify(detail) : ''}`);
  if (!ok) failed = true;
  return ok;
}

// 0) 실서버 확인 — mock이 아니라 실제 FastAPI인지 /health 응답 형태로 구분
try {
  const health = await (await fetch('http://127.0.0.1:8000/health', { signal: AbortSignal.timeout(3000) })).json();
  if (!check(health.status === 'healthy' && 'gpu_available' in health, 'real server /health', health)) process.exit(1);
} catch (e) {
  console.log('FAIL: server not reachable on :8000 —', String(e).slice(0, 120));
  process.exit(1);
}

const userDataDir = mkdtempSync(join(tmpdir(), 'everyric-e2e-'));
// Chrome 137+는 --load-extension 무시. msedge는 사용자의 실사용 Edge가 떠 있으면 기동이
// 179초+ 행에 걸린다(2026-07-28 실측) — 기본은 번들 Chromium(오픈 빌드라 확장 로드 지원).
// 특정 채널을 원하면 EVERYRIC_E2E_CHANNEL=msedge 처럼 지정한다.
const channel = process.env.EVERYRIC_E2E_CHANNEL ?? '';
const ctx = await chromium.launchPersistentContext(userDataDir, {
  // Playwright 신버전은 --disable-extensions-except를 줘도 기본 --disable-extensions를
  // 빼주지 않는다(구버전 특례 삭제) — 이게 남으면 --load-extension이 조용히 무시된다.
  ignoreDefaultArgs: ['--disable-extensions'],
  ...(channel ? { channel } : {}),
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

// shadowRoot 내부를 evaluate로 다루기 위한 공용 헬퍼 (페이지 컨텍스트에서 실행됨)
const SR = `document.getElementById('everyric-root')?.shadowRoot`;

try {
  const sw = ctx.serviceWorkers()[0] ?? await ctx.waitForEvent('serviceworker', { timeout: 15000 });
  console.log('extension loaded:', sw.url());
  const extId = new URL(sw.url()).host;
  const localServerUrl = 'http://127.0.0.1:8000';
  // 로컬 서버는 optional_host_permissions라 설치 시 자동으로는 안 붙는다 — 옵션 페이지의
  // 실제 "허용" 흐름을 재현해 부여한다(scripts/lib/local-server-permission.mjs 참고).
  await ensureLocalServerPermissionForServerUrl(ctx, sw, extId, localServerUrl);
  await sw.evaluate(s => chrome.storage.local.set({ settings: s }), {
    debugInfo: true,
    serverUrl: localServerUrl,
    ...(process.env.EVERYRIC_E2E_SETTINGS ? JSON.parse(process.env.EVERYRIC_E2E_SETTINGS) : {}),
  });

  const page = ctx.pages()[0] ?? await ctx.newPage();
  page.on('console', msg => {
    if (msg.type() === 'error') console.log('[console:error]', msg.text().slice(0, 200));
  });

  console.log('navigating to', videoUrl);
  await page.goto(videoUrl, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForSelector('#everyric-root', { state: 'attached', timeout: 30000 });

  // 1) 자동 검색 결과 대기 — 이 곡은 서버/LRCLIB/vocaro(일본어 제목) 전부 미스여야 한다
  const state1 = await page.waitForFunction(sr => {
    const root = document.getElementById('everyric-root')?.shadowRoot;
    if (!root) return null;
    const lines = root.querySelectorAll('.ey-line').length;
    const stateText = root.querySelector('.ey-state-text')?.textContent ?? '';
    if (lines > 0 || (stateText && !stateText.includes('검색'))) return { lines, stateText };
    return null;
  }, SR, { timeout: 60000, polling: 1000 }).then(h => h.jsonValue());
  console.log('initial search state =', JSON.stringify(state1));
  const emptyStateOk = check(state1.stateText.includes('가사를 찾지 못했어요'), 'empty state (일본어 원제 → 전 소스 미스)', state1);
  // 이 전제(빈 상태 → "다시 검색" UI로 재검색)가 깨지면 아래 2)~7)은 애초에 이 곡으로
  // 검증하려던 폴백 체인이 아니다 — 계속 진행하면 검색폼이 없는 상태에서 입력창을 건드려
  // "Cannot set properties of undefined"라는 무관한 위치의 크래시로 실패 지점이 가려진다.
  // 실패 자체는 위 check()가 이미 기록했으니, 여기서 원인이 분명한 메시지로 멈춘다.
  if (!emptyStateOk) {
    throw new Error(
      `사전 조건 불성립: 초기 상태가 "빈 상태"가 아니라 이미 ${state1.lines}줄이 렌더됨 — ` +
      '이 비디오가 서버/LRCLIB/vocaro 중 하나에서 이미 매치된 것으로 보인다(하네스가 아니라 ' +
      '데이터/제품 쪽 변화일 가능성). 2)~7) 단계(재검색→싱크 생성)는 이 전제 위에서만 의미가 있어 건너뜀.',
    );
  }

  // 2) "다시 검색"에 한국어 곡명 입력 → 보카로 위키 히트
  await page.evaluate(([sel, title]) => {
    const root = document.getElementById('everyric-root')?.shadowRoot;
    const inputs = root.querySelectorAll('.ey-search-form .ey-input');
    inputs[0].value = title;
    inputs[1].value = ''; // 아티스트 비움 — 위키는 곡명으로만 찾는다
    const btn = Array.from(root.querySelectorAll('.ey-search-form button')).find(b => b.textContent === '다시 검색');
    btn.click();
  }, [SR, koreanTitle]);
  console.log(`retry search with "${koreanTitle}"`);

  const vocaro = await page.waitForFunction(() => {
    const root = document.getElementById('everyric-root')?.shadowRoot;
    if (!root) return null;
    const plain = root.querySelectorAll('.ey-line-plain').length;
    if (plain === 0) return null;
    return {
      plainLines: plain,
      translations: root.querySelectorAll('.ey-line-tr').length,
      pronunciations: root.querySelectorAll('.ey-line-pron').length,
      banner: root.querySelector('.ey-banner')?.textContent?.slice(0, 40) ?? root.textContent.includes('타임싱크가 없는 가사예요'),
      source: root.querySelector('.ey-source')?.textContent ?? '',
      sourceIsLink: root.querySelector('.ey-source')?.classList.contains('link') ?? false,
      generateBtn: Array.from(root.querySelectorAll('.ey-generate-btn')).some(b => b.textContent.includes('싱크 생성')),
      firstLine: root.querySelector('.ey-line-plain')?.childNodes[0]?.textContent?.slice(0, 40) ?? '',
    };
  }, null, { timeout: 45000, polling: 1000 }).then(h => h.jsonValue());
  check(vocaro.plainLines > 10, 'vocaro 가사 로드', vocaro);
  check(vocaro.translations > 10, `vocaro 번역 포함 (${vocaro.translations}줄)`);
  check(vocaro.pronunciations > 10, `vocaro 발음 표기 포함 (${vocaro.pronunciations}줄)`);
  check(vocaro.source.includes('보카로'), 'vocaro 출처 배지', vocaro.source);
  check(vocaro.sourceIsLink, 'vocaro 출처 링크(CC BY 표기)');
  await page.screenshot({ path: resolve(__dirname, '../e2e-1-vocaro.png') });
  console.log('screenshot: e2e-1-vocaro.png');

  if (!vocaro.generateBtn) throw new Error('싱크 생성 버튼이 없음 — 서버 health가 안 잡혔을 수 있음');

  // 3) 위키 가사로 싱크 생성 — 실서버가 다운로드+정렬+멜로디까지 수행
  const t0 = Date.now();
  await page.evaluate(() => {
    const root = document.getElementById('everyric-root')?.shadowRoot;
    Array.from(root.querySelectorAll('.ey-generate-btn')).find(b => b.textContent.includes('싱크 생성')).click();
  });
  console.log('generate clicked — waiting for real server job (download + CTC + FCPE)...');

  // 진행 표시 확인 (비치명 — 빠르게 지나갈 수 있음)
  try {
    await page.waitForFunction(() => {
      const root = document.getElementById('everyric-root')?.shadowRoot;
      return /싱크 생성 중|대기열/.test(root?.textContent ?? '');
    }, null, { timeout: 20000, polling: 500 });
    console.log('PASS: generating progress UI 표시됨');
  } catch { console.log('WARN: 진행 UI를 못 봄 (이미 완료됐을 수 있음)'); }

  // 4) 완료 대기 — 디버그 스트립의 src=everyric + 동기화 라인 등장
  const synced = await page.waitForFunction(() => {
    const root = document.getElementById('everyric-root')?.shadowRoot;
    if (!root) return null;
    const dbg = root.querySelector('.ey-debug')?.textContent ?? '';
    const lines = root.querySelectorAll('.ey-line:not(.ey-line-plain)').length;
    if (/src=everyric/.test(dbg) && lines > 0) {
      return {
        lines,
        words: root.querySelectorAll('.ey-word').length,
        pron: root.querySelectorAll('.ey-line-pron').length,
        tr: root.querySelectorAll('.ey-line-tr').length,
        dbg: dbg.slice(0, 120),
      };
    }
    // 생성 진행 문구가 아닌 상태 텍스트가 뜨면 실패로 판단 (showError는 서버 에러를 그대로 표시)
    const st = root.querySelector('.ey-state-text')?.textContent ?? '';
    if (st && !/싱크 생성 중|대기열|검색/.test(st)) return { error: st };
    return null;
  }, null, { timeout: SYNC_TIMEOUT_MS, polling: 2000 }).then(h => h.jsonValue());
  if (synced.error) throw new Error('싱크 생성 실패: ' + synced.error);
  const elapsed = Math.round((Date.now() - t0) / 1000);
  check(synced.lines > 10, `싱크 생성 완료 (${elapsed}s, 서버 실처리)`, synced);
  check(synced.words > 0, `단어(카라오케) 스팬 렌더링 (${synced.words}개)`);
  check(synced.pron > 10, `싱크 후 발음 표기 병합 유지 (${synced.pron}줄)`);
  check(synced.tr > 10, `싱크 후 위키 번역 병합 유지 (${synced.tr}줄)`);

  // 5) 하이라이트 — 생성 동안 영상이 끝났을 수 있으므로 앞부분으로 되감고 재생
  await page.evaluate(() => {
    const v = document.querySelector('video.html5-main-video') ?? document.querySelector('video');
    if (v) { v.currentTime = 25; void v.play().catch(() => {}); }
  });
  await page.waitForFunction(() => {
    const root = document.getElementById('everyric-root')?.shadowRoot;
    return !!root?.querySelector('.ey-line.active');
  }, null, { timeout: 60000, polling: 500 });
  const active = await page.evaluate(() => {
    const root = document.getElementById('everyric-root')?.shadowRoot;
    return root?.querySelector('.ey-line.active')?.textContent?.slice(0, 60) ?? null;
  });
  check(!!active, '재생 위치 하이라이트', active);
  await page.screenshot({ path: resolve(__dirname, '../e2e-2-synced.png') });
  console.log('screenshot: e2e-2-synced.png');

  // 6) PiP + 음정 바 (FCPE notes가 실제로 그려지는지)
  await page.locator('[title="PiP 창으로 보기"]').click();
  await page.waitForTimeout(3000);
  // PiP 안 가사 UI는 메인 창과 같은 인스턴스다 — 읽는 법은 lib/pip-panel.mjs 한 곳에 있다
  const pipRaw = await page.evaluate(readPipPanel());
  const pip = {
    open: pipRaw.open,
    currentLine: pipRaw.currentLine ?? null,
    pron: pipRaw.pron ?? '',
    pitch: pipRaw.lane ?? { present: false, visible: false, drawnPx: 0 },
  };
  check(pip.open, 'PiP 열림', pip.currentLine);
  if (pip.pron) console.log('PASS: PiP 발음 표기 =', JSON.stringify(pip.pron));
  else console.log('WARN: 현재 라인에 발음 표기 없음 (해당 라인이 발음 없는 라인일 수 있음)');
  check(pip.pitch.present && pip.pitch.visible && pip.pitch.drawnPx > 50, '가라오케 음정 바 (실제 FCPE notes)', pip.pitch);
  try {
    const pipPage = ctx.pages().find(p => p !== page);
    if (pipPage) {
      await pipPage.screenshot({ path: resolve(__dirname, '../e2e-3-pip.png') });
      console.log('screenshot: e2e-3-pip.png');

      // 7) 가로로 넓은(낮은) 창에서도 가사 스테이지가 짓눌리지 않는지 (레이아웃 회귀 검증)
      await pipPage.setViewportSize({ width: 1100, height: 330 });
      await pipPage.waitForTimeout(600);
      const wide = await pipPage.evaluate(() => {
        // 가사 영역 = 패널 인스턴스의 호스트(예전 .ey-pip-stage 자리)
        const root = document.getElementById('everyric-root');
        const video = document.querySelector('.ey-pip-video');
        const cur = root?.shadowRoot?.querySelector('.ey-line.active');
        return {
          stageH: root?.clientHeight ?? 0,
          pitchH: root?.shadowRoot?.querySelector('.ey-lane-wrap')?.clientHeight ?? 0,
          videoH: video?.clientHeight ?? 0,
          curFontPx: cur ? parseFloat(getComputedStyle(cur).fontSize) : 0,
          winH: window.innerHeight,
        };
      });
      // 가라오케 레인이 켜지면 스테이지는 의도적으로 숨겨진다(중복 제거) — 가사 영역 = 스테이지 or 레인
      const lyricsH = wide.stageH > 0 ? wide.stageH : wide.pitchH;
      check(lyricsH >= 90, `넓은 창(1100x330)에서 가사영역 확보 (stage=${wide.stageH}/lane=${wide.pitchH})`, wide);
      await pipPage.screenshot({ path: resolve(__dirname, '../e2e-4-pip-wide.png') });
      console.log('screenshot: e2e-4-pip-wide.png');
    }
  } catch (e) { console.log('WARN: pip wide-resize check skipped —', String(e).slice(0, 120)); }
  await page.locator('[title="PiP 창으로 보기"]').click();
  await page.waitForTimeout(1000);

  console.log(failed ? 'REAL E2E: FAIL' : 'REAL E2E: PASS (vocaro → 싱크 생성(실서버) → 하이라이트 → 음정 바)');
  process.exitCode = failed ? 1 : 0;
} catch (e) {
  console.log('REAL E2E: ERROR —', String(e).slice(0, 400));
  try {
    const page = ctx.pages()[0];
    if (page) await page.screenshot({ path: resolve(__dirname, '../e2e-error.png') });
  } catch { /* ignore */ }
  process.exitCode = 1;
} finally {
  await ctx.close();
}
