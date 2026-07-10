// dist를 실제 Chrome에 로드해 YouTube에서 오버레이가 뜨는지 확인하는 스모크 테스트.
// 실행: node scripts/smoke-test.mjs [videoUrl]
import { chromium } from 'playwright';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import { mkdtempSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const distDir = resolve(__dirname, '../dist');
const videoUrl = process.argv[2] ?? 'https://www.youtube.com/watch?v=dQw4w9WgXcQ';
const userDataDir = mkdtempSync(join(tmpdir(), 'everyric-smoke-'));

// 주의: 브랜드 Chrome은 137+에서 --load-extension을 무시함.
// 기본은 번들 Chromium, EVERYRIC_SMOKE_CHANNEL=msedge 등으로 대체 가능.
const channel = process.env.EVERYRIC_SMOKE_CHANNEL;
const ctx = await chromium.launchPersistentContext(userDataDir, {
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

try {
  // 확장 SW가 등록됐는지부터 확인 (로드 실패 시 빠르게 실패)
  const sw = ctx.serviceWorkers()[0] ?? await ctx.waitForEvent('serviceworker', { timeout: 15000 });
  console.log('PASS: extension service worker =', sw.url());

  // 저장값은 확장의 getSettings()에서 DEFAULT_SETTINGS와 병합되므로 부분 주입으로 충분
  const injected = {};
  // EVERYRIC_SMOKE_TRANSLATE=1 → 번역 표시 켜고 시작 (mock-server.mjs가 :8000에 떠 있어야 함)
  if (process.env.EVERYRIC_SMOKE_TRANSLATE === '1') {
    Object.assign(injected, { showTranslation: true, translationLanguage: 'ko' });
  }
  // EVERYRIC_SMOKE_DEBUG=1 → 디버그 정보 스트립 켜고 시작
  if (process.env.EVERYRIC_SMOKE_DEBUG === '1') injected.debugInfo = true;
  if (Object.keys(injected).length > 0) {
    await sw.evaluate(s => chrome.storage.local.set({ settings: s }), injected);
    console.log('settings injected:', JSON.stringify(injected));
  }

  // EVERYRIC_SMOKE_VOCARO=1 → 보카로 위키 조회를 SW 컨텍스트에서 직접 검증
  // (슬러그 직접 추측 경로 + 한글 인덱스 매칭 경로 모두 실제 네트워크로 확인)
  if (process.env.EVERYRIC_SMOKE_VOCARO === '1') {
    for (const [label, query] of [['slug', 'Vampire'], ['index', '다이아몬드']]) {
      const r = await sw.evaluate(t => globalThis.__vocaroLookup(t), query);
      const ok = r && r.lines.length > 0 && r.lines.some(l => l.translation);
      console.log(`${ok ? 'PASS' : 'FAIL'}: vocaro ${label} lookup("${query}") =`, JSON.stringify({
        url: r?.pageUrl, title: r?.pageTitle, lines: r?.lines.length, first: r?.lines[0],
      }));
      if (!ok) process.exitCode = 1;
    }
  }

  const page = ctx.pages()[0] ?? await ctx.newPage();
  page.on('pageerror', err => console.log('[pageerror]', String(err).slice(0, 300)));
  page.on('console', msg => {
    const text = msg.text();
    if (text.includes('Everyric') || msg.type() === 'error') console.log(`[console:${msg.type()}]`, text.slice(0, 300));
  });

  console.log('navigating to', videoUrl);
  await page.goto(videoUrl, { waitUntil: 'domcontentloaded', timeout: 60000 });

  await page.waitForSelector('#everyric-root', { state: 'attached', timeout: 30000 });
  console.log('PASS: #everyric-root attached');

  // 가사 로드까지 대기 (LRCLIB 응답 시간 포함)
  const result = await page.waitForFunction(() => {
    const sr = document.getElementById('everyric-root')?.shadowRoot;
    if (!sr) return null;
    const lines = sr.querySelectorAll('.ey-line').length;
    const stateText = sr.querySelector('.ey-state-text')?.textContent ?? '';
    const title = sr.querySelector('.ey-song-title')?.textContent ?? '';
    if (lines > 0 || (stateText && !stateText.includes('검색 중'))) {
      return { lines, stateText, title };
    }
    return null;
  }, { timeout: 45000, polling: 1000 });

  const info = await result.jsonValue();
  console.log('PASS: overlay state =', JSON.stringify(info));

  // 하이라이트가 실제로 걸릴 때까지 대기 (전주가 긴 곡은 첫 가사까지 20초+)
  try {
    await page.waitForFunction(() => {
      const sr = document.getElementById('everyric-root')?.shadowRoot;
      return !!sr?.querySelector('.ey-line.active');
    }, { timeout: 40000, polling: 500 });
  } catch { /* 미등장 시 아래 sync check가 FAIL/PARTIAL로 판정 */ }
  const active = await page.evaluate(() => {
    const sr = document.getElementById('everyric-root')?.shadowRoot;
    return {
      activeLine: sr?.querySelector('.ey-line.active')?.textContent?.slice(0, 80) ?? null,
      lineCount: sr?.querySelectorAll('.ey-line').length ?? 0,
      footerVisible: (sr?.querySelector('.ey-footer')instanceof HTMLElement) && sr.querySelector('.ey-footer').style.display !== 'none',
    };
  });
  console.log('sync check =', JSON.stringify(active));

  // 디버그 모드면 스트립이 보이고 vid=/t= 값이 채워지는지 확인
  if (process.env.EVERYRIC_SMOKE_DEBUG === '1') {
    const dbg = await page.evaluate(() => {
      const el = document.getElementById('everyric-root')?.shadowRoot?.querySelector('.ey-debug');
      if (!(el instanceof HTMLElement)) return null;
      return { visible: getComputedStyle(el).display !== 'none', text: el.textContent };
    });
    const dbgOk = dbg?.visible && /vid=\S+/.test(dbg.text ?? '') && /t=[\d.]+s/.test(dbg.text ?? '');
    console.log(`${dbgOk ? 'PASS' : 'FAIL'}: debug strip =`, JSON.stringify(dbg));
    if (!dbgOk) process.exitCode = 1;
  }

  // 번역 모드면 번역 라인이 붙을 때까지 대기
  if (process.env.EVERYRIC_SMOKE_TRANSLATE === '1') {
    try {
      await page.waitForFunction(() => {
        const sr = document.getElementById('everyric-root')?.shadowRoot;
        return (sr?.querySelectorAll('.ey-line-tr').length ?? 0) > 0;
      }, { timeout: 30000, polling: 1000 });
      const trCount = await page.evaluate(() =>
        document.getElementById('everyric-root')?.shadowRoot?.querySelectorAll('.ey-line-tr').length ?? 0);
      console.log(`PASS: translations rendered (${trCount} lines)`);
    } catch {
      console.log('WARN: translations did not render (mock server running?)');
    }
  }

  const shot = resolve(__dirname, '../smoke-result.png');
  await page.screenshot({ path: shot });
  console.log('screenshot saved:', shot);

  // PiP 테스트 — playwright 클릭은 실제 user gesture이며 open shadow DOM을 관통한다
  // 기본 설정(pipKeepPanel=true)에서는 PiP를 열어도 패널 가사가 유지되어야 한다.
  let pipResult = 'skipped';
  const pipBtn = page.locator('[title="PiP 창으로 보기"]');
  if (await pipBtn.count() > 0 && await pipBtn.isVisible()) {
    await pipBtn.click();
    await page.waitForTimeout(2500);
    const pipState = await page.evaluate(() => {
      const w = window.documentPictureInPicture?.window;
      const sr = document.getElementById('everyric-root')?.shadowRoot;
      const videoWrap = w?.document.querySelector('.ey-pip-video');
      // 음정 바: canvas 존재/표시 여부 + 실제로 그려진 픽셀 수 (알파 > 0)
      let pitch = { present: false, visible: false, drawnPx: 0 };
      const pitchCanvas = w?.document.querySelector('.ey-pip-pitch');
      if (pitchCanvas) {
        pitch.present = true;
        pitch.visible = w.getComputedStyle(pitchCanvas).display !== 'none';
        try {
          const data = pitchCanvas.getContext('2d')
            .getImageData(0, 0, pitchCanvas.width || 1, pitchCanvas.height || 1).data;
          for (let i = 3; i < data.length; i += 4) if (data[i] > 0) pitch.drawnPx++;
        } catch { /* 크기 0 등 — drawnPx 0 유지 */ }
      }
      return {
        open: !!w,
        pipCurrentLine: w?.document.querySelector('.ey-pip-line.current')?.textContent?.slice(0, 60) ?? null,
        panelLineCount: sr?.querySelectorAll('.ey-line').length ?? 0,
        panelActiveLine: sr?.querySelector('.ey-line.active')?.textContent?.slice(0, 60) ?? null,
        videoMirror: videoWrap ? w.getComputedStyle(videoWrap).display !== 'none' : false,
        hasVolume: !!w?.document.querySelector('.ey-pip-volume'),
        hasDivider: !!w?.document.querySelector('.ey-pip-divider'),
        volumeValue: w?.document.querySelector('.ey-pip-volume')?.value ?? null,
        pitch,
      };
    });
    console.log('pip check (both visible) =', JSON.stringify(pipState));

    // EVERYRIC_SMOKE_PITCH=1 → 음정 바가 실제로 그려졌는지 검증
    // (mock-server를 EVERYRIC_MOCK_SYNC=<notes 포함 timestamps.json>으로 띄워야 함)
    if (process.env.EVERYRIC_SMOKE_PITCH === '1') {
      const p = pipState.pitch;
      const ok = p.present && p.visible && p.drawnPx > 50;
      console.log(`${ok ? 'PASS' : 'FAIL'}: pitch bar =`, JSON.stringify(p));
      if (!ok) process.exitCode = 1;
    } else if (pipState.pitch.visible) {
      console.log('FAIL: notes 없는 곡인데 음정 바가 표시됨');
      process.exitCode = 1;
    }

    // Document PiP 창이 Playwright 페이지로 노출되면 스크린샷 (안 되면 건너뜀)
    try {
      const pipPage = ctx.pages().find(p => p !== page);
      if (pipPage) {
        const pipShot = resolve(__dirname, '../smoke-pip.png');
        await pipPage.screenshot({ path: pipShot });
        console.log('pip screenshot saved:', pipShot);
      } else {
        console.log('pip window not exposed as page — screenshot skipped');
      }
    } catch (e) {
      console.log('pip screenshot failed:', String(e).slice(0, 120));
    }

    // 헤더 PiP 버튼 재클릭으로 닫기 → 패널 가사는 그대로 유지
    await pipBtn.click();
    await page.waitForTimeout(1500);
    const restored = await page.evaluate(() => {
      const sr = document.getElementById('everyric-root')?.shadowRoot;
      return {
        pipStillOpen: !!window.documentPictureInPicture?.window,
        lineCount: sr?.querySelectorAll('.ey-line').length ?? 0,
      };
    });
    console.log('pip close =', JSON.stringify(restored));
    pipResult = pipState.open && pipState.pipCurrentLine && pipState.panelLineCount > 0
      && !restored.pipStillOpen && restored.lineCount > 0 ? 'PASS' : 'FAIL';
  }

  if (active.lineCount > 0 && active.activeLine && pipResult !== 'FAIL') {
    console.log(`SMOKE TEST: PASS (lyrics+highlight OK, pip=${pipResult})`);
  } else if (info.lines > 0 || info.stateText) {
    console.log(`SMOKE TEST: PARTIAL (overlay OK, highlight=${!!active.activeLine}, pip=${pipResult})`);
    process.exitCode = pipResult === 'FAIL' ? 1 : 0;
  } else {
    console.log('SMOKE TEST: FAIL');
    process.exitCode = 1;
  }
} finally {
  await ctx.close();
}
